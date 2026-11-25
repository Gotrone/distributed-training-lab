"""DistilBERT fine-tuning on SQuAD with PyTorch Lightning (DDP).

Usage (single node, 2 GPUs):  python train_ddp.py --nodes 1 --devices 2
Usage (SLURM, 2 nodes x 2 GPUs): sbatch bash.sh

Each process sees a different shard of the training set (DistributedSampler is
injected by Lightning). The batch size is per GPU, so the global batch size is
batch_size x devices x nodes.
"""
import argparse
import collections
import json
import time

import evaluate
import lightning as L
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DistilBertForQuestionAnswering,
    default_data_collator,
)

CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 512
STRIDE = 128


def train_preprocess(examples, tokenizer):
    """Tokenize and compute the start/end token positions of the answer."""
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    start_positions, end_positions = [], []
    for i, offsets in enumerate(inputs["offset_mapping"]):
        answer = examples["answers"][inputs["overflow_to_sample_mapping"][i]]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        ctx_start = sequence_ids.index(1)
        ctx_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if offsets[ctx_start][0] > start_char or offsets[ctx_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = ctx_start
            while idx <= ctx_end and offsets[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            idx = ctx_end
            while idx >= ctx_start and offsets[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs.pop("overflow_to_sample_mapping")
    inputs.pop("offset_mapping")
    return inputs


def validation_preprocess(examples, tokenizer):
    """Tokenize and keep offsets (context tokens only) to map logits back to text."""
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    inputs["base_id"] = [examples["id"][s] for s in sample_map]
    for i, offsets in enumerate(inputs["offset_mapping"]):
        sequence_ids = inputs.sequence_ids(i)
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offsets)
        ]
    return inputs


def postprocess(start_logits, end_logits, features, examples, n_best=20, max_answer_length=30):
    """Pick the best answer span per example and compute Exact Match / F1."""
    example_to_features = collections.defaultdict(list)
    for idx, base_id in enumerate(features["base_id"]):
        example_to_features[base_id].append(idx)

    predictions = []
    for example in examples:
        candidates = []
        for fi in example_to_features[example["id"]]:
            offsets = features["offset_mapping"][fi]
            starts = np.argsort(start_logits[fi])[::-1][:n_best]
            ends = np.argsort(end_logits[fi])[::-1][:n_best]
            for s in starts:
                for e in ends:
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_length:
                        continue
                    candidates.append(
                        (
                            start_logits[fi][s] + end_logits[fi][e],
                            example["context"][offsets[s][0] : offsets[e][1]],
                        )
                    )
        text = max(candidates)[1] if candidates else ""
        predictions.append({"id": example["id"], "prediction_text": text})

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return evaluate.load("squad").compute(predictions=predictions, references=references)


class SquadDataModule(L.LightningDataModule):
    def __init__(self, train_samples, val_samples, batch_size):
        super().__init__()
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def prepare_data(self):
        # Download once per node (Lightning runs this on local rank 0 only).
        load_dataset("squad")
        AutoTokenizer.from_pretrained(CHECKPOINT)
        DistilBertForQuestionAnswering.from_pretrained(CHECKPOINT)

    def setup(self, stage=None):
        raw = load_dataset("squad")
        self.val_examples = raw["validation"].select(range(self.val_samples))
        train_raw = raw["train"].select(range(self.train_samples))

        train = train_raw.map(
            lambda x: train_preprocess(x, self.tokenizer),
            batched=True,
            remove_columns=train_raw.column_names,
        )
        self.train_set = train.with_format("torch")

        self.val_features = self.val_examples.map(
            lambda x: validation_preprocess(x, self.tokenizer),
            batched=True,
            remove_columns=self.val_examples.column_names,
        )
        self.val_set = self.val_features.with_format(
            "torch", columns=["input_ids", "attention_mask"]
        )

    def train_dataloader(self):
        # Lightning replaces the sampler by a DistributedSampler under DDP.
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, collate_fn=default_data_collator
        )


class QAModel(L.LightningModule):
    def __init__(self, data_module, lr=2e-5):
        super().__init__()
        self.model = DistilBertForQuestionAnswering.from_pretrained(CHECKPOINT)
        self.lr = lr
        self.dm = data_module
        self._start, self._end = [], []

    def training_step(self, batch, batch_idx):
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            start_positions=batch["start_positions"],
            end_positions=batch["end_positions"],
        )
        self.log("train_loss", out.loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        self._start.append(out.start_logits)
        self._end.append(out.end_logits)

    def _gather(self, chunks, n_features):
        local = torch.cat(chunks)  # (n_local, seq_len)
        gathered = self.all_gather(local)
        if gathered.dim() == 3:  # (world, n_local, seq_len) -> original order
            gathered = gathered.permute(1, 0, 2).reshape(-1, local.shape[-1])
        return gathered[:n_features].float().cpu().numpy()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self._start.clear()
            self._end.clear()
            return
        n = len(self.dm.val_features)
        start = self._gather(self._start, n)
        end = self._gather(self._end, n)
        self._start.clear()
        self._end.clear()
        if self.trainer.is_global_zero:
            m = postprocess(start, end, self.dm.val_features, self.dm.val_examples)
            self.last_metrics = m
            print(f"Epoch {self.current_epoch + 1}: EM={m['exact_match']:.1f} F1={m['f1']:.1f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


class Timer(L.Callback):
    def on_train_start(self, trainer, pl_module):
        self.t0 = time.time()

    def on_train_end(self, trainer, pl_module):
        self.total = time.time() - self.t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--devices", type=int, default=1, help="GPUs per node")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2, help="per GPU")
    p.add_argument("--train-samples", type=int, default=1000)
    p.add_argument("--val-samples", type=int, default=250)
    p.add_argument("--output", default="results.json")
    args = p.parse_args()

    L.seed_everything(42)
    dm = SquadDataModule(args.train_samples, args.val_samples, args.batch_size)
    model = QAModel(dm)
    timer = Timer()
    world = args.nodes * args.devices
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        strategy="ddp" if world > 1 else "auto",
        max_epochs=args.epochs,
        callbacks=[timer],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, datamodule=dm)

    if trainer.is_global_zero:
        result = {
            "nodes": args.nodes,
            "gpus_per_node": args.devices,
            "global_batch_size": args.batch_size * world,
            "epochs": args.epochs,
            "train_time_s": round(timer.total, 1),
            "exact_match": model.last_metrics["exact_match"],
            "f1": model.last_metrics["f1"],
        }
        print(json.dumps(result, indent=2))
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
