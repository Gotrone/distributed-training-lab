# Distributed version (PyTorch Lightning, DDP)

Multi-GPU, multi-node version of the single-GPU [baseline](../BASELINE): DistilBERT fine-tuning for extractive question answering on SQuAD.

## Status

The script is written but **has not been run on a cluster yet**, so no speedup is reported. The multi-node benchmark (2 nodes × 2 A100, via `bash.sh`) is the remaining step.

## Design

| Component (`train_ddp.py`) | Role |
|---|---|
| `SquadDataModule` | Downloads and tokenizes SQuAD (1,000 train / 250 validation questions by default). Lightning swaps in a `DistributedSampler`, so each process trains on its own shard. |
| `QAModel` | `LightningModule` around `DistilBertForQuestionAnswering` (AdamW, lr 2e-5). Validation logits are gathered from all processes and Exact Match / F1 are computed on rank 0. |
| `Timer` | Measures total training time, written with the metrics to `results.json`. |
| `bash.sh` | SLURM script: 2 nodes, 2 GPUs per node, one task per GPU, NCCL. |

The batch size is per GPU, so the global batch size grows with the number of GPUs (default 2 per GPU). Keep this in mind when comparing accuracy with the baseline.

## Run

```bash
pip install -r requirements.txt
python train_ddp.py --nodes 1 --devices 1     # single GPU, comparable to the baseline
python train_ddp.py --nodes 1 --devices 2     # one node, DDP on 2 GPUs
sbatch bash.sh                                # 2 nodes x 2 GPUs on SLURM
```

Compute nodes without internet need the dataset, model and `squad` metric cached beforehand (run `prepare_data` on a login node).
