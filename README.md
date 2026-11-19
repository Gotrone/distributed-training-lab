# Distributed Training Lab

Accelerating the fine-tuning of a Transformer model with **PyTorch**: a single-GPU baseline, then a multi-node, multi-GPU version with **PyTorch Lightning (DDP)** on a SLURM cluster. Lab from the "AI – HPC Tools" course, High Performance Computing master's programme, Universidade da Coruña (2024).

## Task

Fine-tune **DistilBERT** for extractive question answering on a subset of **SQuAD** (1,000 training / 250 validation samples, 5 epochs) and measure training time and accuracy across configurations.

## Contents

| Folder | Description |
|---|---|
| [`BASELINE/`](BASELINE) | Single-GPU fine-tuning (Hugging Face Transformers, PyTorch): data preprocessing, training loop, per-epoch evaluation (Exact Match, F1). |
| [`DISTRIBUTED/`](DISTRIBUTED) | PyTorch Lightning version with Distributed Data Parallel: `LightningDataModule` with distributed samplers, `LightningModule`, SLURM submission script for 2 nodes × 2 A100 GPUs. |

## Baseline results (single GPU)

| Epoch | Train loss | Exact Match | F1 |
|---:|---:|---:|---:|
| 1 | 0.19 | 16.4 | 49.4 |
| 2 | 0.10 | 21.6 | 56.7 |
| 3 | 0.07 | 16.4 | 53.3 |
| 4 | 0.06 | 19.2 | 51.4 |
| 5 | 0.08 | 19.6 | 53.5 |

About 15 s of training per epoch, 5 min 36 s end to end.

## Roadmap

- Update the distributed data loader to the current PyTorch Lightning API and run the 2-node benchmark.
- Compare training time and speedup against the baseline.

## Tech stack

Python · PyTorch · PyTorch Lightning (DDP) · Hugging Face Transformers / Datasets · SLURM · NCCL
