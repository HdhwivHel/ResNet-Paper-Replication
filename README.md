# ResNet-18 (2015) - Paper Replication

## Overview

This implementation is based on the architecture introduced in the paper:

**Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - _Deep Residual Learning for Image Recognition_ (2015).**

ResNet introduced residual (skip) connections that made it practical to train much deeper convolutional neural networks. The core idea is to learn residual mappings instead of full transformations, which improves optimization and gradient flow.

This repository provides two ways to work with the ResNet-18 replication:

1. **Modular Python Implementation** - for reproducible training and evaluation workflows.
2. **Interactive Notebook Implementation** - for exploration, experimentation, and quick iteration.

---

# Installation

Clone your repository and enter the project folder:

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python >= 3.10
```

---

## 1. Modular Implementation

The modular version separates model code, dataset pipeline, configuration, and train/test scripts.

### Train

```bash
python training/train.py
```

### Evaluate

```bash
python training/test.py
```

Training, testing, scheduler, and model checkpoint names are configured in:

```text
configs/config.yaml
```

---

## 2. Notebook Implementation

The notebook [main.ipynb](main.ipynb) provides a cell-by-cell ResNet-18 workflow including:

- Config handling via YAML or in-notebook dictionary
- CIFAR-10 train/test data pipeline
- Training loop with scheduler
- Evaluation loop
- Gradio-based random-sample inference

---

# Differences from Original ResNet Paper

## 1. Dataset and Task Differences

| Component         | Original Paper | This Implementation | Reason                               |
| ----------------- | -------------- | ------------------- | ------------------------------------ |
| Primary benchmark | ImageNet       | CIFAR-10            | Faster replication and lower compute |
| Output classes    | 1000           | 10                  | Match CIFAR-10 labels                |

---

## 2. Training Differences

| Component   | Original Paper (ImageNet setup) | This Implementation                                  | Reason                                |
| ----------- | ------------------------------- | ---------------------------------------------------- | ------------------------------------- |
| Epochs      | 90                              | 40                                                   | Smaller-scale experimental setup      |
| Loss        | CrossEntropy                    | CrossEntropy (+ label smoothing in modular train.py) | Stabilized training in modular script |
| LR schedule | Step decay                      | Configurable `step` or `multistep`                   | Flexible experiments                  |
| Compilation | Not applicable                  | Optional `torch.compile` on CUDA                     | Modern PyTorch performance path       |

---

## 3. Architecture Notes

| Property      | Value in This Repository              |
| ------------- | ------------------------------------- |
| Block type    | BasicBlock (2 x 3x3 convolutions)     |
| Residual path | Identity or 1x1 downsample projection |
| Stages        | [64, 128, 256, 512], 2 blocks each    |
| Stem          | 7x7 conv, stride 2 + maxpool          |
| Output layer  | Linear(512 -> num_classes)            |

Residual addition and stage transitions follow standard ResNet-18 design.

---

# Training Configuration

Default values in [configs/config.yaml](configs/config.yaml):

| Parameter                   | Value                 |
| --------------------------- | --------------------- |
| Optimizer                   | SGD                   |
| Learning Rate               | 0.01                  |
| Momentum                    | 0.9                   |
| Weight Decay                | 5e-4                  |
| Train Batch Size            | 256                   |
| Test Batch Size             | 256                   |
| Epochs                      | 40                    |
| Scheduler Type              | `step` (configurable) |
| Step Size                   | 15                    |
| Gamma                       | 0.1                   |
| Milestones (multistep mode) | [30, 45]              |

Checkpoint names are also config-driven:

- `model.name` / `model_to_save.name` for training output
- `model_to_test.name` for evaluation and inference target

---

# Dataset

This project trains on **CIFAR-10** using torchvision dataset utilities.

Transforms used:

- Train: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize(CIFAR-10 mean/std)
- Test: Normalize(CIFAR-10 mean/std)

CIFAR-10 stats:

```text
Classes: 10
Training images: 50,000
Test images: 10,000
Resolution: 32 x 32
```

---

# Project Structure

```text
configs/                # YAML configuration
dataset_pipeline/       # CIFAR-10 transforms and dataset loader
models/                 # ResNet model definition
training/               # train.py and test.py
results/checkpoints/    # saved model weights
main.ipynb              # interactive notebook workflow
```

---

# Summary

This repository reproduces the core ResNet-18 ideas with a practical CIFAR-10 pipeline, while keeping the workflow flexible through:

- configuration-driven hyperparameters
- modular train/test scripts
- notebook-first experimentation
- inference through a lightweight Gradio interface

It is designed to be easy to run locally or in Colab while preserving the key residual-learning principles from the original paper.
