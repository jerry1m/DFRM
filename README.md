# DFRM
A normalizing flow-based industrial anomaly detection and localization framework.
# SCOAF: Synergistic Co-Optimization Framework with Adaptive Boundary Contrastive Learning and Feature Refinement for Anomaly Detection

> **SCOAF** (originally BGAD) is a unified framework for industrial anomaly detection that synergistically integrates **Dynamic Feature Refinement Module (DFRM)** and **Uncertainty-Aware Boundary Optimization** to achieve state-of-the-art performance on both unsupervised and few-shot anomaly detection settings.

---

## Table of Contents

- [Overview](#overview)
- [Core Innovations](#core-innovations)
  - [1. DFRM (Dynamic Feature Refinement Module)](#1-dfrm-dynamic-feature-refinement-module)
  - [2. Uncertainty-Aware Boundary Optimization](#2-uncertainty-aware-boundary-optimization)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Testing](#testing)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Training Modes](#training-modes)
  - [Unsupervised Mode (BGAD)](#unsupervised-mode-bgad)
  - [Few-Anomaly-Samples Mode (FAS)](#few-anomaly-samples-mode-fas)
- [Results](#results)
- [Citation](#citation)

---

## Overview

SCOAF is an anomaly detection and localization framework designed for industrial visual inspection. It addresses two fundamental challenges in anomaly detection:

1. **Feature discrimination**: Backbone features contain both normal patterns and background noise. DFRM enhances discriminative features while suppressing irrelevant ones.
2. **Boundary ambiguity**: The decision boundary between normal and abnormal is often unclear. Uncertainty-Aware Boundary Optimization quantifies sample uncertainty and adaptively adjusts the contrastive boundary.

The framework supports two training paradigms:
- **Unsupervised learning** using only normal samples (via normalizing flow + adaptive boundary contrastive loss)
- **Few-shot anomaly learning** with a small number of abnormal samples combined with segmentation and edge supervision

---

## Core Innovations

### 1. DFRM (Dynamic Feature Refinement Module)

**Location**: `models/resnet_se_backbone.py`, `models/quaternion_backbone.py`

DFRM wraps any CNN backbone (ResNet, EfficientNet, etc.) with lightweight channel modulation modules. Each module applies Squeeze-and-Excitation style recalibration:

```
Input Feature → Global Avg Pool → FC(ReLU) → FC(Sigmoid) → Channel-wise Scale → Refined Feature
```

Key characteristics:
- **Residual formulation**: `output = input * scale_factor` — multiplicative gating preserves identity
- **Lightweight**: Only 2 linear layers per feature level with a reduction ratio (default 16)
- **Per-level modulation**: Each feature pyramid level has its own DFRM module
- **Ablation support**: `--disable_dfrm` flag for controlled experiments
- **Analysis tools**: `get_dfrm_channel_importance()` extracts channel importance scores

Two backbone variants are provided:
| Backbone | Description |
|----------|-------------|
| `ResNetDFRMBackbone` (`--backbone_arch resnet_se`) | Wraps timm ResNet (resnet18/34/50) with DFRM modules |
| `QuaternionBackbone` (`--backbone_arch quaternion_cnn`) | Lightweight custom backbone with DFRM, suitable for resource-constrained settings |

### 2. Uncertainty-Aware Boundary Optimization

**Location**: `losses/losses.py`, `engines/bgad_train_engine.py`

This innovation tackles the ambiguity at the normal-abnormal decision boundary through several mechanisms:

#### a) Uncertainty-Aware Boundary Estimation
```python
boundaries = compute_uncertainty_aware_boundary(logps, mask, pos_beta, margin_tau)
```
- Quantifies uncertainty via spatial feature dispersion
- Adaptively sets the normal boundary $b_n$ and abnormal boundary $b_a = b_n - \tau$
- Uses a beta-quantile of normal sample log-likelihoods for robust estimation

#### b) Adaptive Boundary Contrastive Loss
- **Semi-push-pull mechanism**: pushes abnormal samples below $b_a$ and pulls normal samples above $b_n$
- **Asymmetric focal weighting** (optional, `--focal_weighting`): reweights hard normal/anomaly samples
  - `normal_adaptive_weighting()`: assigns higher weights to low-likelihood normal samples
  - `abnormal_adaptive_weighting()`: assigns higher weights to high-likelihood abnormal samples

#### c) Curriculum Margin Scheduling ($\tau(t)$)

The margin $\tau$ follows an exponential decay schedule:

$$\tau(t) = \tau_{\min} + (\tau_{\max} - \tau_{\min}) \cdot \exp\left(-\lambda \cdot \frac{t}{T}\right)$$

Where:
- $\tau_{\max}$: initial (maximum) margin (`--tau_max`, default 0.1)
- $\tau_{\min}$: final (minimum) margin (`--tau_min`, default 0.02)
- $\lambda$: decay rate (`--tau_lambda`, default 5.0)
- $t$: current training progress (epoch + sub_epoch normalization)
- $T$: total meta epochs

This enables the model to start with a relaxed boundary and progressively tighten it for finer discrimination.

#### d) UncertaintyBoundary Module

**Location**: `losses/losses.py` — `class UncertaintyBoundary`

A learnable module that maintains per-class diagonal Gaussian parameters (mean $\mu_c$ and log-variance $\log\sigma_c^2$):
- Estimates sample log-likelihood under a learned class-conditional distribution
- Computes uncertainty-scaled boundary: $b_n = \text{mean}(\mu_c) - k \cdot \text{mean}(\sigma_c)$
- Adaptive weighting: $w = 1 + \lambda_w \cdot \text{mean}(\sigma_c)$ — higher uncertainty → larger gradient signal
- Momentum-based parameter updates through `update_boundary_momentum()`
- Gradient clipping hooks ensure training stability

#### e) Uncertainty Boundary Statistics Logging

During training, per-layer UB statistics are written to CSV (`<output>/<exp_name>/logs/ub_stats_<class>.csv`):
- `uncertainty_mean`: mean standard deviation across feature dimensions
- `weight_mean`: mean adaptive weight
- `ub_b_n / ub_b_a`: learned boundary values from UncertaintyBoundary
- `flow_b_n / flow_b_a`: flow-based boundary estimates
- `normal_violate_ratio / anomaly_violate_ratio`: percentage of samples violating boundary

---

## Architecture

```
Input Image
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Feature Extractor (timm backbone + DFRM modules)     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Layer 1  │──│ Layer 2  │──│ Layer 3  │ ...        │
│  │ + DFRM   │  │ + DFRM   │  │ + DFRM   │            │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘            │
└───────┼─────────────┼──────────────┼──────────────────┘
        │             │              │
        ▼             ▼              ▼
┌─────────────────────────────────────────┐
│  Normalizing Flow (FrEIA) per level     │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Decoder 1│  │ Decoder 2│  │Dec 3.. │ │
│  └────┬─────┘  └────┬─────┘  └───┬────┘ │
└───────┼─────────────┼──────────────┼──────┘
        │             │              │
        ▼             ▼              ▼
┌─────────────────────────────────────────┐
│  Uncertainty-Aware Boundary Optimization│
│  - Adaptive Contrastive Loss            │
│  - UncertaintyBoundary Module (learned) │
│  - Curriculum Margin Scheduling τ(t)    │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐  (FAS mode only)
│  Segmentation Head (U-Net) + EdgeDet    │
│  - Dice Loss (mask)                     │
│  - BCE Loss (edges)                     │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
BGAD-git/
├── config.py                         # Argument parser and configuration
├── main_modified.py                  # Entry point: training/testing orchestration
├── run_all.py                        # Batch experiment runner
├── requirements.txt                  # Python dependencies
│
├── datasets/
│   ├── mvtec.py                      # MVTec AD, BTAD, VisA, ISBI2016 dataset loaders
│   ├── perlin.py                     # Perlin noise mask generator
│   └── nsa.py                        # Normal-to-Normal Patch Exchange (NSA) augmentation
│
├── models/
│   ├── __init__.py                   # Model loading hub (load_flow_model)
│   ├── resnet_se_backbone.py         # ResNetDFRMBackbone: ResNet + DFRM modules
│   ├── quaternion_backbone.py        # QuaternionBackbone: lightweight DFRM backbone
│   ├── modules.py                    # UNetSegmentationHead, EdgeDetector, positionalencoding2d
│   ├── fc_flow.py                    # Normalizing flow architectures (flow_model, conditional_flow_model)
│   └── ...
│
├── losses/
│   ├── __init__.py                   # Loss function exports
│   ├── losses.py                     # Core losses: adaptive weighting, boundary contrastive, UncertaintyBoundary, DiceLoss, BCEWithLogitsLoss
│   └── ...
│
├── engines/
│   ├── bgad_train_engine.py          # Unsupervised training engine (normalizing flow + UncertaintyBoundary)
│   ├── bgad_fas_train_engine.py      # FAS training engine (adds segmentation head + edge detector)
│   └── bgad_test_engine.py           # Unified testing engine
│
├── utils/
│   ├── utils.py                      # MetricRecorder, t2np, get_logp, init_seeds, PRO metric, etc.
│   ├── model_utils.py                # save_weights, load_weights, lr scheduling
│   └── visualizer.py                 # Visualization tools: score maps, masks, edges, overlays
│
├── FrEIA/                            # FrEIA normalizing flow library (git submodule)
│
└── output/                           # Experiment outputs (auto-created)
    └── <exp_name>/
        ├── weights/                  # Model checkpoints
        ├── results/                  # Metric result files
        ├── logs/                     # Training logs and UB statistics
        └── vis_results/              # Visualization outputs
```

---

## Requirements

### Core Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.7.0
- torchvision ≥ 0.8.0
- CUDA-capable GPU

### Python Packages

```
timm                         # Backbone model zoo
FrEIA (VLL-HD fork)          # Normalizing flow framework
numpy, scipy                 # Numerical computing
scikit-learn                 # Evaluation metrics (ROC AUC)
scikit-image                 # Image processing (PRO metric)
matplotlib                   # Visualization
tqdm                         # Progress bars
imgaug, albumentations       # Data augmentation
Pillow                       # Image I/O
OpenCV (optional)            # Seamless cloning augmentation
```

Install via:

```bash
pip install -r requirements.txt
```

The FrEIA library is included as a local dependency (`FrEIA/` directory). If missing, clone it:

```bash
git clone https://github.com/VLL-HD/FrEIA.git
```

---

## Quick Start

### Data Preparation

#### MVTec AD

Download from [MVTec AD official website](https://www.mvtec.com/company/research/datasets/mvtec-ad/), then set `--data_path` to the dataset root directory. Expected structure:

```
<data_path>/
├── bottle/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   ├── broken_large/
│   │   └── ...
│   └── ground_truth/
└── cable/
    └── ...
```

#### BTAD / VisA / ISBI2016

These datasets are also supported. Set `--dataset` to `btad`, `visa`, or `isbi2016` respectively, with the same folder convention.

### Training

**Unsupervised mode** (normal samples only):

```bash
python main_modified.py \
    --gpu 0 \
    --backbone_arch tf_efficientnet_b6 \
    --flow_arch conditional_flow_model \
    --dataset mvtec \
    --class_name bottle \
    --data_path /path/to/mvtec \
    --meta_epochs 25 \
    --sub_epochs 8 \
    --feature_levels 3 \
    --coupling_layers 8 \
    --exp_name scoaf_unsup
```

**FAS mode** (with few abnormal samples):

```bash
python main_modified.py \
    --gpu 0 \
    --backbone_arch tf_efficientnet_b6 \
    --flow_arch conditional_flow_model \
    --dataset mvtec \
    --class_name screw \
    --data_path /path/to/mvtec \
    --with_fas \
    --data_strategy 0,1 \
    --num_anomalies 50 \
    --meta_epochs 40 \
    --sub_epochs 8 \
    --focal_weighting \
    --exp_name scoaf_fas_screw
```

### Testing

```bash
python main_modified.py \
    --gpu 0 \
    --phase test \
    --checkpoint /path/to/checkpoint.pt \
    --dataset mvtec \
    --class_name bottle \
    --exp_name scoaf_test
```

---

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backbone_arch` | `tf_efficientnet_b6` | Backbone architecture (timm models, `resnet_se`, `quaternion_cnn`) |
| `--backbone_base` | `resnet50` | Base architecture for `resnet_se` backbone |
| `--flow_arch` | `conditional_flow_model` | Normalizing flow type |
| `--feature_levels` | 3 | Number of feature pyramid levels |
| `--coupling_layers` | 8 | Number of coupling layers per flow |
| `--meta_epochs` | 25 | Outer training epochs |
| `--sub_epochs` | 8 | Inner training iterations per meta epoch |

### DFRM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--disable_dfrm` | False | Ablation: bypass DFRM channel modulation |

### Uncertainty-Aware Boundary Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pos_beta` | 0.05 | Beta quantile for normal boundary estimation |
| `--margin_tau` | 0.1 | Margin between normal and abnormal boundary |
| `--tau_min` | 0.02 | Minimum margin in curriculum scheduling |
| `--tau_max` | 0.1 | Maximum (initial) margin in curriculum scheduling |
| `--tau_lambda` | 5.0 | Decay rate for curriculum scheduling |
| `--normalizer` | 10 | Log-likelihood normalizer |
| `--bgspp_lambda` | 1 | Loss weight for boundary contrastive loss |
| `--focal_weighting` | False | Enable asymmetric focal weighting |
| `--ub_weight` | 1.0 | Weight for UncertaintyBoundary loss |
| `--ub_max_grad_clip` | 1.0 | Max gradient clip for UB parameters |

### Data Augmentation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_strategy` | `0,1` | Comma-separated: 0=Repeat, 1=CutPaste, 2=Pseudo-anomaly |
| `--num_anomalies` | 5 | Number of anomaly samples per category (FAS mode) |
| `--pseudo_type` | `nsa` | Pseudo anomaly type: `nsa` (patch exchange) or `perlin` |
| `--in_fg_region` | True | Restrict pasting to foreground region |
| `--use_seamless_clone` | False | Use OpenCV seamlessClone for realistic pasting |
| `--strong_ops` | 6 | Number of augment ops for prioritized subtypes |

---

## Datasets

| Dataset | Classes | Supported | Notes |
|---------|---------|-----------|-------|
| **MVTec AD** | 15 | ✅ Full | Primary benchmark |
| **BTAD** | 3 | ✅ | Beauly Technology Anomaly Detection |
| **VisA** | 12 | ✅ | Visual Anomaly dataset |
| **ISBI2016** | 1 | ✅ | Skin lesion segmentation |

---

## Training Modes

### Unsupervised Mode (BGAD)

Uses `engines/bgad_train_engine.py`. Trains normalizing flows on normal samples only, with:
- Maximum likelihood estimation on normal data
- Adaptive boundary contrastive loss for boundary discrimination
- UncertaintyBoundary module with curriculum margin scheduling
- No segmentation head or edge detector

Activate via `--without_fas` (sets `--with_fas False`).

### Few-Anomaly-Samples Mode (FAS)

Uses `engines/bgad_fas_train_engine.py`. Adds to the unsupervised pipeline:
- **UNetSegmentationHead**: Produces pixel-level defect masks
- **EdgeDetector**: Laplacian-style edge detection module
- **Dice Loss** + **BCEWithLogits Loss**: Supervise mask and edge predictions
- Combined loss: `flow_loss + 0.3 * seg_loss + 0.15 * edge_loss`
- First epoch trains on normal samples only for stable initialization

Activate via `--with_fas` (enabled by default).

Key differences between the two modes:

| Aspect | Unsupervised (BGAD) | FAS |
|--------|-------------------|-----|
| Training data | Normal only | Normal + few abnormal |
| Segmentation head | No | UNetSegmentationHead |
| Edge detector | No | EdgeDetector (Laplacian) |
| Boundary loss | Normal only | Both normal + abnormal |
| Adaptive weighting | Normal only | Normal + abnormal |
| Complexity | Lower | Higher (multi-task) |

---

## Results

### Evaluation Metrics

- **Image-level AUROC** (detection): Classify image as normal or anomalous
- **Pixel-level AUROC** (localization): Per-pixel anomaly scoring
- **AUPRO** (localization): Per-region overlap metric
- **Mask AUROC** (FAS mode only): Segmentation mask evaluation

### Inference Latency

Use `--measure_inference` to benchmark per-image latency (milliseconds) and throughput (images/second). Warmup batches are skipped via `--latency_skip`.

---

## Citation

If you use SCOAF in your research, please cite:

```bibtex
@article{scoaf2025,
  title={SCOAF: Synergistic Co-Optimization Framework with Adaptive Boundary Contrastive Learning and Feature Refinement for Anomaly Detection},
  author={},
  journal={},
  year={2025}
}
```

---

## License

This project is intended for research and educational purposes.
