# DFRM

It includes the implementation of DFRM (Dynamic Normalizing Flow for anomaly detection). Able to run on MVTec AD, BTAD, VisA, and ISBI2016 datasets.

It also incorporates FAS (Few Anomaly Samples) training with segmentation head and edge detector.

*This work is completed during internship at Newland AIDC.

---

## Table of Contents

- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Data Preparation](#data-preparation)
- [Datasets](#datasets)
- [Results](#results)


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

## Datasets

| Dataset | Classes | Supported | Notes |
|---------|---------|-----------|-------|
| **MVTec AD** | 15 | ✅ Full | Primary benchmark |
| **BTAD** | 3 | ✅ | Beauly Technology Anomaly Detection |
| **VisA** | 12 | ✅ | Visual Anomaly dataset |
| **ISBI2016** | 1 | ✅ | Skin lesion segmentation |

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

## License

This project is intended for research and educational purposes.
