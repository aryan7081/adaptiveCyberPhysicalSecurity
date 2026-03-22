# Adaptive Cyber-Physical Security

**Domain:** Cybersecurity, Network Monitoring, Critical Infrastructure  
**Project:** Anomaly detection in network traffic using unsupervised/semi-supervised methods.  
**Courses:** Advanced Machine Learning + Deep Learning

## Strategy

- **Pre-train** a Masked Autoencoder (MAE) with BERT-style masking on **benign traffic only**
- **Freeze** the encoder and extract embeddings
- **Train** a One-Class SVM on embeddings for explicit decision boundaries
- **Hybrid** architecture: DL representations + probabilistic anomaly boundary

## Tech Stack

- **ML:** scikit-learn (One-Class SVM, preprocessing)
- **DL:** PyTorch (MAE, Transformer encoder)
- **Network:** Scapy (packet parsing, optional live capture)
- **Data:** NSL-KDD

## Quick Start

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Download NSL-KDD
python scripts/download_data.py

# 3. Run EDA
python scripts/run_eda.py

# 4. Pre-train MAE on benign traffic
python scripts/train_mae.py --device cuda  # or cpu

# 5. Ablation study (Model A, B, C)
python scripts/run_ablation.py --device cuda
```

## Project Structure

```
.
├── config/
│   └── config.yaml          # Reproducible hyperparameters
├── data/
│   ├── raw/                 # KDDTrain+.txt, KDDTest+.txt
│   └── processed/           # EDA outputs, cached features
├── figures/                 # EDA plots, architecture diagram
├── models/                  # Saved MAE, OCSVM checkpoints
├── reports/                 # LaTeX report
├── results/                 # Ablation table, metrics
├── scripts/
│   ├── download_data.py
│   ├── run_eda.py
│   ├── train_mae.py
│   └── run_ablation.py
├── src/
│   ├── data/                # Loader, preprocessing
│   ├── features/            # Feature engineering
│   └── models/              # MAE, OCSVM, Hybrid
├── NSL_KDD_FEATURES.md      # Feature reference
├── requirements.txt
└── README.md
```

## Models (Ablation)

| Model | Description |
|-------|-------------|
| **A: Adv ML only** | One-Class SVM on preprocessed raw features |
| **B: DL only** | MAE reconstruction error as anomaly score (threshold) |
| **C: Hybrid** | Frozen MAE encoder → embeddings → One-Class SVM |

## Dataset

**NSL-KDD** from Canadian Institute for Cybersecurity (CIC), UNB.  
- 41 features (network traffic + host-based)  
- Label: normal vs. attack types (DoS, Probe, R2L, U2R)  
- Train on benign only; evaluate on full test set with unseen attacks  

Download: https://github.com/defcom17/NSL_KDD  
Place `KDDTrain+.txt` and `KDDTest+.txt` in `data/raw/`.

## Phase 1 Deliverables

- [x] Literature review
- [x] EDA (dataset quality, distributions, correlations)
- [x] Feature engineering (ratios, interactions)
- [x] Theoretical rigor (MAE, OCSVM, assumptions)
- [x] Ablation table (A, B, C)
- [x] Architecture diagram
- [x] LaTeX report template
- [ ] Presentation / Video (10 min)
- [ ] Viva voce

## License

Academic use only. NSL-KDD dataset: CIC/UNB.
