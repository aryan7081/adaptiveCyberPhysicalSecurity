# Literature Review: Adaptive Cyber-Physical Security
## Anomaly Detection in Network Traffic

---

## 1. Introduction

This review covers prior work on anomaly detection for network security, with focus on: (i) statistical and classical ML methods, (ii) deep learning approaches, and (iii) hybrid systems. We pit these approaches against each other to highlight gaps—no single method suffices—and defend our MAE + One-Class SVM hybrid as filling those gaps. We link each prior work to our project design.

---

## 2. Statistical and Classical ML Methods

### One-Class SVM
**Schölkopf et al. (2001)** — *"Estimating the support of a high-dimensional distribution"*, Neural Computation.

One-Class SVM learns a boundary around training data (assumed normal). Samples outside the boundary are anomalies. Uses kernel functions (e.g., RBF) for non-linear boundaries. **Link to our work:** We use OCSVM as baseline (Model A) and in our hybrid (Model C) on MAE embeddings.

### NSL-KDD Dataset
**Tavallaee et al. (2009)** — *"A detailed analysis of the KDD CUP 99 data set"*, IEEE CISDA.

Addresses issues in the original KDD Cup 99 dataset: redundant records, difficulty levels. NSL-KDD has 41 features per connection. **Link:** We use NSL-KDD for training on benign-only and evaluation on unseen attacks.

### Isolation Forest, Distance-Based Methods
Classical anomaly detection also uses Isolation Forest (tree-based), Mahalanobis distance, and clustering. These require careful feature engineering. **Gap:** They do not learn representations; performance depends on raw features.

---

## 3. Deep Learning Approaches

### Variational Autoencoders (VAE)
**Kingma & Welling (2014)** — *"Auto-encoding variational bayes"*, ICLR.

VAE learns a latent distribution of normal data. High reconstruction error or low likelihood → anomaly. **Link:** Similar idea to our MAE; we use reconstruction error for Model B.

### Masked Language Modeling / BERT
**Devlin et al. (2019)** — *"BERT: Pre-training of deep bidirectional transformers"*, NAACL.

BERT masks tokens and predicts them, learning contextual representations. **Link:** Our MAE uses BERT-style masking on tabular features—hide ~15%, predict from context.

### Deep One-Class Classification
**Ruff et al. (2018)** — *"Deep one-class classification"*, ICML.

Combines neural networks with one-class objectives. **Link:** Our hybrid follows this philosophy—neural representation + explicit boundary.

---

## 4. Hybrid Approaches

Several works combine learned representations with classical anomaly detectors:

- **AE/SVM hybrids:** Train autoencoder, use latent codes as input to SVM.
- **Transfer learning:** Pre-train on large data, fine-tune for anomaly detection.

**Our contribution:** We use MAE (masked pre-training) instead of vanilla AE, and OCSVM for unsupervised boundary learning. We train only on benign data—no attack labels.

---

## 5. Summary: Gaps Our Project Addresses

| Prior Work | Limitation | Our Approach |
|------------|------------|--------------|
| OCSVM alone | No representation learning | Add MAE for embeddings |
| VAE/AE | Less interpretable boundary | Add OCSVM for explicit boundary |
| BERT | For text, not tabular | Adapt masking to network features |
| Most work | Binary classification | Unsupervised, benign-only training |

---

## 6. Recent Tabular Deep Learning and Anomaly Detection (2020+)

### Tabular representation learning
**Huang et al. (2020)** — *"TabTransformer: Tabular Data Modeling Using Contextual Embeddings"*. Uses column embeddings + self-attention over categorical tokens; motivates **attention over feature groups** for heterogeneous tables.

**Gorishniy et al. (2021)** — *"Revisiting Deep Learning Models for Tabular Data" (FT-Transformer)*, NeurIPS. Shows **Transformer-style** models with **piecewise embeddings** are **strong tabular baselines** vs. GBDTs on many benchmarks. **Link:** Our MAE uses **per-feature tokens + Transformer**, aligned with this lineage (we add **masking** for self-supervision rather than supervised FT).

**Somepalli et al. (2021)** — *SAINT: Improved Neural Networks for Tabular Data* (embedding both features and samples). **Link:** Highlights that **inductive biases** (attention, inter-feature mixing) matter for tabular; our **mean–max readout** is a lightweight bias for **scale-heavy** network features.

### Deep anomaly detection
**Ruff et al. (2018)** already connects to **hybrid** boundaries; follow-on work and **surveys** (e.g., *deep anomaly detection* surveys in the 2020s) stress that **pure reconstruction** can be **brittle** under attack diversity, while **explicit boundaries** (one-class, SVDD-style) improve **calibration** in some regimes.

**Positioning our project:** We combine (i) **self-supervised masked modeling** for **representation learning** on **benign-only** NSL-KDD with (ii) a **kernel one-class boundary** on **frozen embeddings**—a **documented hybrid** in the spirit of **deep one-class** and **AE + classical detector** papers, with **tabular-specific readout** and **modern Transformer blocks** rather than a vanilla MLP autoencoder.

---

## 7. Synthesis: From SOTA Components to This System

| Literature line | Typical limitation | Our step |
|-----------------|-------------------|----------|
| Classical OCSVM on raw features | Weak on nonlinear manifolds without good kernel/features | Add **MAE embeddings** before OCSVM |
| Vanilla AE / VAE anomaly score | Threshold on reconstruction only; fuzzy boundary | Keep MAE for features; add **explicit OCSVM** boundary |
| BERT / MAE (vision/tabular variants) | Often evaluated on supervised or generative tasks | Use **masking objective** for **unsupervised** benign modeling |
| FT-Transformer / TabTransformer | Supervised training | **Self-supervised** phase + **one-class** second stage |

This positions the work as a **logical composition** of **current tabular Transformer practice** and **established anomaly-detection theory**, with **empirical ablations** (Models A–C, readout modes) in code and reports.

---

## 8. References (for report)

**Foundational / cited in design**

1. M. Tavallaee et al., "A detailed analysis of the KDD CUP 99 data set," IEEE CISDA, 2009. [IEEE Xplore](https://ieeexplore.ieee.org/document/5356528) · [Dataset (UNB/CIC)](https://www.unb.ca/cic/datasets/nsl.html)
2. B. Schölkopf et al., "Estimating the support of a high-dimensional distribution," *Neural Computation*, 2001. [MIT Press / DOI](https://direct.mit.edu/neco/article-abstract/13/7/1443/6009)
3. D. P. Kingma and M. Welling, "Auto-encoding variational bayes," ICLR, 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
4. J. Devlin et al., "BERT: Pre-training of deep bidirectional transformers," NAACL, 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
5. L. Ruff et al., "Deep one-class classification," ICML, 2018. [PMLR proceedings](http://proceedings.mlr.press/v80/ruff18a.html) · [arXiv:1801.05365](https://arxiv.org/abs/1801.05365)

**Tabular deep learning (recent context)**

6. X. Huang et al., "TabTransformer: Tabular Data Modeling Using Contextual Embeddings," arXiv:2008.06615, 2020. [arXiv](https://arxiv.org/abs/2008.06615)
7. Y. Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data," NeurIPS, 2021 (FT-Transformer). [arXiv:2106.11959](https://arxiv.org/abs/2106.11959)
8. G. Somepalli et al., "SAINT: Improved Neural Networks for Tabular Data," NeurIPS, 2021. [arXiv:2106.01342](https://arxiv.org/abs/2106.01342)

*Add 1–2 **survey** papers on deep anomaly detection (2020+) from your library search to strengthen “State of the Art” narrative if required by the rubric.*

---

*Use this for Related Work, gap analysis, and viva defense. Pair with `THEORY_AND_METHODS.md` for equations and optimization rationale.*
