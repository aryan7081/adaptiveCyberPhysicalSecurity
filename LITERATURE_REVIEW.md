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

## 6. References (for report)

1. M. Tavallaee et al., "A detailed analysis of the KDD CUP 99 data set," IEEE CISDA, 2009.
2. B. Schölkopf et al., "Estimating the support of a high-dimensional distribution," Neural Computation, 2001.
3. D. P. Kingma and M. Welling, "Auto-encoding variational bayes," ICLR, 2014.
4. J. Devlin et al., "BERT: Pre-training of deep bidirectional transformers," NAACL, 2019.
5. L. Ruff et al., "Deep one-class classification," ICML, 2018.

---

*Use this for the report's Related Work section and for explaining literature in the viva.*
