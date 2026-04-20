# Theory and Methods (Phase 2 — Rigor Checklist)

This document ties **mechanics**, **optimization**, and **validation** to the implementation in `src/` and `scripts/`.

---

## 1. Problem and learning objective

We treat **network intrusion detection** as **unsupervised anomaly detection**: only **benign** connections are available at training time. The model must produce a score or boundary such that **attacks** (held-out, labeled only for evaluation) are **separated** from normal behavior.

**MAE pre-training** optimizes **masked reconstruction**: a random subset of feature dimensions is replaced with a **learnable mask token**; the network predicts the original values at those positions. The objective is **mean squared error (MSE)** on masked coordinates:

\[
\mathcal{L}_{\mathrm{MAE}} = \frac{1}{|M|} \sum_{j \in M} (\hat{x}_j - x_j)^2
\]

where \(M\) is the set of masked feature indices per sample. This is **self-supervised**: no attack labels are used. The encoder builds **contextual representations** of tabular features analogous to BERT-style masking in NLP, adapted to **feature tokens** (each scalar feature is a token after linear projection).

---

## 2. Architecture mechanics

### 2.1 Tokenization and Transformer encoder

Each feature value \(x_j\) is embedded with a shared linear map \(\mathbb{R} \to \mathbb{R}^H\). **Sinusoidal positional encodings** over feature index \(j\) inject **order** across the feature sequence (which feature slot is which). The stack uses **multi-head self-attention** (PyTorch `TransformerEncoderLayer`) so each representation can attend to all other features. **Pre-LayerNorm** (`norm_first=True`) stabilizes optimization in deeper stacks.

**Self-attention** (single head, simplified): for input matrix \(X \in \mathbb{R}^{L \times H}\),

\[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

with \(Q = XW_Q\), \(K = XW_K\), \(V = XW_V\). Gradients flow through softmax and linear maps, enabling **global mixing** across features in one layer.

### 2.2 Readout: mean–max fusion (tabular specialization)

Standard **mean pooling** over tokens can dilute **salient peaks** (e.g., rare large byte counts). We add a **max** over the feature dimension and **concatenate** \([\mathrm{mean}; \mathrm{max}] \in \mathbb{R}^{2H}\), then a **linear + GELU + LayerNorm** map back to \(\mathbb{R}^H\). This is a **deliberate inductive bias** for heterogeneous tabular scales; `readout_mode: mean` in config recovers the baseline for ablation.

### 2.3 Decoder and hybrid stage

The **lightweight decoder** reconstructs from encoded context. For **hybrid detection**, the encoder is **frozen** and its embedding \(z \in \mathbb{R}^H\) feeds a **One-Class SVM**, which learns a **minimum-volume** region containing most benign embeddings in a kernel-induced feature space.

---

## 3. One-Class SVM (boundary)

**One-Class SVM** (Schölkopf et al.) finds a hyperplane in a **Reproducing Kernel Hilbert Space** (RBF kernel in our config) that separates most training data from the origin, or equivalently encloses normal data. Parameter **`nu`** upper-bounds the fraction of **training outliers** (benign points allowed outside). The **decision function** \(f(z)\) is **positive** for typical benign embeddings and **negative** for outliers; we use **\(-f(z)\)** as an **anomaly score** so higher means more anomalous, matching **ROC-AUC** with binary attack labels at evaluation time.

---

## 4. Optimization choices

- **AdamW** decouples **weight decay** from the adaptive gradient, improving generalization vs. classic L2 inside Adam.
- **Weight decay** and **dropout** act as **regularizers**; **early stopping** tracks **validation** reconstruction loss (held-out benign split) to limit overfitting.
- **Cosine annealing** (optional) reduces the learning rate smoothly over epochs, often helping late-stage convergence on noisy tabular objectives.
- **Mixup** on input batches (\(x' = \lambda x + (1-\lambda)\tilde{x}\)) acts as **vicinal risk minimization**, smoothing the reconstruction landscape for tabular MAE.

---

## 5. Data pipeline and leakage control

- **Preprocessor** (`StandardScaler`, imputer, encoders) is **fit on training data only**; **test** is **transform**-only.
- MAE sees **benign-only** rows from the training file; evaluation uses the **official NSL-KDD test split** with labels **only for metrics**.

---

## 6. Metrics and validation narrative

- **F1** and **ROC-AUC** on the **attack vs. normal** binary task are appropriate for **imbalanced** intrusion data (accuracy alone is misleading).
- **Confusion matrix** and **ROC curves** diagnose **false positives vs. false negatives**.
- **Gaussian noise** scaled per-feature by test **standard deviation** probes **robustness** to measurement jitter.
- **Trivial baselines** (majority class, random) contextualize gains.
- **SHAP** on the SVM **decision function** (raw features or embeddings) supports **explainability**: which inputs or latent dimensions push the score toward “anomaly.”

---

## 7. Ablation roadmap (what to cite in the report)

| Component | Script / config knob |
|-----------|----------------------|
| Adv ML only | `run_ablation.py` Model A |
| DL reconstruction score | Model B |
| Hybrid | Model C |
| Mean vs mean–max readout | `mae.readout_mode` |
| Mixup / schedule | `mae.mixup_alpha`, `mae.lr_scheduler` |

---

## 8. Honest scope note (vs. rubric “10 / novel”)

**True novelty** in the research sense (a new layer with a new theorem) is **not** required for a strong engineering project. Here, **novelty** is **compositional**: tabular MAE + mean–max readout + hybrid OCSVM + rigorous validation (curves, robustness, SHAP, ablations). Position this honestly in the report as **a justified design** building on cited **SOTA components**, not a new universal architecture.
