# Phase 1 Rubric Guide: Targeting 10/10

Use this checklist to maximize scores across all criteria.

## 1. Literature Review (10/10)

**Target:** "Reads like a publishable paper introduction. Deep historical awareness."

- [ ] Group papers logically: **Statistical methods** (OCSVM, Isolation Forest, GMM) vs **Neural methods** (AE, VAE, MAE, Transformers)
- [ ] **Pit papers against each other** to highlight the gap: e.g., "OCSVM lacks representation learning; pure AE lacks interpretable boundaries"
- [ ] Explicitly link to **why you chose** MAE + OCSVM hybrid
- [ ] Use **correct citations** (Tavallaee et al. NSL-KDD, Schölkopf OCSVM, Devlin BERT, Kingma VAE)
- [ ] Avoid: laundry lists, SOTA buzzwords without definition, wrong attribution

**Key references to include:**
- Tavallaee et al. (NSL-KDD)
- Schölkopf et al. (One-Class SVM)
- Devlin et al. (BERT masking)
- Kingma & Welling (VAE)
- Ruff et al. (Deep one-class)

---

## 2. Dataset Quality & EDA (10/10)

**Target:** "Non-obvious patterns that dictate modeling strategy. Intimate feel for data."

- [ ] **Characterize distribution**: heavy tails (duration, src_bytes) → log-transform; multi-modality
- [ ] **Identify data leakage risks**: e.g., temporal ordering, test-only attack types
- [ ] **Quantify class imbalance**: normal vs attack; attack type distribution
- [ ] **Missing/duplicate analysis**: dst_host_srv_rerror_rate may be all-NaN in some splits
- [ ] **Correlation matrix** with interpretation (e.g., error rates correlate)
- [ ] EDA **dictates preprocessing**: "We saw skew → applied log transform"
- [ ] Run `python scripts/run_eda.py` and include figures in report

---

## 3. Feature Engineering (10/10)

**Target:** "Feature engineering IS the breakthrough. Novel representation simplifies the task."

- [ ] **Domain-informed features**: bytes_ratio, error_interaction, host_service_density
- [ ] **Test features**: keep what works, discard what doesn't (ablation on features)
- [ ] **Log-transform** for heavy-tailed columns (duration, src_bytes, dst_bytes)
- [ ] **Handle unseen categories** (e.g., new services in test) with __UNK__
- [ ] Optional: PCA/t-SNE for visualization; do not blindly one-hot everything
- [ ] Justify: "We created X because it captures Y (e.g., traffic asymmetry)"

---

## 4. Theoretical Rigor (10/10)

**Target:** "Correct notation. Intuitive proofs or derivations. Implications of violating assumptions."

- [ ] **OCSVM**: Explain $\nu$ (upper bound on anomaly fraction), RBF kernel, decision boundary
- [ ] **MAE**: BERT-style masking, loss only on masked positions, Xavier init
- [ ] **Stationarity / IID**: Acknowledge violations (bursty traffic); justify mitigations
- [ ] **Bias–variance**: MAE complexity vs OCSVM simplicity; hybrid balances both
- [ ] **EM / convergence**: If using GMM, explain; for MAE, gradient descent convergence

---

## 5. Model Application (10/10)

**Target:** "Solution customized. Rigorous failure analysis. Defend choices mathematically."

- [ ] **Ablation table**: A (OCSVM only), B (MAE only), C (Hybrid)
- [ ] **Metrics**: F1 (primary for imbalanced), Precision, Recall, AUC
- [ ] **Hyperparameter tuning**: nu, gamma for OCSVM; mask_ratio, layers for MAE
- [ ] **Failure analysis**: Which attack types does the model miss? Why?
- [ ] Run `python scripts/run_ablation.py` and fill Table 1 in report

---

## 6. GitHub Repository & Code Quality (10/10)

**Target:** "Professional README, reproducible setup, modular clean code, consistent commits."

- [ ] **README**: Problem, approach, results, usage (install, run commands)
- [ ] **requirements.txt** with pinned versions
- [ ] **Modular structure**: src/data, src/features, src/models, scripts
- [ ] **config.yaml** for reproducibility
- [ ] **Run instructions**: `download_data → run_eda → train_mae → run_ablation`
- [ ] Regular commits with meaningful messages

---

## 7. Project Report LaTeX (10/10)

**Target:** "Professionally written, excellent formatting, clear Intro→Method→Results."

- [ ] **Abstract**: Problem, approach, key result (1 paragraph)
- [ ] **Introduction**: Problem statement, contribution (3 bullets)
- [ ] **Related Work**: Grouped, linked to your design
- [ ] **Methods**: Data, MAE, OCSVM, hybrid pipeline
- [ ] **Results**: Ablation table, architecture diagram
- [ ] **Conclusion**: Summary + future work
- [ ] Proper math notation, correct citations
- [ ] Insert actual ablation values from `results/ablation_table.csv`

---

## 8. Presentation / Video (10/10)

**Target:** "Clearly explains motivation, smooth working demo, crystal clear audio, professional."

- [ ] **Problem motivation** (1–2 min): Why anomaly detection for networks?
- [ ] **Approach** (2–3 min): MAE + OCSVM, architecture diagram
- [ ] **Demo** (3–4 min): Run pipeline, show ablation table
- [ ] **Results** (2 min): Key metrics, failure cases
- [ ] 10 min total; good lighting, no background noise
- [ ] Record with OBS or similar; 1080p preferred

---

## 9. Viva Voce (10/10)

**Target:** "Clearly explains full system, answers code questions, justifies design, unquestionable ownership."

- [ ] Know **data flow**: Raw → Preprocess → MAE → Embeddings → OCSVM → Predict
- [ ] Be able to **explain any line** in key files (loader, mae.py, ocsvm)
- [ ] **Why MAE?** Self-supervised on benign only; learns robust representations
- [ ] **Why OCSVM?** Interpretable boundary; nu controls false positive rate
- [ ] **Why hybrid?** Combines learned repr + explicit boundary
- [ ] **Failure cases**: Which attacks are hard? (U2R, R2L often low recall)
- [ ] Practice explaining: masking ratio, Xavier init, log-transform rationale

---

## Quick Commands

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_eda.py
python figures/architecture_diagram.py
python scripts/train_mae.py --epochs 50
python scripts/run_ablation.py
# Fill results/ablation_table.csv into reports/phase1_report.tex Table 1
```
