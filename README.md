# Fairness-Aware ECG-Based Disease Prediction in Wearable Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![MobiHealth 2026](https://img.shields.io/badge/MobiHealth-2026-orange.svg)](https://mobihealth.name/)

Official implementation of the paper accepted at **MobiHealth 2026 (EAI), Heraklion, Greece**.

> Yesmin, F. & Shirmin, N. (2026). *Fairness-Aware Representation Learning for
> ECG-Based Disease Prediction in Wearable Systems.*
> EAI International Conference on Mobile Health (MobiHealth 2026), Heraklion, Greece.
> Preprint: [ResearchGate](https://www.researchgate.net/publication/396441645)

Part of the [FairHealth](https://github.com/Farjana-Yesmin/fairhealth) library —
`pip install fairhealth`

---

## Abstract

Machine learning models for ECG-based disease prediction in wearable devices
often exhibit demographic biases, exacerbating healthcare disparities. We propose
a fairness-aware representation learning framework using adversarial debiasing
tailored for biosignals, focusing on inferior myocardial infarction (IMI)
classification. Evaluated on a 20% subsample of PTB-XL (4,367 records), our
method achieves AUROC=0.8472 and improves disparate impact (DI) for sex
from 0.23 to 0.71. Intersectional analysis reduces AUROC disparity across
sex-age groups from 0.15 (baseline) to 0.08. Robustness validation across three
random subsamples (seeds 42, 123, 456) confirms stability: AUROC 0.847±0.031,
DI 0.71±0.048.

---

## Results

### Overall Performance (PTB-XL 20% Subsample)

| Method | AUROC | Accuracy | F1 | DI-Sex |
|---|---|---|---|---|
| Baseline CNN | 0.92 | 0.85 | 0.55 | 0.23 |
| Reweighting | 0.90 | 0.84 | 0.53 | 0.30 |
| FairMixup | 0.88 | 0.83 | 0.52 | 0.35 |
| AdvFair | 0.85 | 0.82 | 0.51 | 0.45 |
| GroupDRO | 0.84 | 0.81 | 0.50 | 0.50 |
| **Ours** | **0.8472** | **0.81** | **0.50** | **0.71** |

### Per-Group Performance

| Group | AUROC | F1 |
|---|---|---|
| Male | 0.83 | 0.51 |
| Female | 0.81 | 0.49 |
| Age < 40 | 0.79 | 0.42 |
| Age 40–59 | 0.84 | 0.52 |
| Age ≥ 60 | 0.82 | 0.50 |

### Intersectional Analysis (Sex × Age Groups)

| Group | AUROC Baseline | AUROC Ours | F1 Baseline | F1 Ours | n |
|---|---|---|---|---|---|
| Male, < 40 | 0.79 | 0.81 | 0.40 | 0.44 | 87 |
| Male, 40–59 | 0.93 | 0.85 | 0.58 | 0.53 | 245 |
| Male, ≥ 60 | 0.91 | 0.84 | 0.56 | 0.52 | 312 |
| Female, < 40 | 0.78 | 0.77 | 0.36 | 0.39 | 63 |
| Female, 40–59 | 0.89 | 0.83 | 0.54 | 0.50 | 178 |
| Female, ≥ 60 | 0.87 | 0.80 | 0.52 | 0.48 | 270 |
| **AUROC Gap** | **0.15** | **0.08** | — | — | — |
| **F1 Gap** | **0.22** | **0.14** | — | — | — |

Younger females (Female, <40, n=63) face the greatest compounded disadvantage.
Our framework reduces their AUROC gap from 0.15 to 0.08 and F1 gap from 0.22 to 0.14.

### Robustness Validation

Across 3 random subsamples (seeds 42, 123, 456):
- AUROC: 0.847 ± 0.031
- DI: 0.71 ± 0.048

Narrow confidence intervals confirm results are not an artefact of a single random draw.

---

## Model Architecture
Input ECG (12 leads, 10s @ 100Hz)
↓
Encoder: 1D CNN (12→64→128) + FC (31,616→256)
↓
┌─────────────────────────────────────────┐
│  Classifier (256→1, sigmoid)  →  IMI   │
│  Adversary Sex (256→128→2)             │
│  Adversary Age (256→128→3)             │
└─────────────────────────────────────────┘
Loss: L = L_class − λ(L_adv_sex + L_adv_age)
λ = 0.3 (selected by ablation over {0.1, 0.2, 0.3, 0.5, 0.7, 1.0})
Gradient reversal layer forces the encoder to learn
demographic-invariant representations.

---

## Fairness Metrics

| Metric | Description |
|---|---|
| DI (Disparate Impact) | Ratio of positive rates across groups (↑ is better, 1.0 = perfect) |
| DPD (Demographic Parity Difference) | Difference in positive rates (↓ is better, 0 = perfect) |
| EOD (Equal Opportunity Difference) | Difference in true positive rates (↓ is better) |

---

## Quick Start

```bash
pip install wfdb torch>=2.0 pandas scikit-learn matplotlib
```

```python
# Dataset downloads automatically from PhysioNet
# Run the notebook sequentially:
# eai_mobihealth2026 Fairness-Aware ECG Disease Prediction in Wearable Systems.py
```

**Configuration:**
```python
config = {
    'learning_rate': 0.001,
    'batch_size': 128,
    'epochs': 20,
    'lambda_adversary': 0.3,
    'signal_length': 1000,    # 10s @ 100Hz
    'num_leads': 12,
    'random_seeds': [42, 123, 456]  # for robustness validation
}
```

---

## Dataset

**PTB-XL ECG Dataset** (PhysioNet, free access)
- Full dataset: 21,837 records
- This study: 20% subsample (4,367 records) — Colab free-tier constraint
- Splits: 70% train / 15% val / 15% test, stratified by IMI
- Positive cases: 431 IMI across all splits
- SMOTE applied to Age <40 group to address class imbalance

Download: Automatic via `wfdb` library from PhysioNet.
No data use agreement required for research use.

---

## Experimental Setup

| Item | Value |
|---|---|
| Hardware | Google Colab free tier (T4 GPU, ~12GB RAM) |
| Framework | PyTorch 2.6.0 |
| ECG library | wfdb |
| Validation | 5-fold cross-validation |
| Runtime | ~2 hours per fold |

---

## Use with FairHealth

This paper's fairness metrics are available in the
[FairHealth](https://github.com/Farjana-Yesmin/fairhealth) library:

```python
from fairhealth.fairness.metrics import (
    demographic_parity_diff,
    disparate_impact,
    equalized_odds_diff,
    intersectional_fairness
)

# Audit ECG model across sex and age groups
di  = disparate_impact(y_pred, sensitive=sex_array)
dpd = demographic_parity_diff(y_pred, sensitive=age_array)
```

---

## Citation

```bibtex
@inproceedings{yesmin2026ecgfairness,
  author    = {Yesmin, Farjana and Shirmin, Nusrat},
  title     = {Fairness-Aware Representation Learning for ECG-Based
               Disease Prediction in Wearable Systems},
  booktitle = {Proceedings of EAI MobiHealth 2026},
  year      = {2026},
  note      = {Preprint: https://www.researchgate.net/publication/396441645}
}
```

---

## Authors

- **Farjana Yesmin** — [farjana-yesmin.github.io](https://farjana-yesmin.github.io)
- **Nusrat Shirmin** — Independent Researcher, Dhaka, Bangladesh

## License

MIT License — see [LICENSE](LICENSE) file.
