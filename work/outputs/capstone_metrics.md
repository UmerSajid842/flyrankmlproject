# Capstone run metrics

| Measure | Value |
|---|---:|
| Rows | 30,000 |
| Pseudonymized clients | 32 |
| Train / held-out rows | 23,837 / 6,163 |
| Train / held-out clients | 25 / 7 |
| Held-out decline base rate | 0.511 |
| Model ROC-AUC | 0.659 |
| Model average precision | 0.616 |
| Model Precision@50 | 0.540 |
| Baseline Precision@50 | 0.340 |
| Model Precision@100 | 0.460 |
| Baseline Precision@100 | 0.340 |

**Interpretation boundary.** This run prioritizes pages that show an observed decline in the
current snapshot using prior-window and contextual signals. It is directional decision support,
not a causal estimate, a forecast beyond this snapshot, or a claim about Google's algorithm.
