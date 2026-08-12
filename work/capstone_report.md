# Capstone Report — Leakage-Aware Content Decline Prioritization

- **Author:** Umer Sajid
- **Lane:** Content Refresh Prioritization
- **Repo:** `UmerSajid842/flyrankmlproject`
- **Date:** 2026-08-12

> **Working title.** Prioritising visible, ageing content pages for refresh using leakage-aware prior-window and contextual signals.

## Abstract

This capstone asks whether an SEO or content editor can use page context and prior-month signals to prioritise a review queue for potentially declining content. It uses the repository's anonymised snapshot of 30,000 content pages across 32 pseudonymised clients and evaluates transfer to seven clients held out from training. A leakage-aware random-forest model uses only stable/contextual variables and preceding-30-day counts; it is compared with a transparent visibility, freshness, and content-thinness baseline. On the held-out clients, the model reached Precision@50 of 0.540 versus 0.340 for the baseline, but the 0.511 decline base rate makes this a modest prioritisation result rather than an automated-action threshold. The output is a human review queue for directional decision support, not a causal estimate, a traffic forecast beyond the snapshot, or a claim about Google's algorithm.

## 1. Problem framing

This project supports an **SEO or content editor** deciding which pages to investigate for a refresh first when the team has limited editorial capacity. The unit of analysis is one pseudonymised content page. The output is a ranked review queue rather than an autonomous action: a human editor reviews the highest-priority pages, diagnoses the cause, and chooses whether to refresh, monitor, or take another action. A wrong positive consumes editorial time; a wrong negative may leave a genuinely declining page unattended. The analysis uses machine learning because the decision depends on a combination of prior visibility, content age, freshness, content characteristics, and search-context signals that are difficult to combine consistently in a single fixed rule.

The target is an **observed current-snapshot decline proxy**: `trend_direction == "down"`, which means the most recent 30 days of impressions are more than 20% below the preceding 30 days. The primary prioritisation metric is **Precision@50** on held-out clients, reported alongside the held-out decline base rate. ROC-AUC and average precision are supplementary discrimination checks.

## 2. Data safety

The analysis uses only the repository's bundled anonymised data: 30,000 pseudonymised content pages across 32 pseudonymised clients. No names, URLs, titles, or private queries are present in this project. `content_id` and `client_id` are used only for output reference and grouped evaluation; they are never model features.

To prevent direct and time-window leakage, the model excludes `trend_direction`, `trend_pct`, `is_declining_label`, all `*_last_30d` columns, all 90-day traffic totals and derived rates, and the provider/model metadata. Those fields either define the outcome or overlap the outcome period. The predictive feature set is restricted to context and the **preceding** 30-day counts, plus content age/freshness. Numeric missing values are median-imputed with missingness indicators; categorical missing values form an explicit `missing` category rather than being silently treated as zero.

## 3. Baseline

The baseline is a transparent review-priority score, evaluated on the same held-out client set as the model:

`0.50 × percentile(log(1 + prior-month impressions)) + 0.30 × percentile(days since last update) + 0.20 × content thinness`.

It is deliberately simple and label-free. It prioritises pages with meaningful recent visibility, stale content, and relatively low word count. The comparison asks whether the learned model identifies a larger share of current-snapshot declining pages among the first 50 pages reviewed.

## 4. Model / analysis

The model is a random-forest classifier with 300 trees, class balancing, median numeric imputation with missingness indicators, and one-hot encoding for categorical features. It uses a single client-held-out split (`GroupShuffleSplit`, 20% test set, random seed 42). Grouping by client tests whether the prioritisation approach transfers to a client portfolio not seen during training rather than merely memorising client-level patterns.

The exact features are saved to `work/outputs/capstone_metrics.json` on each run. Numeric inputs are search context, content length, log prior-month impressions/clicks/sessions, content age, and days since last update. Categorical inputs are competition level, content type, main intent, and the age/freshness/length tiers.

## 5. Evaluation

The analysis was freshly run with `python work/scripts/run_capstone.py` on the bundled 30,000-row anonymised dataset. The metric table below compares the baseline and model on the **same seven held-out clients** (6,163 pages).

| Measure | Result from fresh run |
|---|---:|
| Held-out decline base rate | 0.511 |
| Baseline Precision@50 | 0.340 |
| Model Precision@50 | 0.540 |
| Baseline Precision@100 | 0.340 |
| Model Precision@100 | 0.460 |
| Model ROC-AUC | 0.659 |
| Model average precision | 0.616 |

The model selects 27 pages labelled `down` among its first 50 versus 17 for the transparent baseline: a **20-percentage-point Precision@50 improvement** over that rule. However, the held-out decline base rate is already 0.511, so the 0.540 Precision@50 result is only modestly above the raw prevalence. The result is therefore a measured improvement over the selected baseline, not evidence of a high-confidence automated refresh decision.

The ranked queue contains a model score, transparent baseline score, and non-label reason codes. False positives are pages selected for review that are not labelled `down`; false negatives are declining pages outside the selected top 50. The output is a prioritisation aid, so a human review step remains mandatory.

## 6. Interpretation

On the held-out clients, **prior-month impressions** were by far the strongest measured signal: permuting `log_impressions_prev_30d` reduced ROC-AUC by about 0.141. Prior-month clicks (about 0.011), content age (about 0.009), and prior-month sessions (about 0.006) made substantially smaller contributions. In plain language, the model mostly distinguished current-snapshot decline using whether a page had an established measurable search footprint in the preceding month; older pages and prior engagement added limited incremental evidence.

Several intended contextual signals, including content type, search-volume context, and content-length measures, had near-zero or negative permutation importance in this split. This is a valid negative result: these features did not materially improve discrimination after the prior-window visibility measures were present. Permutation importance measures model dependence in this particular evaluation split; it does not show that any feature causes a decline.

## 7. Recommendation

A FlyRank editor should review the first 50 rows in `work/outputs/capstone_ranked_queue.csv`, beginning with entries whose reason codes show both meaningful prior visibility and a manageable content issue such as staleness or short content. The recommended action is **review for refresh**, not automatic refresh. Confidence is limited by the anonymous snapshot design, the proxy outcome, and the absence of a post-refresh experiment.

## 8. Reproducibility

From the repository root, install the project requirements and run:

```bash
pip install -r requirements.txt
python work/scripts/run_capstone.py
```

The analysis uses random seed `42`. It reads only `data/raw/content_refresh_anonymized.csv` and writes derived artifacts to `work/outputs/`, which should remain uncommitted. Re-run it after any feature or model change before updating the metric table above.

## Acknowledgements and data credit

This project uses the anonymised FlyRank ML Internship starter dataset supplied with this repository. The source includes 30,000 pseudonymised content pages and contains no client names, domains, URLs, titles, or private queries. The project follows the repository's `DATA_USE.md` restrictions and its data dictionary, and it credits the FlyRank ML Internship repository as the technical and data foundation.

---

> **Claims boundary:** All findings in this report are **observed, measured, directional, and decision-support only**. They do not predict Google's algorithm, establish causality, or identify private clients.
