# ML-09 — Validation and Research Claim Audit

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UmerSajid842/flyrankmlproject/blob/main/work/notebooks/w06_validation_audit.ipynb?flush_cache=true)

This skeleton is yours to fill. Work the sections **in order** — each one has a one-line hint. Simple words, honest numbers.

> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one skill this assignment names on its card.

## 1. Two paper findings + my methodology questions

*Pick two findings from the FlyRank research paper. For each: where does the label come from, and does the validation design carry the claim? Constructive tone.*


```python
from pathlib import Path
import pandas as pd

candidate_paths = [
    Path.cwd() / "data/raw/content_refresh_anonymized.csv",
    Path.cwd().parent / "data/raw/content_refresh_anonymized.csv",
    Path.cwd().parent.parent / "data/raw/content_refresh_anonymized.csv",
]
data_path = next((p for p in candidate_paths if p.exists()), None)
if data_path is None:
    raise FileNotFoundError("Starter data not found in the expected repo locations.")

df = pd.read_csv(data_path)
df[["ctr", "avg_position", "trend_direction", "trend_pct"]].head()

```

## 2. My model under an honest split (before/after)

*Re-run your Week-5 model under a grouped or time-aware split. Show both numbers.*


```python
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["client_id"]))
train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
feature_cols = ["ctr", "avg_position", "engagement_rate", "scroll_rate", "content_age_days", "word_count"]
y_train = train_df["trend_direction"].eq("down")
y_test = test_df["trend_direction"].eq("down")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(train_df[feature_cols].fillna(0), y_train)
scores = model.predict_proba(test_df[feature_cols].fillna(0))[:, 1]
scores[:5]

```

## 3. Leakage audit

*The same hunt from Week 3, on your final feature set.*


```python
roc_auc_score(y_test, scores)

```

## 4. Claim rewrite

*Take your own boldest sentence and rewrite it in safe language: observed, measured, directional, decision-support.*


```python
top_candidates = test_df.iloc[scores.argsort()[::-1], :][["content_id", "trend_direction", "ctr", "avg_position"]].head(20)
top_candidates

```

## Self-check

Before you submit, confirm each line honestly:

- [ ] Every section above is filled — markdown thinking AND the code that backs it
- [ ] The notebook runs top to bottom with no errors (Runtime → Run all)
- [ ] No client names, URLs, or private queries anywhere
- [ ] My claims use careful words: observed, measured, directional, decision-support
- [ ] Committed to my repo under `work/notebooks/` — then submit your repo URL on the card. Done.
