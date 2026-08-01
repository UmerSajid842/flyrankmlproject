# ML-05 — Feature Vector and Leakage/Privacy Check

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UmerSajid842/flyrankmlproject/blob/main/work/notebooks/w03_feature_leakage_check.ipynb?flush_cache=true)

This skeleton is yours to fill. Work the sections **in order** — each one has a one-line hint. Simple words, honest numbers.

> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one skill this assignment names on its card.

## 1. Build the feature vector

*Code that actually builds it — engineered features, categorical handling, fills.*


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
safe_features = [
    c for c in df.columns
    if c not in {"content_id", "client_id", "trend_direction", "trend_pct"}
]
feature_frame = df[safe_features].copy()
feature_frame.head()

```

## 2. Feature notes (meaning, missing, categorical, available-when?)

*For each feature: what it means, how missing values are handled, and whether it exists BEFORE the moment you predict.*


```python
feature_frame.dtypes.to_frame(name="dtype").head(12)

```

## 3. The leakage hunt

*Attack your own features: label-derived columns, future windows, product flags. Show the test.*


```python
label_like = ["trend_direction", "trend_pct"]
exclude_reasons = {
    "trend_direction": "derived from the same signal used in the label",
    "trend_pct": "same label family; cannot be a feature",
}
for col in label_like:
    print(col, "->", exclude_reasons[col])

```

## 4. What I excluded and why

*The list of fields you refused to use — with one line of why each.*


```python
safe_feature_frame = feature_frame.drop(columns=[c for c in feature_frame.columns if feature_frame[c].isna().mean() > 0.25])
safe_feature_frame = safe_feature_frame.fillna(safe_feature_frame.median(numeric_only=True))
safe_feature_frame.head()

```

## Self-check

Before you submit, confirm each line honestly:

- [ ] Every section above is filled — markdown thinking AND the code that backs it
- [ ] The notebook runs top to bottom with no errors (Runtime → Run all)
- [ ] No client names, URLs, or private queries anywhere
- [ ] My claims use careful words: observed, measured, directional, decision-support
- [ ] Committed to my repo under `work/notebooks/` — then submit your repo URL on the card. Done.
