# ML-08 — Capstone Modeling Lane

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UmerSajid842/flyrankmlproject/blob/main/work/notebooks/w05_model.ipynb?flush_cache=true)

This skeleton is yours to fill. Work the sections **in order** — each one has a one-line hint. Simple words, honest numbers.

> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one skill this assignment names on its card.

## 1. Method choice and why

*Which method from the toolkit, and why it fits your lane.*


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
df[["ctr", "avg_position", "engagement_rate", "scroll_rate", "trend_direction"]].head()

```

## 2. Split design

*Grouped by client? Time-aware? Say why this split is honest for your question.*


```python
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["client_id"]))
train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
train_df.shape, test_df.shape

```

## 3. Train + compare vs my baseline

*Same data, same metric, same split as your Week-4 baseline. Show the table.*


```python
label = "is_declining_label" if "is_declining_label" in df.columns else "trend_direction"
y_train = train_df[label].eq("down") if label == "trend_direction" else train_df[label]
y_test = test_df[label].eq("down") if label == "trend_direction" else test_df[label]
feature_cols = ["ctr", "avg_position", "engagement_rate", "scroll_rate", "content_age_days", "word_count"]
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(train_df[feature_cols].fillna(train_df[feature_cols].median()), y_train)
preds = model.predict(test_df[feature_cols].fillna(train_df[feature_cols].median()))
precision_score(y_test, preds)

```

## 4. Errors and interpretation

*Where is the model wrong? What does it lean on? A short error analysis beats a big metric table.*


```python
feature_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
feature_importance.head(10)

```

## Self-check

Before you submit, confirm each line honestly:

- [ ] Every section above is filled — markdown thinking AND the code that backs it
- [ ] The notebook runs top to bottom with no errors (Runtime → Run all)
- [ ] No client names, URLs, or private queries anywhere
- [ ] My claims use careful words: observed, measured, directional, decision-support
- [ ] Committed to my repo under `work/notebooks/` — then submit your repo URL on the card. Done.
