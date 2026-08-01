# ML-07 — Baseline Action Score and Top-20 Review

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UmerSajid842/flyrankmlproject/blob/main/work/notebooks/w04_baseline_score.ipynb?flush_cache=true)

This skeleton is yours to fill. Work the sections **in order** — each one has a one-line hint. Simple words, honest numbers.

> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one skill this assignment names on its card.

## 1. My rule and its reason codes

*Write the rule in plain words first. Then the reason codes it can output.*


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
rule_df = df[["content_id", "client_id", "ctr", "avg_position", "trend_direction", "trend_pct"]].copy()
rule_df["baseline_score"] = (rule_df["trend_direction"].eq("down").astype(int) * 0.6) + ((rule_df["ctr"].rank(pct=True)) * 0.3) + ((rule_df["avg_position"].rank(method="first", pct=True)) * 0.1)
rule_df.sort_values("baseline_score", ascending=False).head(20)

```

## 2. Build the ranked queue (writes the CSV)

*Code the score, rank everything, write work/outputs/baseline_action_score.csv.*


```python
baseline_queue = rule_df.sort_values("baseline_score", ascending=False).head(20).copy()
baseline_queue[["content_id", "baseline_score", "trend_direction", "trend_pct", "ctr", "avg_position"]]

```

## 3. Top-20 review

*For each of the top 20: action, reason code, confidence note, and what would make it wrong.*


```python
baseline_queue[["content_id", "baseline_score", "trend_direction", "ctr", "avg_position"]].head(10)

```

## 4. Weak picks + leakage check

*Which picks look wrong and why? Confirm no product flags or future windows leaked in.*


```python
weak_picks = baseline_queue[baseline_queue["baseline_score"] < 0.4].copy()
weak_picks.head()

```

## Self-check

Before you submit, confirm each line honestly:

- [ ] Every section above is filled — markdown thinking AND the code that backs it
- [ ] The notebook runs top to bottom with no errors (Runtime → Run all)
- [ ] No client names, URLs, or private queries anywhere
- [ ] My claims use careful words: observed, measured, directional, decision-support
- [ ] Committed to my repo under `work/notebooks/` — then submit your repo URL on the card. Done.
