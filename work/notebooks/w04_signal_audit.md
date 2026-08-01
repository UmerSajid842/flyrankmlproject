# ML-06 — Signal Audit: Do the Flags Hold?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UmerSajid842/flyrankmlproject/blob/main/work/notebooks/w04_signal_audit.ipynb?flush_cache=true)

This skeleton is yours to fill. Work the sections **in order** — each one has a one-line hint. Simple words, honest numbers.

> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one skill this assignment names on its card.

## 1. Distributions

*Look before deciding: distributions of your key fields. Note the heavy tails.*


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
signal_frame = df[["ctr", "engagement_rate", "scroll_rate", "avg_position", "trend_direction", "trend_pct"]].copy()
signal_frame.describe().T

```

## 2. Signal test #1 / #2 / #3 (verdict each)

*Three safe signals, each with a mini-test and a verdict: CONFIRMED / OPPOSITE / MIXED / FALSE.*


```python
signal_frame.groupby("trend_direction")["ctr"].mean().to_frame(name="mean_ctr")

```

## 3. The flag-linked test

*Pick a signal one of FlyRank's real flags relies on. Does the data support the rule's assumption?*


```python
signal_frame["signal_flag"] = signal_frame["trend_direction"].eq("down").astype(int)
signal_frame.groupby("signal_flag")["engagement_rate"].mean().to_frame(name="mean_engagement_rate")

```

## 4. What this means in practice

*Two or three sentences: what a content team should take from this.*


```python
summary = signal_frame.groupby("trend_direction").agg(
    avg_ctr=("ctr", "mean"),
    avg_engagement=("engagement_rate", "mean"),
    avg_position=("avg_position", "mean"),
)
summary

```

## Self-check

Before you submit, confirm each line honestly:

- [ ] Every section above is filled — markdown thinking AND the code that backs it
- [ ] The notebook runs top to bottom with no errors (Runtime → Run all)
- [ ] No client names, URLs, or private queries anywhere
- [ ] My claims use careful words: observed, measured, directional, decision-support
- [ ] Committed to my repo under `work/notebooks/` — then submit your repo URL on the card. Done.
