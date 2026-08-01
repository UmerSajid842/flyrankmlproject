# ML-10 — Content Action Playbook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UmerSajid842/flyrankmlproject/blob/main/work/notebooks/w07_action_playbook.ipynb?flush_cache=true)

This skeleton is yours to fill. Work the sections **in order** — each one has a one-line hint. Simple words, honest numbers.

> Working with an AI assistant? Tell it to read `skills/README.md` first and load the one skill this assignment names on its card.

## 1. Ranked actions + reason codes

*The queue: what to do first, and why, in words a human trusts.*


```python
from pathlib import Path
import pandas as pd

candidate_paths = [
    Path.cwd() / "outputs/refresh_queue.csv",
    Path.cwd().parent / "outputs/refresh_queue.csv",
    Path.cwd().parent.parent / "outputs/refresh_queue.csv",
]
queue_path = next((p for p in candidate_paths if p.exists()), None)
if queue_path is None:
    raise FileNotFoundError("The final refresh queue has not been generated yet.")

queue = pd.read_csv(queue_path)
queue.head()

```

## 2. Intended use and limits

*Who uses this, for what — and where it stops being valid.*


```python
queue[["content_id", "final_refresh_score", "suggested_action", "final_reason_codes"]].head(10)

```

## 3. Human review + the no-go list

*What a person must check before acting. What should never be automated.*


```python
queue["suggested_action"].value_counts().to_frame(name="count")

```

## 4. Monitoring / retrain triggers

*What would tell you the recommendations went stale?*


```python
queue.groupby("suggested_action")["final_refresh_score"].mean().to_frame(name="avg_final_score")

```

## 5. Exports for the paper

*Write the queue (and any figures you want to reuse) to work/outputs/ — your paper builds on these files.*


```python
queue.loc[queue["final_refresh_score"].idxmax(), ["content_id", "final_refresh_score", "suggested_action", "final_reason_codes"]]

```

## Self-check

Before you submit, confirm each line honestly:

- [ ] Every section above is filled — markdown thinking AND the code that backs it
- [ ] The notebook runs top to bottom with no errors (Runtime → Run all)
- [ ] No client names, URLs, or private queries anywhere
- [ ] My claims use careful words: observed, measured, directional, decision-support
- [ ] Committed to my repo under `work/notebooks/` — then submit your repo URL on the card. Done.
