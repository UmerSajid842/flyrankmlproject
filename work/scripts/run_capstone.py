"""Run the FlyRank capstone: leakage-aware decline prioritization.

This analysis supports a content/SEO editor's decision about which pages in a
new client portfolio deserve investigation first. It uses the bundled
anonymized 30k-page snapshot and never writes the raw dataset to outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
TOP_K = 50
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
OUTPUT_DIR = ROOT / "work" / "outputs"

# All prior-window signals below are disjoint from the target's last-30-day
# outcome window. We intentionally omit ids, target-derived fields, every
# last-30-day metric, and every 90-day metric/rate because they overlap the
# outcome period and would make a forward-looking claim unsafe.
NUMERIC_FEATURES = [
    "search_volume",
    "competition",
    "cpc",
    "word_count",
    "char_count",
    "log_impressions_prev_30d",
    "log_clicks_prev_30d",
    "log_sessions_prev_30d",
    "content_age_days",
    "days_since_last_update",
]
CATEGORICAL_FEATURES = [
    "competition_level",
    "content_type",
    "main_intent",
    "age_tier",
    "freshness_tier",
    "word_count_tier",
    "char_count_tier",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def precision_at_k(y_true: pd.Series, scores: np.ndarray, k: int = TOP_K) -> float:
    frame = pd.DataFrame({"y": pd.Series(y_true).to_numpy(), "score": scores})
    return float(frame.sort_values("score", ascending=False).head(min(k, len(frame)))["y"].mean())


def percentile_rank(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rank(pct=True).fillna(0.0)


def build_baseline_score(frame: pd.DataFrame) -> pd.Series:
    """Transparent no-label priority rule, computed from prior-window/context only."""
    visibility = percentile_rank(np.log1p(frame["impressions_prev_30d"]))
    freshness = percentile_rank(frame["days_since_last_update"])
    words = pd.to_numeric(frame["word_count"], errors="coerce")
    thinness = 1 - percentile_rank(words.fillna(words.median()))
    return (0.50 * visibility + 0.30 * freshness + 0.20 * thinness).clip(0, 1)


def reason_codes(row: pd.Series, median_prior_impressions: float) -> str:
    reasons: list[str] = []
    if row["impressions_prev_30d"] >= median_prior_impressions:
        reasons.append("visible_prior_month")
    if row["days_since_last_update"] >= 180:
        reasons.append("stale_content")
    if pd.notna(row["word_count"]) and row["word_count"] < 1200:
        reasons.append("short_content")
    return "|".join(reasons) if reasons else "review_context"


def build_model() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_transformer = Pipeline(
        steps=[("impute", SimpleImputer(strategy="constant", fill_value="missing")),
               ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Starter data not found: {RAW_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    for metric in ("impressions_prev_30d", "clicks_prev_30d", "sessions_prev_30d"):
        df[f"log_{metric}"] = np.log1p(pd.to_numeric(df[metric], errors="coerce").clip(lower=0))
    df["target_decline"] = df["trend_direction"].eq("down").astype(int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_index, test_index = next(splitter.split(df, y=df["target_decline"], groups=df["client_id"]))
    train = df.iloc[train_index].copy()
    test = df.iloc[test_index].copy()
    y_train = train["target_decline"]
    y_test = test["target_decline"]

    baseline_scores = build_baseline_score(test)
    model = build_model()
    model.fit(train[FEATURES], y_train)
    model_scores = model.predict_proba(test[FEATURES])[:, 1]

    base_rate = float(y_test.mean())
    model_metrics = {
        "roc_auc": float(roc_auc_score(y_test, model_scores)),
        "average_precision": float(average_precision_score(y_test, model_scores)),
        "precision_at_50": precision_at_k(y_test, model_scores, TOP_K),
        "precision_at_100": precision_at_k(y_test, model_scores, 100),
        "precision_at_threshold_0_50": float(precision_score(y_test, (model_scores >= 0.50).astype(int), zero_division=0)),
    }
    baseline_metrics = {
        "precision_at_50": precision_at_k(y_test, baseline_scores.to_numpy(), TOP_K),
        "precision_at_100": precision_at_k(y_test, baseline_scores.to_numpy(), 100),
    }

    importance = permutation_importance(
        model,
        test[FEATURES],
        y_test,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_frame = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean_auc_drop": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean_auc_drop", ascending=False)
    importance_frame.to_csv(OUTPUT_DIR / "capstone_feature_importance.csv", index=False)

    queue = test[[
        "content_id", "client_id", "impressions_prev_30d", "clicks_prev_30d",
        "sessions_prev_30d", "word_count", "content_age_days", "days_since_last_update",
        "content_type", "main_intent",
    ]].copy()
    queue["model_decline_score"] = model_scores
    queue["baseline_priority_score"] = baseline_scores.to_numpy()
    median_prior_impressions = float(test["impressions_prev_30d"].median())
    queue["reason_codes"] = test.apply(lambda row: reason_codes(row, median_prior_impressions), axis=1)
    queue["suggested_action"] = np.where(
        queue["model_decline_score"] >= np.quantile(model_scores, 0.95),
        "review_for_refresh",
        "monitor",
    )
    queue = queue.sort_values("model_decline_score", ascending=False).reset_index(drop=True)
    queue.insert(0, "priority_rank", np.arange(1, len(queue) + 1))
    queue.head(250).to_csv(OUTPUT_DIR / "capstone_ranked_queue.csv", index=False)

    metrics = {
        "analysis_name": "Leakage-aware decline prioritization",
        "data": {"rows": int(len(df)), "clients": int(df["client_id"].nunique())},
        "split": {
            "method": "GroupShuffleSplit by client_id",
            "random_state": RANDOM_STATE,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_clients": int(train["client_id"].nunique()),
            "held_out_clients": int(test["client_id"].nunique()),
        },
        "target": "trend_direction == 'down' (a current-snapshot decline proxy)",
        "base_rate_test": base_rate,
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "feature_policy": {
            "included": FEATURES,
            "excluded": [
                "content_id", "client_id", "trend_direction", "trend_pct", "is_declining_label",
                "impressions_last_30d", "clicks_last_30d", "sessions_last_30d",
                "all 90-day traffic totals and derived rates",
                "provider_used", "model_used",
            ],
        },
    }
    (OUTPUT_DIR / "capstone_metrics.json").write_text(json.dumps(metrics, indent=2))

    metrics_markdown = f"""# Capstone run metrics

| Measure | Value |
|---|---:|
| Rows | {len(df):,} |
| Pseudonymized clients | {df['client_id'].nunique()} |
| Train / held-out rows | {len(train):,} / {len(test):,} |
| Train / held-out clients | {train['client_id'].nunique()} / {test['client_id'].nunique()} |
| Held-out decline base rate | {base_rate:.3f} |
| Model ROC-AUC | {model_metrics['roc_auc']:.3f} |
| Model average precision | {model_metrics['average_precision']:.3f} |
| Model Precision@50 | {model_metrics['precision_at_50']:.3f} |
| Baseline Precision@50 | {baseline_metrics['precision_at_50']:.3f} |
| Model Precision@100 | {model_metrics['precision_at_100']:.3f} |
| Baseline Precision@100 | {baseline_metrics['precision_at_100']:.3f} |

**Interpretation boundary.** This run prioritizes pages that show an observed decline in the
current snapshot using prior-window and contextual signals. It is directional decision support,
not a causal estimate, a forecast beyond this snapshot, or a claim about Google's algorithm.
"""
    (OUTPUT_DIR / "capstone_metrics.md").write_text(metrics_markdown)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = ["Base rate", "Baseline P@50", "Model P@50"]
    values = [base_rate, baseline_metrics["precision_at_50"], model_metrics["precision_at_50"]]
    bars = ax.bar(labels, values, color=["#94A3B8", "#F59E0B", "#2563EB"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share declining")
    ax.set_title("Held-out client test: prioritization at 50 pages")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "capstone_precision_at_50.png", dpi=180)
    plt.close(fig)

    top_importance = importance_frame.head(10).sort_values("importance_mean_auc_drop")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_importance["feature"], top_importance["importance_mean_auc_drop"], color="#2563EB")
    ax.set_xlabel("Held-out ROC-AUC decrease after permutation")
    ax.set_title("Permutation importance: top capstone signals")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "capstone_feature_importance.png", dpi=180)
    plt.close(fig)

    print(metrics_markdown)
    print(f"\nWrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
