"""Shared aggregation, plots, and summary export for haiku eval runs."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.embeddings import EmbeddingConfig

# Paired power analysis constants: alpha = 0.05 two-sided, 80% power.
_Z_ALPHA = 1.96
_Z_BETA = 0.84

EMBEDDING_NOTES = {
    "local": "Local sentence-transformers model; provider-neutral, no API",
    "openai": "OpenAI embeddings API; fixed model for all evaluated LLMs",
}


def compute_metrics(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "syllable_l1_error_mean": float(df["syllable_l1_error"].mean()),
        "syllable_perfect_rate": float(df["syllable_perfect"].mean()),
        "line_count_ok_rate": float(df["line_count_ok"].mean()),
        "subject_cosine_full_mean": float(df["subject_cosine_full"].mean()),
        "subject_cosine_word_sum_mean": float(df["subject_cosine_word_sum"].mean()),
        "subject_mentioned_rate": float(df["subject_mentioned"].mean()),
        "cosine_full_minus_word_sum_mean": float(
            (df["subject_cosine_full"] - df["subject_cosine_word_sum"]).mean()
        ),
    }


def pairwise_power_analysis(df: pd.DataFrame) -> dict:
    """Paired comparison of subject_cosine_full for every model pair in the run."""
    pairs = []
    for a, b in combinations(sorted(df["model"].unique()), 2):
        da = df[df["model"] == a].set_index("scenario_id")["subject_cosine_full"]
        db = df[df["model"] == b].set_index("scenario_id")["subject_cosine_full"]
        diffs = (da - db).dropna()
        n = len(diffs)
        if n < 2:
            continue
        delta = float(diffs.mean())
        sd = float(diffs.std(ddof=1))
        se = sd / np.sqrt(n)
        pairs.append(
            {
                "model_a": a,
                "model_b": b,
                "n_scenarios": n,
                "delta_subject_cosine_full": delta,
                "sd_of_paired_diffs": sd,
                "t_paired": delta / se if se > 0 else float("inf"),
                "scenarios_needed_80pct_power": (
                    float((_Z_ALPHA + _Z_BETA) ** 2 * (sd / delta) ** 2)
                    if delta != 0
                    else None
                ),
                "min_detectable_delta_at_n": float((_Z_ALPHA + _Z_BETA) * se),
            }
        )
    return {
        "method": (
            "Paired t on per-scenario subject_cosine_full differences; "
            "alpha=0.05 two-sided, power=0.80. Treats scenarios as exchangeable, "
            "which overstates power given shared subjects; with multiple pairs, "
            "apply a multiple-comparison correction before claiming significance."
        ),
        "pairs": pairs,
    }


def build_summary(
    df: pd.DataFrame,
    *,
    embedding: EmbeddingConfig | None = None,
    source: str = "unknown",
    eval_logs: list[str] | None = None,
) -> dict:
    summary: dict = {
        "source": source,
        "models": {},
        "by_stratum": {},
        "by_prompt_variant": {},
        "power_analysis": pairwise_power_analysis(df) if df["model"].nunique() > 1 else None,
    }
    if eval_logs:
        summary["eval_logs"] = eval_logs
    if embedding is not None:
        summary["embedder"] = embedding.embedder
        summary["embedding_model"] = embedding.model
        summary["embedding_label"] = embedding.label
        summary["embedding_note"] = EMBEDDING_NOTES[embedding.embedder]

    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        summary["models"][model] = compute_metrics(mdf)
        summary["by_stratum"][model] = {}
        for stratum in sorted(mdf["stratum"].unique()):
            summary["by_stratum"][model][stratum] = compute_metrics(mdf[mdf["stratum"] == stratum])
        summary["by_prompt_variant"][model] = {}
        for variant in sorted(mdf["prompt_variant"].unique()):
            summary["by_prompt_variant"][model][variant] = compute_metrics(
                mdf[mdf["prompt_variant"] == variant]
            )
    return summary


def plot_form_vs_semantics(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        ax.scatter(
            sub["syllable_l1_error"],
            sub["subject_cosine_full"],
            alpha=0.65,
            label=model,
        )
    ax.set_xlabel("Syllable L1 error (lower = better form)")
    ax.set_ylabel("Subject cosine (full haiku embedding)")
    ax.set_title("Form vs semantic alignment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_embedding_methods(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        ax.scatter(
            sub["subject_cosine_word_sum"],
            sub["subject_cosine_full"],
            alpha=0.65,
            label=model,
        )
    lims = [
        min(df["subject_cosine_word_sum"].min(), df["subject_cosine_full"].min()) - 0.02,
        max(df["subject_cosine_word_sum"].max(), df["subject_cosine_full"].max()) + 0.02,
    ]
    ax.plot(lims, lims, "k--", alpha=0.4, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Word-sum cosine")
    ax.set_ylabel("Full-text cosine")
    ax.set_title("Embedding methods compared")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_run_outputs(df: pd.DataFrame, out_dir: Path, summary: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_form_vs_semantics(df, out_dir / "form_vs_semantics.png")
    plot_embedding_methods(df, out_dir / "embedding_methods.png")
    df.to_csv(out_dir / "by_scenario.csv", index=False)

    for grouping, key in [("by_stratum", "stratum"), ("by_prompt_variant", "prompt_variant")]:
        rows = []
        for model, groups in summary[grouping].items():
            for name, metrics in groups.items():
                rows.append({"model": model, key: name, **metrics})
        pd.DataFrame(rows).to_csv(out_dir / f"{grouping}.csv", index=False)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path
