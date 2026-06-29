"""Score haiku predictions from legacy predictions.jsonl runs.

Prefer Inspect for new runs (src/inspect_eval.py) and use src/report.py for
Inspect logs. This module remains for older run_eval output and re-scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.embeddings import EmbeddingConfig, default_embedding_config, subject_mentioned, subject_similarity_scores
from src.reporting import build_summary, write_run_outputs
from src.schema import (
    DEFAULT_EMBEDDER,
    DEFAULT_EMBEDDING_MODEL_LOCAL,
    DEFAULT_EMBEDDING_MODEL_OPENAI,
    Embedder,
    ROOT,
    PredictionRecord,
    Scenario,
    load_scenarios,
)
from src.syllables_util import line_syllable_counts, syllable_l1_error, syllable_perfect

RESULTS_DIR = ROOT / "results" / "latest"


def load_predictions(path: Path) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(PredictionRecord.model_validate_json(line))
    return records


def score_haiku(
    lines: list[str],
    subject: str,
    *,
    embedding: EmbeddingConfig,
) -> dict:
    haiku_text = "\n".join(lines)
    syllables = line_syllable_counts(lines)
    sim = subject_similarity_scores(haiku_text, subject, config=embedding)
    return {
        "syllable_line1": syllables[0] if len(syllables) > 0 else None,
        "syllable_line2": syllables[1] if len(syllables) > 1 else None,
        "syllable_line3": syllables[2] if len(syllables) > 2 else None,
        "syllable_l1_error": syllable_l1_error(lines),
        "syllable_perfect": int(syllable_perfect(lines)),
        "line_count_ok": int(len(lines) == 3),
        "subject_mentioned": int(subject_mentioned(haiku_text, subject)),
        **sim,
    }


def build_frame(
    scenarios: list[Scenario],
    predictions: list[PredictionRecord],
    *,
    embedding: EmbeddingConfig,
) -> pd.DataFrame:
    scenario_map = {s.id: s for s in scenarios}
    rows = []
    for rec in predictions:
        sc = scenario_map.get(rec.scenario_id)
        if not sc:
            continue
        lines = rec.prediction.lines()
        metrics = score_haiku(lines, sc.subject, embedding=embedding)
        rows.append(
            {
                "scenario_id": rec.scenario_id,
                "model": rec.model,
                "stratum": sc.stratum,
                "subject": sc.subject,
                "prompt_variant": sc.prompt_variant,
                "line1": lines[0],
                "line2": lines[1],
                "line3": lines[2],
                "latency_ms": rec.latency_ms,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def score_run(
    run_dir: Path,
    scenarios_path: Path,
    *,
    embedding: EmbeddingConfig,
) -> dict:
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing {pred_path}; run run_eval or inspect eval first")

    scenarios = load_scenarios(scenarios_path)
    snap = run_dir / "scenarios_snapshot.yaml"
    if snap.exists():
        import yaml

        with snap.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        scenarios = [Scenario.model_validate(s) for s in raw]

    predictions = load_predictions(pred_path)
    df = build_frame(scenarios, predictions, embedding=embedding)
    if df.empty:
        raise ValueError("No matching predictions for scenarios")

    summary = build_summary(df, embedding=embedding, source="legacy_predictions_jsonl")
    write_run_outputs(df, run_dir, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Score legacy predictions.jsonl run")
    parser.add_argument("--run", type=Path, default=RESULTS_DIR)
    parser.add_argument("--scenarios", type=Path, default=ROOT / "data" / "scenarios.yaml")
    parser.add_argument(
        "--embedder",
        choices=["local", "openai"],
        default=DEFAULT_EMBEDDER,
        help="Embedding backend: local sentence-transformers (default) or OpenAI API",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Model id for the embedder "
            f"(default local: {DEFAULT_EMBEDDING_MODEL_LOCAL}, "
            f"openai: {DEFAULT_EMBEDDING_MODEL_OPENAI})"
        ),
    )
    args = parser.parse_args(argv)

    embedder: Embedder = args.embedder
    model = args.embedding_model
    if model is None:
        model = DEFAULT_EMBEDDING_MODEL_OPENAI if embedder == "openai" else DEFAULT_EMBEDDING_MODEL_LOCAL
    embedding = default_embedding_config(embedder=embedder, model=model)

    summary = score_run(args.run, args.scenarios, embedding=embedding)
    print(json.dumps(summary["models"], indent=2))
    print(f"Wrote summary to {args.run / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
