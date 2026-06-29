"""Export Inspect eval logs to CSV summaries, plots, and power analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.embeddings import EmbeddingConfig, default_embedding_config
from src.inspect_metrics import HAIKU_SCORE_KEYS
from src.reporting import build_summary, write_run_outputs
from src.schema import (
    DEFAULT_EMBEDDING_MODEL_LOCAL,
    DEFAULT_EMBEDDING_MODEL_OPENAI,
    ROOT,
)

SCORER_NAME = "haiku_scorer"
RESULTS_DIR = ROOT / "results" / "latest"


def _collect_eval_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {path}")
    logs = sorted(path.glob("**/*.eval"))
    if not logs:
        raise FileNotFoundError(f"No .eval logs found under {path}")
    return logs


def _metrics_from_sample(score) -> dict:
    if isinstance(score.value, dict):
        return dict(score.value)
    if score.metadata:
        return {k: score.metadata[k] for k in HAIKU_SCORE_KEYS if k in score.metadata}
    if isinstance(score.value, int | float):
        return {"subject_cosine_full": float(score.value)}
    return {}


def _lines_from_sample(score, metadata: dict | None) -> tuple[str, str, str]:
    if metadata:
        if all(k in metadata for k in ("line1", "line2", "line3")):
            return metadata["line1"], metadata["line2"], metadata["line3"]
    if score.answer:
        parts = score.answer.split("\n")
        while len(parts) < 3:
            parts.append("")
        return parts[0], parts[1], parts[2]
    return "", "", ""


def frame_from_eval_log(log_path: Path) -> pd.DataFrame:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    model = log.eval.model or log_path.stem
    task_args = (log.eval.task_args_passed or log.eval.task_args or {}) if log.eval else {}

    rows = []
    for sample in log.samples or []:
        if not sample.scores or SCORER_NAME not in sample.scores:
            continue
        score = sample.scores[SCORER_NAME]
        meta = sample.metadata or {}
        metrics = _metrics_from_sample(score)
        line1, line2, line3 = _lines_from_sample(score, score.metadata)
        rows.append(
            {
                "scenario_id": sample.id,
                "model": model,
                "stratum": meta.get("stratum"),
                "subject": meta.get("subject"),
                "prompt_variant": meta.get("prompt_variant"),
                "line1": line1,
                "line2": line2,
                "line3": line3,
                "eval_log": log_path.name,
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and task_args:
        df.attrs["task_args"] = task_args
    return df


def frame_from_eval_paths(paths: list[Path]) -> pd.DataFrame:
    frames = [frame_from_eval_log(p) for p in paths]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError("No scored samples found in eval log(s)")
    return pd.concat(frames, ignore_index=True)


def embedding_from_task_args(task_args: dict) -> EmbeddingConfig | None:
    embedder = task_args.get("embedder")
    if not embedder:
        return None
    return default_embedding_config(
        embedder=embedder,
        model=task_args.get("embedding_model"),
    )


def report_eval_logs(
    log_path: Path,
    output_dir: Path,
    *,
    embedding: EmbeddingConfig | None = None,
) -> dict:
    paths = _collect_eval_paths(log_path)
    df = frame_from_eval_paths(paths)

    if embedding is None and paths:
        from inspect_ai.log import read_eval_log

        first = read_eval_log(str(paths[0]))
        task_args = (first.eval.task_args_passed or first.eval.task_args or {}) if first.eval else {}
        embedding = embedding_from_task_args(task_args)

    summary = build_summary(
        df,
        embedding=embedding,
        source="inspect",
        eval_logs=[p.name for p in paths],
    )
    write_run_outputs(df, output_dir, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Report from Inspect eval log(s)")
    parser.add_argument(
        "log",
        type=Path,
        help="Path to a .eval file or directory containing .eval logs",
    )
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--embedder",
        choices=["local", "openai"],
        default=None,
        help="Override embedder label in summary (scores come from the log)",
    )
    parser.add_argument("--embedding-model", default=None)
    args = parser.parse_args(argv)

    embedding = None
    if args.embedder is not None:
        model = args.embedding_model
        if model is None:
            model = (
                DEFAULT_EMBEDDING_MODEL_OPENAI
                if args.embedder == "openai"
                else DEFAULT_EMBEDDING_MODEL_LOCAL
            )
        embedding = default_embedding_config(embedder=args.embedder, model=model)

    summary = report_eval_logs(args.log, args.output, embedding=embedding)
    print(json.dumps(summary["models"], indent=2))
    print(f"Wrote summary to {args.output / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
