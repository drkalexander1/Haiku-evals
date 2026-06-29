"""Inspect (inspect_ai) task for the haiku eval.

Custom scoring lives in src/score.py (score_haiku). This module wires scenarios,
structured generation, and Inspect metrics around that logic.

Run:
    inspect eval src/inspect_eval.py --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5
    inspect eval src/inspect_eval.py -T embedder=openai -T embedding_model=text-embedding-3-small
    python -m src.report logs/ --output results/my-run
"""

from __future__ import annotations

import json

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ResponseSchema
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState, generate
from inspect_ai.util import json_schema

from src.embeddings import EmbeddingConfig, default_embedding_config
from src.inspect_metrics import (
    HAIKU_SCORE_KEYS,
    INVALID_HAIKU_SCORE,
    haiku_scorer_metrics,
)
from src.schema import (
    DEFAULT_EMBEDDER,
    Embedder,
    HaikuPrediction,
    load_prompt_template,
    load_scenarios,
)
from src.score import score_haiku

SCORER_NAME = "haiku_scorer"

_RESPONSE_SCHEMA = ResponseSchema(
    name="haiku_prediction",
    json_schema=json_schema(HaikuPrediction),
    strict=True,
)


def _render_prompt(subject: str, prompt_variant: str) -> str:
    template = load_prompt_template(prompt_variant)
    return template.format(subject=subject)


def haiku_dataset(scenarios_path=None) -> MemoryDataset:
    scenarios = load_scenarios(scenarios_path)
    samples = [
        Sample(
            input=_render_prompt(s.subject, s.prompt_variant),
            id=s.id,
            metadata={
                "stratum": s.stratum,
                "subject": s.subject,
                "prompt_variant": s.prompt_variant,
            },
        )
        for s in scenarios
    ]
    return MemoryDataset(samples, name="haiku_scenarios")


def _score_value(metrics: dict) -> dict:
    return {key: metrics[key] for key in HAIKU_SCORE_KEYS}


def build_haiku_scorer(embedding: EmbeddingConfig):
    @scorer(name=SCORER_NAME, metrics=haiku_scorer_metrics())
    def haiku_scorer():
        async def score(state: TaskState, target: Target) -> Score:
            try:
                prediction = HaikuPrediction.model_validate_json(state.output.completion)
                lines = prediction.lines()
            except (ValueError, json.JSONDecodeError) as exc:
                return Score(
                    value=INVALID_HAIKU_SCORE,
                    explanation=f"Invalid prediction JSON: {exc}",
                )

            subject = state.metadata["subject"]
            metrics = score_haiku(lines, subject, embedding=embedding)
            return Score(
                value=_score_value(metrics),
                answer="\n".join(lines),
                metadata={
                    "line1": lines[0],
                    "line2": lines[1],
                    "line3": lines[2],
                    "syllable_line1": metrics["syllable_line1"],
                    "syllable_line2": metrics["syllable_line2"],
                    "syllable_line3": metrics["syllable_line3"],
                },
            )

        return score

    return haiku_scorer


@task
def haiku_eval(
    embedder: Embedder = DEFAULT_EMBEDDER,
    embedding_model: str | None = None,
) -> Task:
    embedding = default_embedding_config(embedder=embedder, model=embedding_model)
    return Task(
        dataset=haiku_dataset(),
        solver=generate(),
        scorer=build_haiku_scorer(embedding)(),
        config=GenerateConfig(response_schema=_RESPONSE_SCHEMA),
    )
