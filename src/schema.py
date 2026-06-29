"""Pydantic models for haiku scenarios and model predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCENARIOS_PATH = DATA_DIR / "scenarios.yaml"
PROMPT_PATHS = {
    "strict": ROOT / "prompts" / "strict_v1.txt",
    "natural": ROOT / "prompts" / "natural_v1.txt",
}

Stratum = Literal["concrete_nature", "concrete_object", "abstract", "compound"]
PromptVariant = Literal["strict", "natural"]
TARGET_SYLLABLES = (5, 7, 5)

Embedder = Literal["local", "openai"]

# Default: local sentence-transformers (no API, same space for all LLM providers).
DEFAULT_EMBEDDER: Embedder = "local"
DEFAULT_EMBEDDING_MODEL_LOCAL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"

# Back-compat alias used when only a local model id is passed.
DEFAULT_EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL_LOCAL


class Scenario(BaseModel):
    id: str
    stratum: Stratum
    subject: str
    prompt_variant: PromptVariant
    notes: str = ""

    @field_validator("subject")
    @classmethod
    def strip_subject(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("subject must be non-empty")
        return v


class HaikuPrediction(BaseModel):
    line1: str = Field(min_length=1, max_length=120)
    line2: str = Field(min_length=1, max_length=120)
    line3: str = Field(min_length=1, max_length=120)

    def lines(self) -> list[str]:
        return [self.line1.strip(), self.line2.strip(), self.line3.strip()]

    def full_text(self) -> str:
        return "\n".join(self.lines())


def _strict_json_schema(schema: dict) -> dict:
    out = dict(schema)
    if out.get("type") == "object":
        out["additionalProperties"] = False
    if "properties" in out:
        out["properties"] = {k: _strict_json_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _strict_json_schema(out["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        if key in out:
            out[key] = [_strict_json_schema(s) for s in out[key]]
    if "$defs" in out:
        out["$defs"] = {k: _strict_json_schema(v) for k, v in out["$defs"].items()}
    return out


def prediction_json_schema() -> dict:
    return _strict_json_schema(HaikuPrediction.model_json_schema())


def parse_prediction(data: dict) -> HaikuPrediction:
    return HaikuPrediction.model_validate(data)


class PredictionRecord(BaseModel):
    scenario_id: str
    model: str
    provider: str
    prediction: HaikuPrediction
    latency_ms: float | None = None
    raw_response: str | None = None


def load_scenarios(path: Path | None = None) -> list[Scenario]:
    path = path or SCENARIOS_PATH
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}")
    return [Scenario.model_validate(item) for item in raw]


def load_prompt_template(variant: PromptVariant, path: Path | None = None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    prompt_path = PROMPT_PATHS[variant]
    return prompt_path.read_text(encoding="utf-8")
