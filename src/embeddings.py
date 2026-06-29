"""Embedding similarity: full-text and word-sum cosine vs subject."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from src.schema import (
    DEFAULT_EMBEDDING_MODEL_LOCAL,
    DEFAULT_EMBEDDING_MODEL_OPENAI,
    DEFAULT_EMBEDDER,
    Embedder,
)

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_STOPWORDS = frozenset(
    """
    a an the and or but in on at to for of with by from as is are was were be been
    being it its this that these those i you he she we they my your his her our their
    """.split()
)

_OPENAI_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@dataclass(frozen=True)
class EmbeddingConfig:
    embedder: Embedder = DEFAULT_EMBEDDER
    model: str = DEFAULT_EMBEDDING_MODEL_LOCAL

    @property
    def label(self) -> str:
        return f"{self.embedder}:{self.model}"


def default_embedding_config(
    *,
    embedder: Embedder = DEFAULT_EMBEDDER,
    model: str | None = None,
) -> EmbeddingConfig:
    if model is None:
        model = (
            DEFAULT_EMBEDDING_MODEL_OPENAI
            if embedder == "openai"
            else DEFAULT_EMBEDDING_MODEL_LOCAL
        )
    return EmbeddingConfig(embedder=embedder, model=model)


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def content_words(text: str) -> list[str]:
    return [w for w in tokenize_words(text) if w not in _STOPWORDS and len(w) > 1]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


class _LocalBackend:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float64)


class _OpenAIBackend:
    def __init__(self, model_name: str) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set (required for --embedder openai scoring)"
            )
        self.model_name = model_name
        self._client = OpenAI(api_key=api_key)
        self._dimension = _OPENAI_EMBEDDING_DIMS.get(model_name)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.encode(["probe"])[0])
        return self._dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float64)
        response = self._client.embeddings.create(input=texts, model=self.model_name)
        ordered = sorted(response.data, key=lambda row: row.index)
        return np.asarray([row.embedding for row in ordered], dtype=np.float64)


@lru_cache(maxsize=8)
def _get_backend(embedder: Embedder, model: str):
    if embedder == "openai":
        return _OpenAIBackend(model)
    return _LocalBackend(model)


def encode_texts(texts: list[str], *, config: EmbeddingConfig) -> np.ndarray:
    return _get_backend(config.embedder, config.model).encode(texts)


def embed_full(text: str, *, config: EmbeddingConfig) -> np.ndarray:
    return encode_texts([text], config=config)[0]


def embed_word_sum(text: str, *, config: EmbeddingConfig) -> np.ndarray:
    words = content_words(text)
    backend = _get_backend(config.embedder, config.model)
    if not words:
        return np.zeros(backend.dimension, dtype=np.float64)
    word_vectors = backend.encode(words)
    return normalize(word_vectors.sum(axis=0))


def subject_similarity_scores(
    haiku_text: str,
    subject: str,
    *,
    config: EmbeddingConfig | None = None,
) -> dict[str, float]:
    """Return full-text and word-sum cosine similarity to the subject phrase."""
    cfg = config or default_embedding_config()
    subject_vec = embed_full(subject, config=cfg)
    full_vec = embed_full(haiku_text, config=cfg)
    sum_vec = embed_word_sum(haiku_text, config=cfg)
    return {
        "subject_cosine_full": cosine_similarity(full_vec, subject_vec),
        "subject_cosine_word_sum": cosine_similarity(sum_vec, subject_vec),
    }


def subject_mentioned(haiku_text: str, subject: str) -> bool:
    haystack = haiku_text.lower()
    subject_tokens = content_words(subject)
    if not subject_tokens:
        return subject.strip().lower() in haystack
    return all(tok in haystack for tok in subject_tokens)
