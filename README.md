# Haiku Evals

LLM benchmark for English haiku: **syllable form** (5-7-5) and **subject grounding** via embedding similarity (full-text and word-sum cosine).

Follow-on to [Michigan bird evals](../michigan-bird-evals) and [Florida weather evals](../Florida-weather-evals).

## Why this design

Models write a three-line haiku for a curated **subject** drawn from stratified buckets (concrete nature, objects, abstract, compound phrases). Two **prompt variants** test whether explicit form instructions change outcomes:

| Stratum | What it tests |
|---------|----------------|
| `concrete_nature` | Easy semantic anchor (ocean, snow, …) |
| `concrete_object` | Everyday specificity |
| `abstract` | Topical alignment without literal imagery |
| `compound` | Multi-word subject grounding |

**Scoring** uses your custom Python logic (`score_haiku`) for syllables and embedding cosines. [Inspect](https://inspect.aisi.org.uk/) is the **run harness**; reporting is exported via `src/report.py`.

| `--embedder` (task arg) | Default model | Notes |
|-------------------------|---------------|-------|
| `local` (default) | `all-MiniLM-L6-v2` | sentence-transformers, provider-neutral |
| `openai` | `text-embedding-3-small` | OpenAI embeddings API |

## Metrics

| Metric | What it measures |
|--------|------------------|
| **syllable_l1_error** | Primary (form) — Σ \|actual − target\| per line; target 5-7-5 |
| **syllable_perfect_rate** | Share exactly 5-7-5 |
| **subject_cosine_full** | Primary (semantics) — cosine(full haiku embed, subject embed) |
| **subject_cosine_word_sum** | Bag-of-words hypothesis — cosine(normalize(Σ content-word embeds), subject) |
| **subject_mentioned** | Lexical sanity check — subject tokens appear in text |

## Quick start

Use a **project venv** if you also have bird/weather evals installed (all three use a top-level `src` package):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
python -m src.validate_scenarios
```

### Demo (no API)

```bash
python scripts/generate_demo_predictions.py
python -m src.score --run results/demo
```

### Inspect eval (recommended)

```bash
cp .env.example .env
# Add API keys when running real models.

# One or more models (Inspect handles provider routing)
inspect eval src/inspect_eval.py \
  --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5,anthropic/claude-sonnet-4-6 \
  --log-dir logs/frontier

# OpenAI embeddings for scoring (task args)
inspect eval src/inspect_eval.py \
  --model anthropic/claude-sonnet-4-6 \
  -T embedder=openai \
  -T embedding_model=text-embedding-3-small

# Re-score an existing log with a different embedder (no new LLM calls)
inspect score logs/frontier/your-run.eval \
  -T embedder=openai \
  -T embedding_model=text-embedding-3-small

# Export CSVs, plots, summary.json, and power analysis
python -m src.report logs/frontier/ --output results/frontier
```

See [RESULTS.md](RESULTS.md) for a worked example from a real 3-model run.

### Legacy pipeline

The original `run_eval` → `predictions.jsonl` → `score` path still works for older runs:

```bash
python -m src.run_eval --models gpt-4o-mini,claude-haiku-4-5 --output results/my-run
python -m src.score --run results/my-run
```

## Outputs

**Inspect:** `.eval` logs under `logs/` (use `inspect view` or `python -m src.report`).

**Report export** (`results/<run>/`): `summary.json`, `by_scenario.csv`, `by_stratum.csv`, `by_prompt_variant.csv`, `form_vs_semantics.png`, `embedding_methods.png`.

## License

MIT
