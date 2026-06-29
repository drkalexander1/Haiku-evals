# Results

**Status: closed.** This was an exploratory side-eval, not an ongoing project — the post
below ships it and moves on. Lukewarm by design: the weekly cadence is the asset here, not
this particular eval. The interesting result turned out to be about evals, not about haiku.

Two runs of the identical 24-scenario crossed design (3 models × 4 strata × 3 subjects × 2
prompt variants strict/natural), against the same scoring code, from two separate live API
calls:

- `results/frontier-v2` — custom `run_eval.py`/`score.py` pipeline
- `logs/frontier` — [Inspect](https://inspect.ai-safety-institute.org.uk/) translation
  (`src/inspect_eval.py`), reported via `python -m src.report logs/frontier`

## The durable finding

Same scenarios, same scoring, two independent generation passes. The claude-haiku-4-5 vs
claude-sonnet-4-6 contrast on `subject_cosine_full` flipped between runs:

| Run | Δ subject_cosine_full | t (paired, n=24) |
|---|---|---|
| custom pipeline (`frontier-v2`) | 0.007 | **0.31** — nowhere near significant |
| Inspect (`logs/frontier`) | −0.044 | **−2.39** — clears the t≈1.96 threshold |

Nothing changed between runs except the models' stochastic outputs. The paired power
analysis on the Inspect run shows why this is expected, not surprising: the
claude-sonnet-4-6 vs gpt-4o-mini gap (Δ=0.049) would need ~42 scenarios for 80% power, and
the claude-haiku-4-5 vs gpt-4o-mini gap (Δ=0.006) would need ~2,068. At n=24, a single run
sits well inside the region where generation noise alone can manufacture a
significant-looking model difference and then erase it on the next run.

**Takeaway: one run isn't an eval.** This is a point about eval methodology, not a claim
about which model writes better haiku — the haiku is just the substrate that happened to
surface it.

## The two-measure design

Every haiku is scored against the subject's embedding two ways:

- **`subject_cosine_full`** — embed the whole haiku as one string, cosine against the
  subject embedding.
- **`subject_cosine_word_sum`** ("additive") — strip stopwords, embed each remaining content
  word individually, sum the vectors, normalize, then cosine against the subject embedding.

The motivation: cosine similarity only tells you two vectors point the same direction, not
that they land in the same place. "Water," "river," "ocean," and "rain" are all high-cosine
to a *water* subject without necessarily being additively similar to it. `word_sum` tests
whether naive additive composition actually lands near the subject; `full` is the control
introduced to pressure-test that. The original expectation was that additive similarity
would track grounding — `full` was added to check it, not the other way around.

Result, `full − word_sum`, across all three models:

| Model | full − word_sum |
|---|---|
| gpt-4o-mini | −0.154 |
| claude-haiku-4-5 | −0.154 |
| claude-sonnet-4-6 | −0.126 |

Negative and near-identical across models: **word-sum consistently lands closer to the
subject than the full-sentence embedding does.** This within-haiku contrast survives n=24
because it's a per-haiku, cross-model-cancelling comparison — unlike the cross-model cosine
differences above, it doesn't need a paired t-test to be a stable finding.

## subject_mentioned_rate — not a quality or contamination signal

| Model | subject_mentioned_rate |
|---|---|
| gpt-4o-mini | 0.08 |
| claude-haiku-4-5 | 0.00 |
| claude-sonnet-4-6 | 0.21 |

This is not a content-quality metric, and a high rate is not contamination. Literal subject
mentions are a legitimate haiku move (mic-drop final word, name-then-circle-back structure).
The only genuinely degenerate pattern this could catch is generic filler that nets ~0
semantic grounding, rescued by dropping the subject word in at the end. A high
`subject_mentioned_rate` flags *which* haikus are worth reading, not a failure mode by
itself. Read literally: claude-haiku-4-5's similarity scores in this run are entirely from
oblique reference (zero literal mentions); claude-sonnet-4-6 reaches its (slightly higher)
`subject_cosine_full` with help from naming the subject directly about 1 in 5 times.

## What this run does NOT support

- **No form/content tradeoff claim.** Any apparent anti-correlation between syllable
  accuracy and subject grounding is inside the noise floor at n=24 — `syllable_perfect_rate`
  confidence intervals fully overlap across models.
- **No "model X is better at haiku" ranking** on any single measure — see the durable
  finding above.
- **No earned-vs-rescued mention quantification.** Telling apart "the subject word does real
  work" from "filler haiku, subject word bolted on" needs more than reading the mention rate;
  n=24 only supports a case-study read-through, not a measured rate.

## What I'd do with more scenarios (parked, not started)

- **Token-ablation as a standing metric**: recompute `subject_cosine_word_sum` with the
  subject token removed from the haiku before embedding. A score that survives removal is
  evidence the mention was earned (the rest of the haiku already carried the grounding); a
  score that collapses is evidence of filler-plus-keyword.
- **Stratify scenarios on mention status**, not just topic — need enough
  mentioned/not-mentioned and earned/rescued cases to actually measure the split above
  instead of reading a handful of examples.
- **Larger n** to move any cross-model contrast (the durable finding's whole point) out of
  the noise floor — the power analysis above gives concrete targets (~42–2,000+ scenarios
  depending on the pair and metric).

## Reproducing

```bash
# custom pipeline
python -m src.run_eval --models gpt-4o-mini,claude-haiku-4-5,claude-sonnet-4-6 --output results/my-run
python -m src.score --run results/my-run

# Inspect translation
inspect eval src/inspect_eval.py --model openai/gpt-4o-mini,anthropic/claude-haiku-4-5,anthropic/claude-sonnet-4-6
python -m src.report logs/ --output results/my-inspect-run
```

## Known limitations (instrument-level, not addressed this round)

- **The syllable counter is an unvalidated heuristic** (`syllables.estimate`, third-party,
  not spot-checked against known-correct counts). A systematic bias would show up identically
  across all models and isn't caught by this design.
- **n=24 is thin** even for the within-haiku word_sum-vs-full contrast's stratum breakdowns.
- **Embedder choice affects absolute cosine values** — all numbers above use the local
  `all-MiniLM-L6-v2` embedder.
