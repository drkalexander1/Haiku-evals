"""Validate haiku scenario dataset structure."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from src.schema import SCENARIOS_PATH, Scenario, load_scenarios

STRATUM_COUNTS = {
    "concrete_nature": 6,
    "concrete_object": 6,
    "abstract": 6,
    "compound": 6,
}
PROMPT_VARIANTS = {"strict", "natural"}


def validate_scenarios(scenarios: list[Scenario]) -> list[str]:
    errors: list[str] = []
    if len(scenarios) != 24:
        errors.append(f"scenarios: expected 24 rows, got {len(scenarios)}")

    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        errors.append("scenarios: duplicate id values")

    by_stratum = Counter(s.stratum for s in scenarios)
    for stratum, expected in STRATUM_COUNTS.items():
        if by_stratum[stratum] != expected:
            errors.append(
                f"scenarios: stratum '{stratum}' expected {expected} rows, got {by_stratum[stratum]}"
            )

    by_variant = Counter(s.prompt_variant for s in scenarios)
    for variant in PROMPT_VARIANTS:
        if by_variant[variant] != 12:
            errors.append(f"scenarios: prompt_variant '{variant}' expected 12 rows, got {by_variant[variant]}")

    # Design is crossed: every subject must appear under both variants exactly once.
    by_subject_variants: dict[str, set[str]] = {}
    for s in scenarios:
        by_subject_variants.setdefault(s.subject.lower(), set()).add(s.prompt_variant)
    for subject, variants in by_subject_variants.items():
        if variants != PROMPT_VARIANTS:
            errors.append(f"scenarios: subject '{subject}' missing variants {PROMPT_VARIANTS - variants}")

    subject_variant_pairs = [(s.subject.lower(), s.prompt_variant) for s in scenarios]
    if len(subject_variant_pairs) != len(set(subject_variant_pairs)):
        errors.append("scenarios: duplicate (subject, prompt_variant) pair")

    for s in scenarios:
        if s.prompt_variant not in PROMPT_VARIANTS:
            errors.append(f"scenarios[{s.id}]: unknown prompt_variant '{s.prompt_variant}'")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate haiku eval dataset")
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_PATH)
    args = parser.parse_args(argv)

    errors: list[str] = []
    try:
        scenarios = load_scenarios(args.scenarios)
        errors.extend(validate_scenarios(scenarios))
    except Exception as exc:
        errors.append(f"scenarios: {exc}")

    if errors:
        print("Validation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    strata = Counter(s.stratum for s in scenarios)
    variants = Counter(s.prompt_variant for s in scenarios)
    print(
        f"Validation OK: {len(scenarios)} scenarios, "
        f"strata {dict(strata)}, variants {dict(variants)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
