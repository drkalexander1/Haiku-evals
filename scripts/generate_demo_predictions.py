"""Generate synthetic haiku predictions for results/demo (no API keys)."""

from __future__ import annotations

import json
from pathlib import Path

from src.schema import HaikuPrediction, PredictionRecord, Scenario, load_scenarios

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "demo"

# Hand-crafted 5-7-5 demos; quality varies by stratum for metric spread.
DEMO_HAIKUS: dict[str, tuple[str, str, str]] = {
    "ocean": ("Salt waves crash and roll", "Seagulls cry above the deep blue sea", "Tide marks fade at dusk"),
    "snow": ("Soft white flakes descend", "Blanketing the silent winter woods", "Footprints melt away"),
    "cherry blossom": ("Pink petals drift down", "Spring breeze scatters blossoms lightly", "Paths turn pale rose"),
    "thunderstorm": ("Dark clouds gather fast", "Lightning splits the heavy summer sky", "Rain drums on the roof"),
    "autumn leaves": ("Crimson leaves spiral", "Wind lifts them from the maple trees", "Crisp air fills the lane"),
    "bicycle": ("Wheels spin on pavement", "A rusty chain clicks through morning light", "Helmet hangs on hook"),
    "coffee cup": ("Steam curls from the mug", "Bitter warmth wakes my sleepy mind", "Saucer rings the wood"),
    "window": ("Rain streaks on the glass", "Outside world blurs through condensation", "Finger trails a line"),
    "clock": ("Tick tock marks the hours", "Brass pendulum swings through dim light", "Midnight chimes once"),
    "mirror": ("Face meets glass at dawn", "Reflection blinks with tired grey eyes", "Steam fogs the edge"),
    "grief": ("Empty chair by fire", "Letters folded in a drawer remain", "Rain taps the window"),
    "loneliness": ("One cup on the table", "Streetlight pools on quiet pavement", "Phone screen stays dark"),
    "time": ("Sand drains through glass", "Calendar pages curl at corners", "Sun sets without pause"),
    "memory": ("Old photos yellow", "Voices echo in a hallway dim", "Names fade on the page"),
    "silence": ("No birds at sunrise", "Snow absorbs the distant highway hum", "Breath clouds the air"),
    "morning frost": ("Ice lace on the pane", "Boots crunch through stiff silver grass", "Breath hangs white and still"),
    "city traffic": ("Horns blur at the light", "Exhaust hangs over wet asphalt", "Crosswalk signal blinks"),
    "old bookshelf": ("Dust motes in sunbeams", "Spines cracked on forgotten novels", "Page smell fills the room"),
    "summer rain": ("Warm drops on hot stone", "Petrichor rises from the pavement", "Umbrellas bloom below"),
    "winter night": ("Stars pierce the cold dark", "Frost etches patterns on the gate", "Embers glow indoors"),
}


def demo_haiku(scenario: Scenario, model: str) -> HaikuPrediction:
    lines = DEMO_HAIKUS.get(scenario.subject.lower())
    if lines is None:
        lines = (
            f"Demo line about {scenario.subject}",
            "Middle line with seven syllables here",
            "Short last line ends",
        )
    if "mini" in model:
        # Worse form on purpose for the smaller demo model.
        return HaikuPrediction(line1=lines[0], line2=lines[1] + " today", line3=lines[2][:-3] or lines[2])
    return HaikuPrediction(line1=lines[0], line2=lines[1], line3=lines[2])


def main() -> None:
    scenarios = load_scenarios()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "predictions.jsonl"
    models = ["gpt-4o-mini-demo", "gpt-4o-demo"]

    with path.open("w", encoding="utf-8") as f:
        for model in models:
            for i, sc in enumerate(scenarios):
                pred = demo_haiku(sc, model)
                rec = PredictionRecord(
                    scenario_id=sc.id,
                    model=model,
                    provider="DemoProvider",
                    prediction=pred,
                    latency_ms=100.0 + i,
                )
                f.write(rec.model_dump_json() + "\n")

    manifest = {
        "created_at": "demo",
        "models": models,
        "scenario_count": len(scenarios),
        "predictions_file": "predictions.jsonl",
        "note": "Synthetic data for scoring smoke test only",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(scenarios) * len(models)} lines to {path}")


if __name__ == "__main__":
    main()
