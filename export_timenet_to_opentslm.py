"""Export the built noaa/spaceweather-cot TimeF dataset into OpenTSLM training rows.

Reads the dataset through the TimeNet SDK (``TimeNet().open(...).read()``), so the
export exercises the same TimeF artifacts a consumer would see, and writes one JSON
object per issuance:

    {
      "issued_at": "2024-03-01T00:30:00+00:00",
      "split": "train" | "val" | "test",
      "prompt": "...",                # AnswerTask prompt
      "rationale": "...",             # forecaster's 24 hr Summary text
      "forecast": "...",              # forecaster's Forecast text (training target)
      "scalars": {"kp_max_3day": 4.0, "r1_r2_radio_blackout_pct": 30.0,
                  "s1_radiation_storm_pct": 1.0},
      "series": {"xrsb_flux": [...], "bz_gsm": [...], "speed": [...],
                 "density": [...], "kp": [...]}
    }

Series values keep ``None`` for missing timesteps; the OpenTSLM dataset class decides
how to handle them at batch time.

Temporal split (no random leakage): train < 2024-12-01, val = December 2024,
test >= 2025-01-01 (fully unseen future months, including Jan-Apr 2025).
September 2024 has no archived discussions (a real NOAA gap).

Usage:
    uv run --project /tmp/TimeNet python export_timenet_to_opentslm.py \
        --dataset noaa/spaceweather-cot --out spacweather_cot.jsonl
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np

from timenet import TimeNet
from timenet.types import AnswerTask, ScalarPredictionTask, read_value

# Temporal split boundaries on the issuance timestamp (UTC).
VAL_START = date(2024, 12, 1)
TEST_START = date(2025, 1, 1)


def _split_for(issued_at: datetime) -> str:
    day = issued_at.date()
    if day >= TEST_START:
        return "test"
    if day >= VAL_START:
        return "val"
    return "train"


def _series_values(signal) -> list[float | None]:
    """Return a signal's values as floats, preserving missing timesteps as None."""
    values = signal.to_arrow().to_pylist()
    flat = [item for sub in values for item in (sub if isinstance(sub, list) else [sub])]
    return [None if v is None else float(v) for v in flat]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="noaa/spaceweather-cot")
    parser.add_argument("--version", default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    dataset = TimeNet().open(args.dataset, args.version).read()
    rows = []
    for record in dataset.records:
        issued_at_raw = next(
            (read_value(a.value) for a in record.annotations if a.key == "issued_at"), None
        )
        if not isinstance(issued_at_raw, str):
            raise ValueError("record is missing its issued_at annotation")
        issued_at = datetime.fromisoformat(issued_at_raw)

        series = {signal.name: _series_values(signal) for signal in record.signals}

        tasks = dataset.tasks_for(record)
        answers = [t for t in tasks if isinstance(t, AnswerTask)]
        scalars = [t for t in tasks if isinstance(t, ScalarPredictionTask)]
        if len(answers) != 1:
            raise ValueError(f"expected 1 AnswerTask for {issued_at}, found {len(answers)}")
        answer = answers[0]
        forecast_targets = answer.targets
        forecast_text = read_value(forecast_targets[0]) if forecast_targets else None
        if not isinstance(forecast_text, str):
            raise ValueError(f"AnswerTask targets are not text for {issued_at}")
        prompt_text = read_value(answer.prompt)
        rationale_text = read_value(answer.rationale)

        scalar_map = {}
        for task in scalars:
            if task.targets is None or task.target_name is None:
                continue
            scalar_map[task.target_name] = float(read_value(task.targets[0]))

        rows.append(
            {
                "issued_at": issued_at.isoformat(),
                "split": _split_for(issued_at),
                "prompt": prompt_text,
                "rationale": rationale_text,
                "forecast": forecast_text,
                "scalars": scalar_map,
                "series": series,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        counts[row["split"]] += 1
    print(f"wrote {len(rows)} rows to {args.out}: {counts}")


if __name__ == "__main__":
    main()
