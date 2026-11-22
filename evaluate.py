"""Evaluate SpaceWeather-CoT test predictions against NOAA's own forecasts.

The model generates forecast-discussion text; NOAA's numeric forecasts live in
the export's ``scalars`` field. This script extracts the same quantities from the
model's generated text with regexes and reports agreement (MAE) plus coverage
(how often each quantity could be parsed at all). It also computes two trivial
baselines for context: persistence of the scalar values is not meaningful for
probabilities, so baselines are (a) always predicting the train-split mean and
(b) NOAA's own numbers treated as the reference itself.

Honest scope note: scalars are NOAA *forecast* values, so this measures agreement
with the human forecasters, not forecast skill vs realized weather. Realized-
outcome scoring needs observed Kp/flare event data (future work).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

KP_RE = re.compile(r"Kp[^\d]{0,40}?(\d+(?:\.\d+)?)", re.IGNORECASE)
PCT_RE = re.compile(r"(\d{1,3})\s*%")


def _extract(text: str) -> dict[str, float | None]:
    out = {"kp_max_3day": None, "r1_r2_radio_blackout_pct": None, "s1_radiation_storm_pct": None}
    m = KP_RE.search(text)
    if m:
        out["kp_max_3day"] = float(m.group(1))
    pcts = [float(p) for p in PCT_RE.findall(text)]
    if pcts:
        # Heuristic: radiation-storm chances are typically the smallest quoted
        # probabilities; blackout chances the largest. Take max/min as proxies.
        out["r1_r2_radio_blackout_pct"] = max(pcts)
        out["s1_radiation_storm_pct"] = min(pcts)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="test_predictions.jsonl from training")
    parser.add_argument("--data", type=Path, default=Path("spacweather_cot.jsonl"))
    args = parser.parse_args()

    # Train-split means as the dumb baseline.
    train_vals = {"kp_max_3day": [], "r1_r2_radio_blackout_pct": [], "s1_radiation_storm_pct": []}
    with args.data.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "train":
                for k, v in row["scalars"].items():
                    if k in train_vals:
                        train_vals[k].append(v)
    baseline = {k: float(np.mean(v)) for k, v in train_vals.items()}

    rows = [json.loads(l) for l in args.predictions.open()]
    print(f"{len(rows)} test predictions; train-mean baseline: {baseline}")
    for key in ("kp_max_3day", "r1_r2_radio_blackout_pct", "s1_radiation_storm_pct"):
        pairs = [(r["scalars"].get(key), _extract(r["prediction"])[key]) for r in rows]
        pairs = [(ref, pred) for ref, pred in pairs if ref is not None]
        n_parsed = sum(1 for _, p in pairs if p is not None)
        errs = [abs(ref - pred) for ref, pred in pairs if pred is not None]
        base_errs = [abs(ref - baseline[key]) for ref, _ in pairs]
        print(
            f"{key}: n={len(pairs)} parsed={n_parsed} "
            f"model_MAE={np.mean(errs):.3f} baseline_MAE={np.mean(base_errs):.3f}"
            if errs
            else f"{key}: no parseable predictions"
        )


if __name__ == "__main__":
    main()
