# SpaceWeather-CoT

A TimeNet Online Hackathon entry: forecast-discussion text + numeric probabilities
from aligned solar/geomagnetic time series, built around a new TimeNet dataset and
two complementary models — one that writes, one that calculates.

## What this is

**Problem.** Space weather threatens power grids, GPS, satellites and aviation.
NOAA's Space Weather Prediction Center issues a human-written forecast twice daily,
plus numeric probabilities (expected max Kp, radio-blackout chance, radiation-storm
chance). We pair that expert text with the actual instrument data the forecaster
read, so a model can learn the real forecasting process — not a GPT-generated
caption of it.

**Dataset: `noaa/spaceweather-cot`** ([TimeNet PR #842](https://github.com/OpenTSLM/TimeNet/pull/842))
- 656 records, March 2024 – April 2025 (the full span the Forecast Discussion archive
  and science-quality GOES-16 data both cover at build time).
- Each record: 24h of GOES-16 X-ray flux (1-min) + OMNI2 solar-wind/Kp (hourly),
  paired with NOAA's own 24h summary, 3-day forecast text, and 3-day numeric
  probabilities.
- TimeNet's first space-weather dataset, and its first with genuinely expert-written
  (not LLM-distilled) chain-of-thought rationale.

**Two models, one for each half of the forecaster's job:**

1. **A fine-tuned OpenTSLM (Llama-3.2-1B, LoRA, soft-prompt)** writes the narrative:
   given the raw series, it produces a structurally correct, NOAA-formatted Forecast
   Discussion (Solar Activity / Energetic Particle / Solar Wind / Geospace, each with
   a summary and a forecast). It captures the forecaster's *voice and structure*
   faithfully; it does **not** reliably ground specific facts (flare classes, numeric
   probabilities) in the input — a known, measured limitation of LLM numeric
   generation, not hidden in this submission.

2. **A gradient-boosted decision model** (`decision_model.py`) predicts NOAA's three
   numeric quantities directly from physics features (flare-class level, flare
   counts, Bz, solar-wind speed/shock, Kp) **plus persistence** (the previous 1-2
   issuances' own NOAA forecasts, which turn out to dominate: 73-89% of feature
   importance). This mirrors NOAA's own real technique, Model Output Statistics: a
   statistical layer that calibrates numeric guidance, separate from the written
   discussion.

## Results (held out: Jan-Apr 2025, never seen in training)

| Target | Decision-model test R² | Fine-tuned LLM (free-text numbers) |
|---|---|---|
| Kp max (3-day) | **0.537** | worse than baseline |
| R1-R2 radio blackout % | **0.842** | worse than baseline |
| S1 radiation storm % | 0.436 | ~tied with baseline |

All three decision-model R² values beat even the best-possible constant baseline
(predicting the test period's own true mean — information not available in real
deployment). The LLM, when asked to free-generate the same numbers as text, loses to
a trivial mean baseline on 2 of 3 — the textbook failure mode of numeric regression
via next-token prediction. Full comparison and the math behind it: `evaluate.py`,
`decision_model.py`.

**Known gap, investigated and documented, not hidden:** S1's defining physical driver
(>10 MeV proton flux) is present in OMNI2's schema but **100% missing/fill in the
2024-2025 files** (verified directly against the raw archive — an upstream data gap,
not a connector defect). GOES's own SEISS proton sensor (`sgps-l2-avg5m`) would supply
it; ingesting it is scoped as the clear next step, not attempted here due to its
per-day (not single pinned annual) file layout.

## Files

- `export_timenet_to_opentslm.py` — TimeF dataset → OpenTSLM training JSONL,
  with a temporal split (train < Dec 2024, val = Dec 2024, test = Jan-Apr 2025).
- `spacweather_cot_dataset.py` — OpenTSLM `QADataset`: 5 channels (log/z-scored),
  NOAA's own text as rationale + forecast + numeric guidance line.
- `train_spacweather.py` — LoRA fine-tune of `OpenTSLM/llama-3.2-1b-ecg-sp`.
- `decision_model.py` — the statistical decision model and its evaluation.
- `evaluate.py` — regex-based scoring of the LLM's free-text numeric claims.
- `modal_app.py` — Modal volumes/functions (`smoke`, `train`) used for all runs.

## Reproduce

```bash
# 1. Build the dataset (in the OpenTSLM/TimeNet repo)
timenet-build build noaa/spaceweather-cot --no-isolation

# 2. Export to OpenTSLM's training format
uv run --project /tmp/TimeNet python export_timenet_to_opentslm.py \
    --dataset noaa/spaceweather-cot --out spacweather_cot.jsonl

# 3. Decision model (numeric forecast, seconds on CPU)
python3 decision_model.py --data spacweather_cot.jsonl

# 4. Fine-tune OpenTSLM (Modal, A10G, ~25-40 min)
modal volume put --force spacweather-cot-data spacweather_cot.jsonl /spacweather_cot.jsonl
modal run modal_app.py::train --epochs 8 --batch-size 4
```

## Links

- Dataset PR: https://github.com/OpenTSLM/TimeNet/pull/842
- Project repo: https://github.com/asdspal/spacweather-cot
- NOAA SWPC Forecast Discussion archive: https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/discussion/
- NASA OMNI2: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/
