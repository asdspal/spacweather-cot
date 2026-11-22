# SpaceWeather-CoT

A Temporal AI Model that writes NOAA-style space-weather forecast discussions
from raw solar/geomagnetic observations — with rationales written by human
forecasters, not LLMs.

TimeNet Online Hackathon entry (Round 1).

## What it is

- **Dataset** `noaa/spaceweather-cot` (TimeNet PR: OpenTSLM/TimeNet#842): 465
  records, Mar–Dec 2024. Each record = one SWPC Forecast Discussion issuance
  (0030/1230 UTC) + 24h of GOES-16 X-ray flux and OMNI2 solar-wind/Kp data.
  Expert-written rationale (forecaster's "24 hr Summary") + forecast target +
  NOAA's own numeric forecast probabilities as scalar tasks.
- **Model**: OpenTSLM soft-prompt (Llama-3.2-1B) fine-tuned with LoRA from the
  stage-5 ECG-CoT checkpoint, trained on Modal (A10G).
- **Eval**: held-out Oct–Dec 2024 issuances vs NOAA's own forecasts.

## Files

- `export_timenet_to_opentslm.py` — TimeF dataset → training JSONL
- `spacweather_cot_dataset.py` — OpenTSLM QADataset (5 channels, log/z-score)
- `train_spacweather.py` — LoRA fine-tune (HF or ModelScope LLM weights)
- `evaluate.py` — agreement vs NOAA scalar forecasts
- `modal_app.py` — Modal volumes/functions for smoke/train

## Reproduce

```bash
# build dataset (in OpenTSLM/TimeNet repo)
timenet-build build noaa/spaceweather-cot --no-isolation
uv run --project /tmp/TimeNet python export_timenet_to_opentslm.py \
    --dataset noaa/spaceweather-cot --out spacweather_cot.jsonl
modal volume put spacweather-cot-data spacweather_cot.jsonl /spacweather_cot.jsonl
modal run modal_app.py::train --epochs 20 --batch-size 4
```
