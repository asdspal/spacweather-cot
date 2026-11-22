"""SpaceWeather-CoT dataset for OpenTSLM fine-tuning.

Loads the JSONL export produced by ``export_timenet_to_opentslm.py`` and formats
each issuance as an OpenTSLM ``PromptWithAnswer`` sample: five time-series prompts
(GOES X-ray flux plus OMNI solar-wind and Kp channels) embedded in an expert
space-weather forecaster prompt, with the forecaster's own summary+forecast text
as the answer.

Preprocessing choices (training-time only, documented here):
- GOES XRS-B flux is log10-transformed (standard for X-ray flux; flare dynamic
  range spans orders of magnitude).
- Every channel is z-score normalized per record and its mean/std are stated in
  the channel label, matching OpenTSLM's ECG-QA convention.
- Missing timesteps (``None`` in the export) are linearly interpolated before
  normalization; a channel with fewer than 2 valid points fails the sample.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from datasets import Dataset

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset

# Channel metadata: export name -> (label, cadence description, transform, decimation).
# xrsb_flux is decimated 5x (1-min -> 5-min) to keep the LLM sequence length tractable;
# flare-scale structure is preserved at this cadence.
_CHANNELS = {
    "xrsb_flux": ("GOES-16 X-ray flux 0.1-0.8 nm, log10(W/m^2), 5-minute cadence", "log10", 5),
    "bz_gsm": ("OMNI IMF Bz (GSM), hourly cadence", "zscore", 1),
    "speed": ("OMNI solar wind speed, km/s, hourly cadence", "zscore", 1),
    "density": ("OMNI proton density, cm^-3, hourly cadence", "zscore", 1),
    "kp": ("OMNI planetary Kp index, hourly cadence", "zscore", 1),
}

_PRE_PROMPT = """You are an expert space weather forecaster at NOAA's Space Weather Prediction Center. You are given the past 24 hours of solar and geomagnetic observations from space-based instruments: GOES X-ray flux (solar flares), the interplanetary magnetic field's Bz component, solar wind speed and proton density, and the planetary Kp geomagnetic index. Each channel is z-score normalized; its label states the raw mean and standard deviation so you can recover approximate physical values.

Your task is to examine these time series and write the SWPC Forecast Discussion: first summarize what the instruments showed over the past 24 hours (flares, particle events, solar wind structures, geomagnetic activity), then forecast solar and geomagnetic activity for the next 3 days in the forecaster's own structured style, covering Solar Activity, Energetic Particles, Solar Wind, and Geospace.

Question: {question}"""

_POST_PROMPT = "Write the forecast discussion now, starting with the 24-hour summary."


def _clean_series(values: list[float | None], transform: str) -> np.ndarray:
    """Interpolate missing values, transform, and z-score one channel."""
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype="float64")
    valid = np.isfinite(arr)
    if valid.sum() < 2:
        raise ValueError("channel has fewer than 2 valid samples")
    if not valid.all():
        arr = np.interp(np.arange(arr.size), np.flatnonzero(valid), arr[valid])
    if transform == "log10":
        arr = np.log10(np.clip(arr, 1e-12, None))
    mean, std = float(arr.mean()), float(arr.std())
    if std > 1e-8:
        arr = (arr - mean) / std
    else:
        arr = arr - mean
    return arr.astype("float32"), mean, std


def _decimate(arr: np.ndarray, factor: int) -> np.ndarray:
    return arr[::factor] if factor > 1 else arr


class SpaceWeatherCoTDataset(QADataset):
    """SpaceWeather-CoT: NOAA forecaster text + aligned GOES/OMNI/Kp series.

    Args:
        split: "train", "validation", or "test" (mapped from the export's
            train/val/test split names).
        EOS_TOKEN: End-of-sequence token.
        jsonl_path: Path to the export JSONL. Defaults to ``spacweather_cot.jsonl``
            next to this file.
        format_sample_str: Forwarded to :class:`QADataset`.
        time_series_format_function: Forwarded to :class:`QADataset`.
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        jsonl_path: str | Path | None = None,
        format_sample_str: bool = False,
        time_series_format_function=None,
        max_samples: int | None = None,
    ):
        self.jsonl_path = Path(jsonl_path) if jsonl_path else Path(__file__).with_name("spacweather_cot.jsonl")
        self.max_samples = max_samples
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self):
        rows = {"train": [], "validation": [], "test": []}
        with self.jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = "validation" if row["split"] == "val" else row["split"]
                rows[key].append(row)
        datasets = []
        for name in ("train", "validation", "test"):
            data = rows[name]
            if self.max_samples and len(data) > self.max_samples:
                data = data[: self.max_samples]
            datasets.append(Dataset.from_list(data))
        return tuple(datasets)

    def _get_answer(self, row) -> str:
        """The forecaster's summary+forecast, plus NOAA's numeric 3-day guidance.

        The discussion prose is qualitative; the probabilities live in the paired
        3-Day Forecast product. Appending them as text teaches the model the
        numeric forecast style and makes its numbers parseable for eval.
        """
        parts = [p for p in (row.get("rationale"), row.get("forecast")) if p]
        scalars = row.get("scalars") or {}
        guidance = []
        if "kp_max_3day" in scalars:
            guidance.append(f"greatest expected 3 hr Kp {scalars['kp_max_3day']:.2f}")
        if "r1_r2_radio_blackout_pct" in scalars:
            guidance.append(f"R1-R2 radio blackout chance {scalars['r1_r2_radio_blackout_pct']:.0f}%")
        if "s1_radiation_storm_pct" in scalars:
            guidance.append(f"S1 radiation storm chance {scalars['s1_radiation_storm_pct']:.0f}%")
        if guidance:
            parts.append("NOAA 3-day forecast guidance: " + "; ".join(guidance) + ".")
        return "\n\n".join(parts)

    def _get_pre_prompt(self, row) -> str:
        return _PRE_PROMPT.format(question=row.get("prompt") or "")

    def _get_post_prompt(self, row) -> str:
        return _POST_PROMPT

    def _get_text_time_series_prompt_list(self, row):
        prompts = []
        for name, (label, transform, decimate) in _CHANNELS.items():
            values = row["series"].get(name)
            if not values:
                raise ValueError(f"sample missing channel {name}")
            arr, mean, std = _clean_series(values, transform)
            arr = _decimate(arr, decimate)
            prompts.append(
                TextTimeSeriesPrompt(
                    f"{label} (raw mean {mean:.4g}, raw std {std:.4g}):",
                    arr.tolist(),
                )
            )
        return prompts
