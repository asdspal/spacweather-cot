"""Modal app for SpaceWeather-CoT: export check, smoke test, and OpenTSLM fine-tune.

Volumes:
  spacweather-cot-data   — input JSONL export (put spacweather_cot.jsonl here)
  spacweather-cot-runs   — training outputs (best_model.pt, history, predictions)

Usage:
  modal volume create spacweather-cot-data
  modal volume put spacweather-cot-data spacweather_cot.jsonl /spacweather_cot.jsonl
  modal run modal_app.py::smoke        # CPU: validate dataset formatting end-to-end
  modal run modal_app.py::train        # A10G: full fine-tune
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

HERE = Path(__file__).parent
OPENTSLM_DIR = Path("/tmp/OpenTSLM")

app = modal.App("spacweather-cot")

data_vol = modal.Volume.from_name("spacweather-cot-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("spacweather-cot-runs", create_if_missing=True)

# CUDA torch wheels are the default on linux; this image is reused by smoke (cpu ok)
# and train (gpu) functions.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "torch>=2.9.1",
        "transformers>=4.57.3",
        "peft>=0.18.0",
        "datasets>=4.4.1",
        "huggingface-hub>=0.36.0",
        "einops>=0.8.1",
        "open-flamingo>=0.0.2",
        "modelscope>=1.20",
        "numpy>=2.3.5",
        "tqdm>=4.67.1",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "PYTHONPATH": "/opt/OpenTSLM/src:/opt/work",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_dir(str(OPENTSLM_DIR), remote_path="/opt/OpenTSLM", ignore=[".git", "results", "data", "__pycache__"])
    .add_local_dir(str(HERE), remote_path="/opt/work", ignore=["__pycache__", "*.jsonl"])
)


@app.function(image=image, volumes={"/data": data_vol, "/runs": runs_vol}, timeout=1800)
def smoke(max_samples: int = 4) -> str:
    """CPU check: load the dataset, collate one batch, print shapes — no model."""
    import torch
    from torch.utils.data import DataLoader

    from opentslm.model_config import PATCH_SIZE
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
    from spacweather_cot_dataset import SpaceWeatherCoTDataset

    out = []
    for split in ("train", "validation", "test"):
        ds = SpaceWeatherCoTDataset(split, EOS_TOKEN="</s>", jsonl_path="/data/spacweather_cot.jsonl", max_samples=max_samples)
        sample = ds[0]
        n_ts = len(sample["time_series"])
        lens = [len(ts) for ts in sample["time_series"]]
        batch = extend_time_series_to_match_patch_size_and_aggregate([sample], patch_size=PATCH_SIZE)
        out.append(f"{split}: n={len(ds)} ts_prompts={n_ts} lens={lens} batch_keys={sorted(batch[0].keys())}")
    return "\n".join(out)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": data_vol, "/runs": runs_vol, "/cache": modal.Volume.from_name("hf-cache", create_if_missing=True)},
    timeout=60 * 60 * 8,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def train(epochs: int = 20, batch_size: int = 4, max_samples: int = 0) -> str:
    """Fine-tune OpenTSLM-SP on SpaceWeather-CoT. Writes to /runs/<ts>/."""
    import subprocess
    import time

    out_dir = f"/runs/spacweather-cot-{int(time.time())}"
    cmd = [
        "python", "/opt/work/train_spacweather.py",
        "--data", "/data/spacweather_cot.jsonl",
        "--out", out_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--device", "cuda",
        "--llm-source", "hf",
    ]
    if max_samples:
        cmd += ["--max-samples", str(max_samples)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout[-8000:])
    print(proc.stderr[-4000:])
    runs_vol.commit()
    return f"exit={proc.returncode} out={out_dir}"


@app.local_entrypoint()
def main():
    print(smoke.remote())
