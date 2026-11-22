"""Fine-tune OpenTSLM (soft-prompt variant) on SpaceWeather-CoT.

Starts from the public OpenTSLM stage-5 ECG-CoT checkpoint
(``OpenTSLM/llama-3.2-1b-ecg-sp`` — the curriculum's last reasoning stage, so the
model already knows how to write structured rationales over multiple aligned
signals), enables LoRA exactly as curriculum stages 3+ do, and trains on the
SpaceWeather-CoT export for a fixed number of epochs with validation-loss model
selection.

Run inside the OpenTSLM repo environment (``uv sync`` there) or the Modal image:

    python train_spacweather.py --data spacweather_cot.jsonl --out runs/spacweather-cot \
        --epochs 20 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

from spacweather_cot_dataset import SpaceWeatherCoTDataset

BASE_REPO = "OpenTSLM/llama-3.2-1b-ecg-sp"
LLM_ID = "meta-llama/Llama-3.2-1B"

LR_ENCODER = 2e-4
LR_PROJECTOR = 1e-4  # also used for LoRA params, matching curriculum_learning.py
WEIGHT_DECAY = 0.01


def _collate(batch):
    return extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=PATCH_SIZE)


def _optimizer(model, lr_scale: float = 1.0):
    from torch.optim import AdamW

    groups = [
        {"params": list(model.encoder.parameters()), "lr": LR_ENCODER * lr_scale, "weight_decay": WEIGHT_DECAY},
        {
            "params": list(model.projector.projector.parameters()),
            "lr": LR_PROJECTOR * lr_scale,
            "weight_decay": WEIGHT_DECAY,
        },
    ]
    lora_params = model.get_lora_parameters()
    if lora_params:
        groups.append({"params": lora_params, "lr": LR_PROJECTOR * lr_scale, "weight_decay": WEIGHT_DECAY})
    return AdamW(groups)


def _evaluate(model, loader) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            total += model.compute_loss(batch).item()
            count += 1
    model.train()
    return total / max(count, 1)


def _load_test_rows(data_path: Path) -> list[dict]:
    rows = []
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == "test":
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Export JSONL path")
    parser.add_argument("--out", required=True, type=Path, help="Run output directory")
    parser.add_argument("--base-repo", default=BASE_REPO)
    parser.add_argument("--llm-id", default=LLM_ID)
    parser.add_argument(
        "--llm-source",
        choices=["hf", "modelscope"],
        default="hf",
        help="'hf' uses the gated meta-llama repo (needs license); 'modelscope' downloads "
        "the identical Llama-3.2-1B weights from ModelScope's LLM-Research mirror.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per split (smoke tests)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    llm_id = args.llm_id
    if args.llm_source == "modelscope":
        import subprocess

        llm_dir = Path("/cache/huggingface/llama-3.2-1b-modelscope")
        if not (llm_dir / "config.json").exists():
            llm_dir.mkdir(parents=True, exist_ok=True)
            print("downloading Llama-3.2-1B from ModelScope (LLM-Research mirror)...")
            subprocess.run(
                [
                    "modelscope", "download", "--model", "LLM-Research/Llama-3.2-1B",
                    "--local_dir", str(llm_dir),
                ],
                check=True,
            )
        llm_id = str(llm_dir)
        print(f"using local LLM weights at {llm_id}")

    print(f"device={args.device} base={args.base_repo}")
    model = OpenTSLMSP(llm_id=llm_id, device=args.device)
    # Enable LoRA before loading so the stage-5 LoRA adapters come along too.
    model.enable_lora(lora_r=16, lora_alpha=32, lora_dropout=0.0)
    ckpt_path = hf_hub_download(args.base_repo, filename="model_checkpoint.pt")
    model.load_from_file(ckpt_path)
    if args.device == "cuda":
        model.llm.gradient_checkpointing_enable()
        model.llm.config.use_cache = False
        model.llm.enable_input_require_grads()
    model.train()

    eos = model.get_eos_token()
    train_ds = SpaceWeatherCoTDataset("train", EOS_TOKEN=eos, jsonl_path=args.data, max_samples=args.max_samples)
    val_ds = SpaceWeatherCoTDataset("validation", EOS_TOKEN=eos, jsonl_path=args.data, max_samples=args.max_samples)
    test_ds = SpaceWeatherCoTDataset("test", EOS_TOKEN=eos, jsonl_path=args.data, max_samples=args.max_samples)
    print(f"splits: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=_collate, num_workers=0)

    optimizer = _optimizer(model)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(10, total_steps // 20)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val, best_path = float("inf"), args.out / "best_model.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        train_loss = running / max(steps_per_epoch, 1)
        val_loss = _evaluate(model, val_loader)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} ({time.time()-t0:.0f}s)")
        (args.out / "history.json").write_text(json.dumps(history, indent=2))
        if val_loss < best_val:
            best_val = val_loss
            model.store_to_file(str(best_path))
            print(f"  saved best (val={val_loss:.4f})")

    # Generate on the held-out test split for qualitative eval + demo.
    print("generating test predictions...")
    model.load_from_file(str(best_path))
    model.eval()
    test_rows = _load_test_rows(args.data)
    if args.max_samples:
        test_rows = test_rows[: args.max_samples]
    preds_path = args.out / "test_predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as handle:
        for i, sample in enumerate(test_ds):
            batch = _collate([sample])
            pred = model.generate(batch, max_new_tokens=600)[0]
            row = test_rows[i] if i < len(test_rows) else {}
            handle.write(
                json.dumps(
                    {
                        "issued_at": row.get("issued_at"),
                        "question": row.get("prompt"),
                        "reference": (
                            "\n\n".join(p for p in (row.get("rationale"), row.get("forecast")) if p) or None
                        ),
                        "scalars": row.get("scalars"),
                        "prediction": pred,
                    }
                )
                + "\n"
            )
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(test_ds)}")
    print(f"done. best val={best_val:.4f}; outputs in {args.out}")


if __name__ == "__main__":
    main()
