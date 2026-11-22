"""Decision model: a small statistical regressor for NOAA's 3-day numeric forecast.

This is the "decision model" discussed as an alternative to asking an LLM to generate
calibrated numbers via next-token prediction. LLMs are known to be poor at numeric
regression through free-text generation (BPE splits digits inconsistently, and
cross-entropy loss has no notion of numeric distance). Instead, this extracts the same
physical features a human forecaster would look at from the raw input series, plus
persistence (lag-1) features from NOAA's own previous issuance, and fits a classical
regressor directly to NOAA's own scalar targets.

This mirrors NOAA's own real technique: Model Output Statistics (MOS), a statistical
regression layer that corrects/calibrates raw model guidance, separate from the human-
written narrative discussion.

Investigated and rejected: OMNI2's "Proton flux >10 MeV" field (word 46), which is
exactly what NOAA's S1 radiation-storm scale is defined on, is 100% fill/missing in
the 2024-2025 low_res_omni files (upstream data gap, not a connector defect). GOES's
own SEISS proton-flux product (``sgps-l2-avg5m``) would supply it, but is published as
per-day files rather than one pinned annual archive, so ingesting it is future work.

Usage:
    python decision_model.py --data spacweather_cot_v3.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _flare_class_ordinal(max_flux: float | None) -> float:
    """Map a GOES XRS-B flux value to an ordinal flare-class score (A=0 .. X=4)."""
    if not max_flux or max_flux <= 0:
        return 0.0
    log_flux = math.log10(max_flux)
    if log_flux >= -4:
        return 4.0  # X
    if log_flux >= -5:
        return 3.0  # M
    if log_flux >= -6:
        return 2.0  # C
    if log_flux >= -7:
        return 1.0  # B
    return 0.0  # A


def _physical_features(row: dict) -> list[float]:
    """Extract forecaster-relevant physical features from one record's input series."""
    series = row["series"]
    xray = [v for v in series["xrsb_flux"] if v is not None]
    bz = [v for v in series["bz_gsm"] if v is not None]
    speed = [v for v in series["speed"] if v is not None]
    density = [v for v in series["density"] if v is not None]
    kp = [v for v in series["kp"] if v is not None]

    max_flux = max(xray) if xray else 0.0
    mean_flux = float(np.mean(xray)) if xray else 0.0
    # Flare counts by NOAA class threshold (W/m^2), a proxy for flare activity level
    # and, indirectly, for the chance of an associated energetic-particle event.
    count_c = sum(1 for v in xray if v >= 1e-6)
    count_m = sum(1 for v in xray if v >= 1e-5)
    count_x = sum(1 for v in xray if v >= 1e-4)
    # Largest minute-to-minute jump in solar-wind speed: a crude shock/CME-arrival proxy.
    speed_jump = float(np.max(np.diff(speed))) if len(speed) > 1 else 0.0

    return [
        _flare_class_ordinal(max_flux),
        math.log10(max(mean_flux, 1e-12)),
        float(count_c),
        float(count_m),
        float(count_x),
        min(bz) if bz else 0.0,
        float(np.mean(bz)) if bz else 0.0,
        max(speed) if speed else 400.0,
        float(np.mean(speed)) if speed else 400.0,
        speed_jump,
        float(np.mean(density)) if density else 5.0,
        kp[-1] if kp else 2.0,  # most recent observed Kp: the persistence feature
        max(kp) if kp else 2.0,  # max observed Kp in the lookback window
        float(np.mean(kp)) if kp else 2.0,
    ]


PHYSICAL_FEATURE_NAMES = [
    "flare_class_ordinal",
    "log_mean_flux",
    "count_c_class_minutes",
    "count_m_class_minutes",
    "count_x_class_minutes",
    "min_bz",
    "mean_bz",
    "max_speed",
    "mean_speed",
    "speed_jump",
    "mean_density",
    "last_kp",
    "max_kp",
    "mean_kp",
]

TARGETS = ["kp_max_3day", "r1_r2_radio_blackout_pct", "s1_radiation_storm_pct"]
LAG_STEPS = (1, 2)
LAG_FEATURE_NAMES = [f"prev{'' if s == 1 else s}_{t}" for s in LAG_STEPS for t in TARGETS]
FEATURE_NAMES = PHYSICAL_FEATURE_NAMES + LAG_FEATURE_NAMES

# An issuance missing its immediate predecessor (the dataset's start, or a true NOAA
# archive gap like September 2024) falls back to this many hours as the "too stale to
# use" cutoff; its lag features are then imputed with the train-set mean instead.
_MAX_LAG_GAP_HOURS = 18


def _load_sorted(path: Path) -> list[dict]:
    """Load every row with complete scalar targets, sorted chronologically."""
    rows = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if all(t in row["scalars"] for t in TARGETS):
                rows.append(row)
    rows.sort(key=lambda r: r["issued_at"])
    return rows


def _add_lag_features(rows: list[dict], train_means: dict[str, float] | None) -> None:
    """Attach ``_lag`` (previous two issuances' own NOAA forecasts) to each row, in place.

    Includes both the immediately preceding issuance (~12h earlier) and the one before
    that (~24h earlier), since NOAA's 3-day forecasts change slowly and a two-step
    history smooths single-issuance noise, particularly useful for the sparser
    radiation-storm target.

    Args:
        rows: Chronologically sorted rows (across all splits: lag uses true history,
            which is legitimate in a real forecasting deployment regardless of where
            our evaluation split boundaries fall).
        train_means: Per-target fallback for a row with no usable predecessor (the
            dataset's start, or a gap wider than :data:`_MAX_LAG_GAP_HOURS`, such as the
            September 2024 archive gap).
    """
    from datetime import datetime

    history: list[dict] = []
    for row in rows:
        issued_at = datetime.fromisoformat(row["issued_at"])
        lag = []
        for steps_back in LAG_STEPS:
            candidate = history[-steps_back] if len(history) >= steps_back else None
            if candidate is not None:
                gap_hours = (issued_at - datetime.fromisoformat(candidate["issued_at"])).total_seconds() / 3600
            if candidate is not None and gap_hours <= _MAX_LAG_GAP_HOURS * steps_back:
                lag.extend(candidate["scalars"][t] for t in TARGETS)
            else:
                lag.extend(train_means[t] for t in TARGETS)
        row["_lag"] = lag
        history.append(row)


def _features(row: dict) -> list[float]:
    return _physical_features(row) + row["_lag"]


def _load(path: Path) -> dict[str, list]:
    rows = _load_sorted(path)
    train_rows = [r for r in rows if r["split"] == "train"]
    train_means = {t: float(np.mean([r["scalars"][t] for r in train_rows])) for t in TARGETS}
    _add_lag_features(rows, train_means)
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        by_split[row["split"]].append(row)
    return by_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--model",
        choices=["ridge", "gbr"],
        default="gbr",
        help="ridge = linear baseline, gbr = gradient-boosted trees (default)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="grid-search GBR hyperparameters against the validation set (62 rows: "
        "noisy enough that this sometimes overfits val and hurts test R2; off by default)",
    )
    args = parser.parse_args()

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    by_split = _load(args.data)
    print({k: len(v) for k, v in by_split.items()})

    X_train = np.array([_features(r) for r in by_split["train"]])
    X_val = np.array([_features(r) for r in by_split["val"]])
    X_test = np.array([_features(r) for r in by_split["test"]])

    for target in TARGETS:
        y_train = np.array([r["scalars"][target] for r in by_split["train"]])
        y_val = np.array([r["scalars"][target] for r in by_split["val"]])
        y_test = np.array([r["scalars"][target] for r in by_split["test"]])

        if args.model == "ridge":
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
        elif args.tune:
            best_val_mae, model = float("inf"), None
            for n_estimators in (50, 100, 200):
                for max_depth in (1, 2, 3):
                    for lr in (0.02, 0.05, 0.1):
                        candidate = GradientBoostingRegressor(
                            n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, random_state=0
                        )
                        candidate.fit(X_train, y_train)
                        mae = float(np.mean(np.abs(candidate.predict(X_val) - y_val)))
                        if mae < best_val_mae:
                            best_val_mae, model = mae, candidate
            print(
                f"  best hyperparams: n_estimators={model.n_estimators} "
                f"max_depth={model.max_depth} lr={model.learning_rate}"
            )
        else:
            model = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, random_state=0)
            model.fit(X_train, y_train)

        baseline = float(np.mean(y_train))
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        val_base_mae = float(np.mean(np.abs(baseline - y_val)))
        test_mae = float(np.mean(np.abs(test_pred - y_test)))
        test_base_mae = float(np.mean(np.abs(baseline - y_test)))
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        # R^2 of the constant-baseline predictor, for reference (usually near 0 on
        # train-set mean by construction, but can be negative out-of-sample).
        test_base_r2 = r2_score(y_test, np.full_like(y_test, baseline))

        print(
            f"{target}: val MAE model={val_mae:.3f} baseline={val_base_mae:.3f} | "
            f"test MAE model={test_mae:.3f} baseline={test_base_mae:.3f}"
        )
        print(
            f"  R2: val={val_r2:.3f} test={test_r2:.3f} (baseline test R2={test_base_r2:.3f})"
        )
        if args.model == "gbr":
            importances = sorted(
                zip(FEATURE_NAMES, model.feature_importances_), key=lambda kv: -kv[1]
            )
            top = ", ".join(f"{name}={imp:.2f}" for name, imp in importances[:4])
            print(f"  top features: {top}")


if __name__ == "__main__":
    main()
