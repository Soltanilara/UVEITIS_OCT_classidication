#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


NUM_ZONES = 10


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = []
    for zone_idx in range(1, NUM_ZONES + 1):
        required.extend(
            [
                f"Zone{zone_idx}_Observed",
                f"Zone{zone_idx}_True",
                f"Zone{zone_idx}_Prob_1",
            ]
        )
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Prediction CSV is missing required columns: {missing}")
    return df


def load_initial_thresholds(path: str | None) -> np.ndarray:
    if not path:
        return np.full(NUM_ZONES, 0.5, dtype=np.float32)

    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "thresholds" in payload:
        payload = payload["thresholds"]

    if isinstance(payload, dict):
        values = [payload.get(f"Zone{i}", 0.5) for i in range(1, NUM_ZONES + 1)]
    elif isinstance(payload, list):
        values = payload
    else:
        raise ValueError(f"Unsupported threshold JSON format in {path}")

    if len(values) != NUM_ZONES:
        raise ValueError(f"Expected {NUM_ZONES} thresholds, found {len(values)}")
    return np.asarray(values, dtype=np.float32)


def zone_mean_f1(df: pd.DataFrame, thresholds: np.ndarray) -> float:
    scores = []
    for zone_idx in range(1, NUM_ZONES + 1):
        observed = df[f"Zone{zone_idx}_Observed"].to_numpy(dtype=int) == 1
        if not observed.any():
            continue
        y_true = df.loc[observed, f"Zone{zone_idx}_True"].to_numpy(dtype=int)
        y_pred = (df.loc[observed, f"Zone{zone_idx}_Prob_1"].to_numpy(dtype=float) >= thresholds[zone_idx - 1]).astype(int)
        scores.append(f1_score(y_true, y_pred, zero_division=0))
    return float(np.mean(scores)) if scores else 0.0


def derived_any_positive_f1(df: pd.DataFrame, thresholds: np.ndarray) -> float:
    any_true = []
    any_pred = []
    for _, row in df.iterrows():
        observed_true = []
        observed_pred = []
        for zone_idx in range(1, NUM_ZONES + 1):
            if int(row[f"Zone{zone_idx}_Observed"]) != 1:
                continue
            observed_true.append(int(row[f"Zone{zone_idx}_True"]))
            observed_pred.append(int(float(row[f"Zone{zone_idx}_Prob_1"]) >= thresholds[zone_idx - 1]))
        if not observed_true:
            continue
        any_true.append(int(any(observed_true)))
        any_pred.append(int(any(observed_pred)))
    if not any_true:
        return 0.0
    return float(f1_score(any_true, any_pred, zero_division=0))


def tune_zone_thresholds(df: pd.DataFrame, search_grid: np.ndarray, initial_thresholds: np.ndarray):
    tuned = initial_thresholds.copy()
    per_zone_rows = []

    for zone_idx in range(1, NUM_ZONES + 1):
        observed = df[f"Zone{zone_idx}_Observed"].to_numpy(dtype=int) == 1
        if not observed.any():
            per_zone_rows.append(
                {
                    "Zone": zone_idx,
                    "ObservedCount": 0,
                    "InitialThreshold": float(initial_thresholds[zone_idx - 1]),
                    "BestThreshold": float(initial_thresholds[zone_idx - 1]),
                    "BestF1": np.nan,
                }
            )
            continue

        y_true = df.loc[observed, f"Zone{zone_idx}_True"].to_numpy(dtype=int)
        y_prob = df.loc[observed, f"Zone{zone_idx}_Prob_1"].to_numpy(dtype=float)

        best_threshold = float(initial_thresholds[zone_idx - 1])
        best_score = -1.0
        for threshold in search_grid:
            y_pred = (y_prob >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score or (np.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                best_score = score
                best_threshold = float(threshold)

        tuned[zone_idx - 1] = best_threshold
        per_zone_rows.append(
            {
                "Zone": zone_idx,
                "ObservedCount": int(observed.sum()),
                "InitialThreshold": float(initial_thresholds[zone_idx - 1]),
                "BestThreshold": best_threshold,
                "BestF1": float(best_score),
            }
        )

    return tuned, pd.DataFrame(per_zone_rows)


def main():
    parser = argparse.ArgumentParser(description="Tune one threshold per zone from validation predictions.")
    parser.add_argument("--predictions_csv", type=str, required=True, help="Validation predictions CSV from train_kFold_binary.py")
    parser.add_argument("--output_json", type=str, default="", help="Where to save the tuned threshold JSON.")
    parser.add_argument("--output_csv", type=str, default="", help="Where to save per-zone tuning details.")
    parser.add_argument("--initial_thresholds_json", type=str, default="", help="Optional JSON with starting thresholds.")
    parser.add_argument("--search_start", type=float, default=0.30)
    parser.add_argument("--search_end", type=float, default=0.70)
    parser.add_argument("--search_step", type=float, default=0.02)
    args = parser.parse_args()

    df = load_predictions(args.predictions_csv)
    initial_thresholds = load_initial_thresholds(args.initial_thresholds_json or None)
    search_grid = np.round(np.arange(args.search_start, args.search_end + 1e-8, args.search_step), 4)

    baseline_zone_f1 = zone_mean_f1(df, initial_thresholds)
    baseline_any_positive_f1 = derived_any_positive_f1(df, initial_thresholds)
    tuned_thresholds, per_zone_df = tune_zone_thresholds(df, search_grid, initial_thresholds)
    tuned_zone_f1 = zone_mean_f1(df, tuned_thresholds)
    tuned_any_positive_f1 = derived_any_positive_f1(df, tuned_thresholds)

    output_json = args.output_json or os.path.join(os.path.dirname(args.predictions_csv), "zone_thresholds.json")
    output_csv = args.output_csv or os.path.join(os.path.dirname(args.predictions_csv), "zone_threshold_tuning.csv")

    payload = {
        "source_predictions_csv": args.predictions_csv,
        "search_grid": [float(x) for x in search_grid],
        "thresholds": {f"Zone{i}": float(tuned_thresholds[i - 1]) for i in range(1, NUM_ZONES + 1)},
        "baseline": {
            "mean_binary_f1": float(baseline_zone_f1),
            "any_positive_visit_f1": float(baseline_any_positive_f1),
        },
        "tuned": {
            "mean_binary_f1": float(tuned_zone_f1),
            "any_positive_visit_f1": float(tuned_any_positive_f1),
        },
        "kept_only_if_improves_both": bool(
            tuned_zone_f1 > baseline_zone_f1 and tuned_any_positive_f1 > baseline_any_positive_f1
        ),
    }

    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    per_zone_df.to_csv(output_csv, index=False)

    print(f"Baseline zone mean F1: {baseline_zone_f1:.4f}")
    print(f"Tuned zone mean F1:    {tuned_zone_f1:.4f}")
    print(f"Baseline any-pos F1:   {baseline_any_positive_f1:.4f}")
    print(f"Tuned any-pos F1:      {tuned_any_positive_f1:.4f}")
    print(f"Saved thresholds to:   {output_json}")
    print(f"Saved per-zone table:  {output_csv}")


if __name__ == "__main__":
    main()
