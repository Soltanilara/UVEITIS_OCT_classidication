#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def derive_experiment_fields(root: Path, run_dir: Path) -> dict[str, str]:
    rel_parts = run_dir.relative_to(root).parts
    experiment = rel_parts[0] if len(rel_parts) >= 1 else run_dir.name
    fold = next((part for part in rel_parts if part.startswith("fold_")), "")
    return {
        "experiment": experiment,
        "fold": fold,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
    }


def collect_run_rows(root: Path, split: str) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(root.rglob(f"{split}_summary.json")):
        run_dir = summary_path.parent
        summary = load_json(summary_path)
        metadata_path = run_dir / "train_metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}

        row = derive_experiment_fields(root, run_dir)
        row.update(
            {
                "split": split,
                "mean_zone_binary_f1": summary.get("mean_binary_f1"),
                "mean_zone_accuracy": summary.get("mean_accuracy"),
                "mean_zone_precision": summary.get("mean_precision"),
                "mean_zone_recall": summary.get("mean_recall"),
                "mean_zone_specificity": summary.get("mean_specificity"),
                "any_positive_visit_f1": summary.get("any_positive_visit_f1"),
                "any_positive_visit_accuracy": summary.get("any_positive_visit_accuracy"),
                "any_positive_visit_sensitivity": summary.get("any_positive_visit_sensitivity"),
                "any_positive_visit_specificity": summary.get("any_positive_visit_specificity"),
                "any_positive_visit_roc_auc": summary.get("any_positive_visit_roc_auc"),
                "any_positive_visit_pr_auc": summary.get("any_positive_visit_pr_auc"),
                "loss": summary.get("loss"),
                "best_epoch": metadata.get("best_epoch"),
                "input_mode": metadata.get("input_mode"),
                "drop_missing_zone_rows": metadata.get("drop_missing_zone_rows"),
                "label_smoothing": metadata.get("label_smoothing"),
                "loss_name": metadata.get("loss"),
                "gamma": metadata.get("gamma"),
                "swa_enabled": metadata.get("swa_enabled"),
                "fundus_pretrained_ckpt": metadata.get("fundus_pretrained_ckpt"),
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(
        by=["mean_zone_binary_f1", "mean_zone_accuracy", "any_positive_visit_f1"],
        ascending=[False, False, False],
        inplace=True,
    )
    return df


def aggregate_experiments(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return run_df

    grouped = (
        run_df.groupby("experiment", dropna=False)
        .agg(
            runs=("run_dir", "count"),
            mean_zone_binary_f1_mean=("mean_zone_binary_f1", "mean"),
            mean_zone_binary_f1_std=("mean_zone_binary_f1", "std"),
            mean_zone_accuracy_mean=("mean_zone_accuracy", "mean"),
            any_positive_visit_f1_mean=("any_positive_visit_f1", "mean"),
            any_positive_visit_accuracy_mean=("any_positive_visit_accuracy", "mean"),
            any_positive_visit_specificity_mean=("any_positive_visit_specificity", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .reset_index()
    )
    grouped.sort_values(
        by=["mean_zone_binary_f1_mean", "mean_zone_accuracy_mean", "any_positive_visit_f1_mean"],
        ascending=[False, False, False],
        inplace=True,
    )
    return grouped


def main():
    parser = argparse.ArgumentParser(description="Rank zone-classification runs by zone and derived any-positive metrics.")
    parser.add_argument("--root", type=str, required=True, help="Root output directory to scan.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--output_csv", type=str, default="", help="Run-level CSV output path.")
    parser.add_argument("--output_grouped_csv", type=str, default="", help="Experiment-level CSV output path.")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root)
    run_df = collect_run_rows(root, args.split)
    if run_df.empty:
        raise SystemExit(f"No {args.split}_summary.json files found under {root}")

    grouped_df = aggregate_experiments(run_df)
    output_csv = args.output_csv or str(root / f"{args.split}_run_ranking.csv")
    output_grouped_csv = args.output_grouped_csv or str(root / f"{args.split}_experiment_ranking.csv")

    run_df.to_csv(output_csv, index=False)
    grouped_df.to_csv(output_grouped_csv, index=False)

    print(f"Saved run-level ranking to: {output_csv}")
    print(f"Saved experiment-level ranking to: {output_grouped_csv}")
    print()
    print("Top runs:")
    display_cols = [
        "experiment",
        "fold",
        "run_name",
        "mean_zone_binary_f1",
        "mean_zone_accuracy",
        "any_positive_visit_f1",
        "any_positive_visit_accuracy",
        "any_positive_visit_specificity",
    ]
    print(run_df[display_cols].head(args.top_k).to_string(index=False))
    print()
    print("Top experiments:")
    print(grouped_df.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
