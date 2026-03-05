import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, norm


# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------

def fisher_z_transform(r):
    """Convert correlation r to Fisher's z. r must be in (-1, +1)."""
    return 0.5 * np.log((1 + r) / (1 - r))


def inverse_fisher_z(z):
    """Convert Fisher's z back to correlation."""
    return (math.exp(2 * z) - 1) / (math.exp(2 * z) + 1)


def stouffer_z(p_value, r=None):
    """
    Convert a *two-sided* p-value to a signed Z-score for Stouffer’s method.
    If r is given, we use sign(r) to indicate direction of effect.
    """
    if not (0 < p_value < 1):
        # degenerate or out-of-bounds p
        return np.nan

    # The 2-sided p => 1-sided tail is p/2.
    # Z_abs = inverseCDF(1 - p/2)
    z_abs = norm.ppf(1 - p_value / 2)
    if r is None or r == 0 or math.isnan(r):
        sign_factor = 1.0  # no direction info, treat as positive
    else:
        sign_factor = np.sign(r)

    return sign_factor * z_abs


# --------------------------------------------------------------------
# Main aggregator function
# --------------------------------------------------------------------
def aggregate_correlations_and_pvalues(
        df_full,
        min_dates=4,
        output_folder="Count for aggregate severity diagnosis"
):
    """
    1) Group by (patient, eye, region).
    2) Skip groups with < min_dates distinct date points.
    3) Compute Spearman and Pearson correlation + p-value for each group.
    4) Use Fisher’s Z with weight=(n-3) to get a meta-analytic correlation for each method,
       but clip perfect ±1 to ±(1 - 1e-12) so we can do the transform.
    5) Combine p-values with Weighted Stouffer’s Z, weighting by sqrt(n).
    6) Return a final DataFrame with each group's correlation stats + a single row of global stats.
    7) Save output to a CSV in output_folder.
    """

    os.makedirs(output_folder, exist_ok=True)

    # (A) Compute per-group correlation
    group_rows = []
    grouped = df_full.groupby(["patient", "eye", "region"])
    for (patient, eye, region), subdf_group in grouped:
        # How many distinct dates?
        n_dates = subdf_group["date"].nunique()
        if n_dates < min_dates:
            # skip entire group
            continue

        # We'll group by date (within this sub-group) to produce pairs:
        #   x = weighted_normalized, y = normalized_positive
        date_vals = []
        weighted_vals = []
        norm_pos_vals = []

        # Build the per-date stats
        severity_scores = {"negative": 0, "mild": 1, "moderate": 2, "severe": 3}
        sub_grouped_by_date = subdf_group.groupby("date")
        for date_val, date_data in sub_grouped_by_date:
            n_rows = len(date_data)
            # # fraction labeled mild/moderate/severe
            # mms_count = date_data["Label"].isin(["mild", "moderate", "severe"]).sum()
            # norm_pos = mms_count / n_rows
            prob_pos_vals = date_data["prob_positive"]
            norm_pos = prob_pos_vals.mean()

            # weighted normalized
            score_sum = 0
            for lbl, sc in severity_scores.items():
                score_sum += sc * (date_data["Label"] == lbl).sum()
            weighted_norm = score_sum / (3.0 * n_rows)

            date_vals.append(date_val)
            weighted_vals.append(weighted_norm)
            norm_pos_vals.append(norm_pos)

        # If fewer than 2 points remain (somehow), skip
        if len(weighted_vals) < 2:
            continue

        # Spearman
        rho_spearman, pval_spearman = spearmanr(weighted_vals, norm_pos_vals)
        # Pearson
        rho_pearson, pval_pearson = pearsonr(weighted_vals, norm_pos_vals)

        # Store
        group_rows.append({
            "patient": patient,
            "eye": eye,
            "region": region,
            "n_dates": len(weighted_vals),  # typically == n_dates
            "spearman_r": rho_spearman,
            "spearman_p": pval_spearman,
            "pearson_r": rho_pearson,
            "pearson_p": pval_pearson
        })

    df_corr = pd.DataFrame(group_rows)
    if df_corr.empty:
        print(f"No groups have at least {min_dates} distinct dates. Nothing to aggregate.")
        return None, None

    # (B) Meta-analysis for correlation (Fisher’s Z)
    # We'll do it for Spearman and Pearson separately, weighting by (n-3).
    # We will handle perfect ±1 by clipping to ±(1 - 1e-12).

    def meta_fisher_z(df_in, r_col="spearman_r"):
        """
        Weighted average correlation using Fisher Z.
        We'll use w_i = (n-3).
        We skip correlation if it is NaN, but clip ±1 to ±(1 - 1e-12).
        """
        valid_rows = []
        for i, row in df_in.iterrows():
            r_val = row[r_col]
            n_val = row["n_dates"]

            if pd.isna(r_val):
                # constant input => correlation is NaN => skip
                continue

            # clip perfect correlation
            if abs(r_val) >= 1.0:
                r_val = 0.999999999999 if r_val > 0 else -0.999999999999

            # now transform
            try:
                z_val = fisher_z_transform(r_val)
            except Exception:
                continue  # skip if it fails for some numerical reason

            w_val = (n_val - 3)
            # Must be > 0 for inverse variance weighting
            if w_val > 0:
                valid_rows.append((z_val, w_val))

        if not valid_rows:
            # nothing to combine
            return (np.nan, np.nan, np.nan)

        zs, ws = zip(*valid_rows)
        z_weighted_avg = np.average(zs, weights=ws)
        sum_w = sum(ws)

        # convert back to correlation
        r_meta = inverse_fisher_z(z_weighted_avg)

        # approximate SE on z-scale
        se_z = 1.0 / math.sqrt(sum_w)
        z_low = z_weighted_avg - 1.96 * se_z
        z_high = z_weighted_avg + 1.96 * se_z
        r_low = inverse_fisher_z(z_low)
        r_high = inverse_fisher_z(z_high)

        return (r_meta, r_low, r_high)

    # (C) Weighted Stouffer’s for p-values, weighting by sqrt(n).
    # We'll do sign(rho) in the Z to keep track of direction, but
    # the final p-value is two-sided.

    def weighted_stouffer_p(df_in, r_col="spearman_r", p_col="spearman_p"):
        """
        Weighted Stouffer’s method combining p-values with weights = sqrt(n).
        We'll skip if p is NaN or out of (0,1), or if correlation is NaN.
        We'll also do the same ±1 clipping for the correlation's sign
        (though sign(±1) is just ±1).
        """
        z_vals = []
        w_vals = []
        for i, row in df_in.iterrows():
            r_val = row[r_col]
            p_val = row[p_col]
            n_val = row["n_dates"]

            # skip degenerate p
            if not (0 < p_val < 1):
                continue
            if pd.isna(r_val):
                continue

            # clip correlation if ±1
            if abs(r_val) >= 1.0:
                r_val = 0.999999999999 if r_val > 0 else -0.999999999999

            # compute the single-study Z
            zi = stouffer_z(p_val, r=r_val)
            if math.isnan(zi):
                continue

            w = math.sqrt(n_val)
            if w <= 0:
                continue

            z_vals.append(zi)
            w_vals.append(w)

        if not z_vals:
            return np.nan

        numerator = 0.0
        denom = 0.0
        for z_i, w_i in zip(z_vals, w_vals):
            numerator += w_i * z_i
            denom += (w_i ** 2)
        if denom <= 0:
            return np.nan

        Z_combined = numerator / math.sqrt(denom)
        # two-sided p
        p_comb = 2.0 * (1.0 - norm.cdf(abs(Z_combined)))
        return p_comb

    # Spearman meta-correlation
    spearman_r_meta, spearman_r_low, spearman_r_high = meta_fisher_z(df_corr, "spearman_r")
    # Pearson meta-correlation
    pearson_r_meta, pearson_r_low, pearson_r_high = meta_fisher_z(df_corr, "pearson_r")

    # Spearman combined p (Stouffer, sqrt(n))
    spearman_p_stouffer = weighted_stouffer_p(df_corr, r_col="spearman_r", p_col="spearman_p")
    # Pearson combined p (Stouffer, sqrt(n))
    pearson_p_stouffer = weighted_stouffer_p(df_corr, r_col="pearson_r", p_col="pearson_p")

    # (D) Summaries
    summary = {
        "spearman_r_meta": spearman_r_meta,
        "spearman_r_meta_ci_low": spearman_r_low,
        "spearman_r_meta_ci_high": spearman_r_high,
        "spearman_p_stouffer": spearman_p_stouffer,
        "pearson_r_meta": pearson_r_meta,
        "pearson_r_meta_ci_low": pearson_r_low,
        "pearson_r_meta_ci_high": pearson_r_high,
        "pearson_p_stouffer": pearson_p_stouffer,
        "num_valid_groups": len(df_corr)
    }

    # (E) Save aggregated results
    df_corr_out_path = os.path.join(output_folder, "per_group_correlations.csv")
    df_corr.to_csv(df_corr_out_path, index=False)
    print(f"Saved per-group correlations to: {df_corr_out_path}")

    summary_df = pd.DataFrame([summary])  # single-row
    summary_out_path = os.path.join(output_folder, "meta_aggregate_statistics.csv")
    summary_df.to_csv(summary_out_path, index=False)
    print(f"Saved aggregated meta statistics to: {summary_out_path}")

    return df_corr, summary


def fisher_confidence_interval(r, n, alpha=0.05):
    """
    Approximate 1 - alpha confidence interval for a correlation 'r' via Fisher's z-transform,
    using sample size n.
    If |r| >= 1 or n < 4, returns (NaN, NaN).
    """
    if abs(r) >= 1 or n < 4:
        return (np.nan, np.nan)
    # Fisher z
    z = 0.5 * np.log((1 + r) / (1 - r))
    # Standard error for z ~ 1/sqrt(n - 3)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha/2)  # e.g. ~1.96 for alpha=0.05
    z_low = z - z_crit * se
    z_high = z + z_crit * se
    # Inverse transform
    r_low = (math.exp(2*z_low) - 1) / (math.exp(2*z_low) + 1)
    r_high = (math.exp(2*z_high) - 1) / (math.exp(2*z_high) + 1)
    return (r_low, r_high)

def compute_per_patient_date_aggregates(
    df,
    output_folder="Count for aggregate severity diagnosis"
):
    """
    For each (patient, date, eye, region) with >=2 row entries,
    compute:
      - total_rows,
      - normalized_positive (fraction labeled mild/moderate/severe),
      - weighted_normalized = sum of severity scores / (3 * total_rows).

    Then, across all valid groups, compute Pearson & Spearman correlation
    between (normalized_positive) and (weighted_normalized),
    along with approximate 95% confidence intervals from Fisher's z-transform.

    Saves:
      1) 'per_patient_date_eye_region_aggregates.csv'
      2) 'per_patient_date_eye_region_correlation_stats.csv'
    in the given output_folder.
    """

    os.makedirs(output_folder, exist_ok=True)

    # Severity mapping
    severity_scores = {"negative": 0, "mild": 1, "moderate": 2, "severe": 3}

    # Group by (patient, date, eye, region)
    results = []
    grouped = df.groupby(["patient", "date", "eye", "region"])
    for (patient, date_val, eye, region), group_data in grouped:
        n_rows = len(group_data)

        # Skip if group has only 1 row
        if n_rows < 2:
            continue

        # # 1) normalized_positive
        # n_pos = group_data["Label"].isin(["mild", "moderate", "severe"]).sum()
        # normalized_positive = n_pos / n_rows
        normalized_positive = group_data["prob_positive"].mean()  # mean predicted P(+)

        # 2) weighted_normalized
        score_sum = 0
        for lbl, sc in severity_scores.items():
            count_lbl = (group_data["Label"] == lbl).sum()
            score_sum += sc * count_lbl
        weighted_normalized = score_sum / (3.0 * n_rows)

        results.append({
            "patient": patient,
            "date": date_val,
            "eye": eye,
            "region": region,
            "total_rows": n_rows,
            "normalized_positive": normalized_positive,
            "weighted_normalized": weighted_normalized
        })

    df_agg = pd.DataFrame(results)

    # Save the per-group aggregates
    agg_path = os.path.join(output_folder, "per_patient_date_eye_region_aggregates.csv")
    df_agg.to_csv(agg_path, index=False)
    print(f"Saved group-level aggregates to: {agg_path}")

    # If fewer than 2 total groups remain, correlation is meaningless
    if len(df_agg) < 2:
        print("Not enough valid (patient,date,eye,region) groups to compute correlation.")
        return df_agg, None

    # ------------------------------------------------------------
    # Compute the overall correlation across all valid groups
    # x = normalized_positive, y = weighted_normalized
    # ------------------------------------------------------------
    X = df_agg["normalized_positive"].values
    Y = df_agg["weighted_normalized"].values

    # Pearson
    pearson_r, pearson_p = pearsonr(X, Y)
    (pearson_r_low, pearson_r_high) = fisher_confidence_interval(pearson_r, len(df_agg))

    # Spearman
    spearman_r, spearman_p = spearmanr(X, Y)
    (spearman_r_low, spearman_r_high) = fisher_confidence_interval(spearman_r, len(df_agg))

    corr_stats = {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "pearson_r_ci_low": pearson_r_low,
        "pearson_r_ci_high": pearson_r_high,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "spearman_r_ci_low": spearman_r_low,
        "spearman_r_ci_high": spearman_r_high,
        "num_groups": len(df_agg)
    }
    df_corr_stats = pd.DataFrame([corr_stats])

    corr_path = os.path.join(output_folder, "per_patient_date_eye_region_correlation_stats.csv")
    df_corr_stats.to_csv(corr_path, index=False)
    print(f"Saved global correlation stats (with CIs) to: {corr_path}")

    return df_agg, df_corr_stats


# --------------------------------------------------------------------
# Example driver code
# --------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load the CSV
    # severity_csv = "output_ext_test/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_0.001_decay_1e-05_10_10_1_1e-05_epochs_100_hflip_seed_0/test.csv"  # ← ground-truth, same format as before
    # pred_csv = "output_ext_test/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_0.001_decay_1e-05_10_10_1_1e-05_epochs_100_hflip_seed_0/test_probabilities.csv"  # ← Prob_Positive, Prob_Negative, True_Label
    # out_folder = "Count (test probabilities) ext test"
    # severity_csv = "output_nebraska/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_0.001_decay_1e-05_10_10_1_1e-05_epochs_100_hflip_seed_0/test.csv"  # ← ground-truth, same format as before
    # pred_csv = "output_nebraska/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_0.001_decay_1e-05_10_10_1_1e-05_epochs_100_hflip_seed_0/test_probabilities.csv"  # ← Prob_Positive, Prob_Negative, True_Label
    # out_folder = "Count (test probabilities) nebraska"
    severity_csv = "CrossValidationOutputs/test_aggregated_folds.csv"
    pred_csv = "CrossValidationOutputs/test_probabilities_aggregated_folds.csv"  # ← Prob_Positive, Prob_Negative, True_Label
    out_folder = "Count (test probabilities) all folds"
    df_truth = pd.read_csv(severity_csv)
    df_pred = pd.read_csv(pred_csv)

    if len(df_truth) != len(df_pred):
        raise ValueError("The two CSVs must contain the same number of rows "
                         "in identical order!")

    df_truth["prob_positive"] = df_pred["Prob_Positive"].values

    # 2) Parse out patient, date, eye, region from "Image File"
    df_truth["parts"] = df_truth["Image File"].str.split('/')
    df_truth["patient"] = df_truth["parts"].str[0]
    df_truth["date"] = df_truth["parts"].str[1]
    df_truth["eye"] = df_truth["parts"].str[2]
    df_truth["region"] = df_truth["parts"].str[3]


    # 3) Run aggregator: skip groups with fewer than 4 distinct dates
    results_df, meta_stats = aggregate_correlations_and_pvalues(
        df_full=df_truth,
        min_dates=4,
        output_folder=out_folder
    )

    print("\n--- Per-group correlation results (head) ---")
    if results_df is not None and not results_df.empty:
        print(results_df.head())

    print("\n--- Global meta statistics ---")
    if meta_stats is not None:
        for k, v in meta_stats.items():
            print(f"{k}: {v}")

    # Compute and save results
    df_agg, df_corr_stats = compute_per_patient_date_aggregates(
        df_truth,
        output_folder=out_folder
    )