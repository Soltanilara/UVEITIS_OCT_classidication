#!/usr/bin/env python3
# evaluate_bootstrap.py
"""
Patient-level bootstrap evaluation of ROC/PR curves.

Example
-------
python evaluate_bootstrap.py \
    --meta test.csv \
    --probs test_probabilities.csv \
    --threshold 0.50 \
    --n_boot 2000 \
    --save_dir "Bootstrap Outputs"
"""
import os, argparse, sys, warnings
import numpy as np
import pandas as pd
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='serif', size=14)   # or 12-14 range
matplotlib.rc('axes', titlesize=14)
matplotlib.rc('axes', labelsize=14)
matplotlib.rc('legend', fontsize=12)
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- helper functions -------------------------------------------------
def get_patient_id(path: str) -> str:
    """Extract patient identifier (first path component)."""
    return str(path).split("/")[0]

def bootstrap_patients(df, n_boot, fpr_grid, rec_grid, rng, threshold):
    """
    Return dictionaries containing bootstrap distributions:
      • roc_auc_boot   (n_boot,)
      • tpr_boot       (n_boot, len(fpr_grid))
      • pr_auc_boot    (n_boot,)
      • prec_boot      (n_boot, len(rec_grid))
      • pos_rate_boot  (n_boot,)
      • thr_metrics    dict of arrays (precision, recall, spec, f1, youden)
    """
    patients = df["PatientID"].unique()
    n_patients = len(patients)

    # containers
    roc_auc_b     = []
    pr_auc_b      = []
    pos_rate_b    = []
    tpr_b         = []
    prec_b        = []
    tm_precision  = []
    tm_recall     = []
    tm_spec       = []
    tm_f1         = []
    tm_youden     = []

    # data columns for speed
    y_full   = df["True_Label"].to_numpy()
    p_full   = df["Prob_Positive"].to_numpy()
    pid_full = df["PatientID"].to_numpy()

    # index array per patient (pre-computed once for efficiency)
    idx_per_patient = {
        pid: np.where(pid_full == pid)[0] for pid in patients
    }

    for _ in trange(n_boot, leave=False):
        # sample patients with replacement until both classes appear
        while True:
            samp_pids = rng.choice(patients, size=n_patients, replace=True)
            idx = np.concatenate([idx_per_patient[pid] for pid in samp_pids])
            y, p = y_full[idx], p_full[idx]
            if y.min() != y.max():        # contains both classes
                break

        # prevalence bootstrapped (for PR “no-skill” band)
        pos_rate_b.append(y.mean())

        # ROC
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc_b.append(auc(fpr, tpr))
        tpr_b.append(np.interp(fpr_grid, fpr, tpr))

        # PR
        prec, rec, _ = precision_recall_curve(y, p)
        pr_auc_b.append(auc(rec, prec))
        prec_b.append(np.interp(rec_grid, rec[::-1], prec[::-1]))

        # threshold metrics at p ≥ 0.5  (change if you want another thr)
        y_pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
        tm_precision.append(precision_score(y, y_pred, zero_division=0))
        tm_recall.append(recall_score(y, y_pred, zero_division=0))
        tm_spec.append(tn / (tn + fp))
        tm_f1.append(f1_score(y, y_pred, zero_division=0))
        tm_youden.append(tm_recall[-1] + tm_spec[-1] - 1)

    return dict(
        roc_auc=np.array(roc_auc_b),
        pr_auc=np.array(pr_auc_b),
        pos_rate=np.array(pos_rate_b),
        tpr=np.vstack(tpr_b),
        prec=np.vstack(prec_b),
        thr_metrics=dict(
            precision=np.array(tm_precision),
            recall=np.array(tm_recall),
            specificity=np.array(tm_spec),
            f1=np.array(tm_f1),
            youden=np.array(tm_youden)
        )
    )

def ci(arr, alpha=0.95):
    lo, hi = (1 - alpha) / 2 * 100, (1 + alpha) / 2 * 100
    return np.percentile(arr, [lo, hi])

# ---------- main -------------------------------------------------------------
def main(args):
    # ------------------------------------------------------------------- I/O
    os.makedirs(args.save_dir, exist_ok=True)
    df_meta = pd.read_csv(args.meta)
    df_prob = pd.read_csv(args.probs)
    if len(df_meta) != len(df_prob):
        sys.exit("❌ CSV row counts differ – make sure they correspond 1-to-1.")

    # merge (row-wise correspondence)
    df = pd.concat([df_meta.reset_index(drop=True),
                    df_prob.reset_index(drop=True)], axis=1)

    # clean / harmonise labels
    if "Label" in df.columns:                   # text labels
        df["True_Label"] = df["Label"].str.lower().map({"negative":0, "mild":1, "moderate":1, "severe":1})
    if "True_Label" not in df.columns:
        sys.exit("❌ 'True_Label' column not found.")

    df["PatientID"] = df["Image File"].apply(get_patient_id)

    # point estimates on full data
    y_true = df["True_Label"].to_numpy()
    y_prob = df["Prob_Positive"].to_numpy()
    fpr, tpr, _          = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    roc_auc_full = auc(fpr, tpr)
    pr_auc_full  = auc(recall, precision)
    pos_rate     = y_true.mean()

    # ---------------------------------------------------------------- bootstrap
    fpr_grid  = np.linspace(0, 1, 101)
    rec_grid  = np.linspace(0, 1, 101)
    rng       = np.random.default_rng(args.seed)
    boot = bootstrap_patients(
        df=df, n_boot=args.n_boot, fpr_grid=fpr_grid,
        rec_grid=rec_grid, rng=rng, threshold = args.threshold
    )

    # ------------- confidence intervals for curves & scalar metrics ----------
    tpr_lo = np.percentile(boot["tpr"], q=(100 - args.ci) / 2, axis=0)
    tpr_hi = np.percentile(boot["tpr"], q=100 - (100 - args.ci) / 2, axis=0)
    prec_lo = np.percentile(boot["prec"], q=(100 - args.ci) / 2, axis=0)
    prec_hi = np.percentile(boot["prec"], q=100 - (100 - args.ci) / 2, axis=0)
    roc_auc_ci       = ci(boot["roc_auc"], alpha=args.ci/100)
    pr_auc_ci        = ci(boot["pr_auc"],  alpha=args.ci/100)
    base_ci          = ci(boot["pos_rate"], alpha=args.ci/100)

    # threshold metrics CIs
    thr_ci = {k: ci(v, alpha=args.ci/100) for k, v in boot["thr_metrics"].items()}
    # point metrics
    y_pred_full = (y_prob >= args.threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_full, labels=[0,1]).ravel()
    pt_precision = precision_score(y_true, y_pred_full, zero_division=0)
    pt_recall    = recall_score(y_true, y_pred_full, zero_division=0)
    pt_spec      = tn / (tn + fp)
    pt_f1        = f1_score(y_true, y_pred_full, zero_division=0)
    pt_youden    = pt_recall + pt_spec - 1

    # ------------------------------- plotting ---------------------------------
    dpi = 300
    # ROC
    plt.figure(figsize=(5, 5), dpi=dpi)
    plt.fill_between(fpr_grid, tpr_lo, tpr_hi, color="tab:blue", alpha=0.2,
                     label=f"95% CI")
    plt.plot(fpr, tpr, lw=3, color="tab:blue",
             label=f"ROC (AUC={roc_auc_full:.3f} "
                   f"[{roc_auc_ci[0]:.3f},{roc_auc_ci[1]:.3f}])")
    plt.plot([0,1],[0,1],'--', color='gray', lw=1, label="No Skill")
    plt.xlabel("1 – Specificity (FPR)"); plt.ylabel("Sensitivity (TPR)")
    plt.xlim(-0.01,1.01); plt.ylim(-0.01,1.01); plt.legend(loc="lower right")
    # plt.grid()
    plt.tight_layout()
    roc_path = os.path.join(args.save_dir, "test_roc.png")
    plt.savefig(roc_path); plt.close()

    # PR
    plt.figure(figsize=(5, 5), dpi=dpi)
    plt.fill_between(rec_grid, prec_lo, prec_hi, color="tab:blue", alpha=0.2,
                     label="95% CI")
    plt.plot(recall, precision, lw=3, color="tab:blue",
             label=f"PR (AUC={pr_auc_full:.3f} "
                   f"[{pr_auc_ci[0]:.3f},{pr_auc_ci[1]:.3f}])")
    # no-skill baseline prevalence
    plt.axhline(pos_rate, color="gray", linestyle="--",
                label=f"No Skill (AUC={pos_rate:.3f})")
    # # shaded CI for prevalence
    # plt.axhspan(base_ci[0], base_ci[1], color="gray", alpha=0.15,
    #             label="No Skill 95% CI")
    plt.xlabel("Sensitivity (Recall)"); plt.ylabel("PPV (Precision)")
    plt.xlim(-0.01,1.01); plt.ylim(-0.01,1.01); plt.legend(loc="lower left")
    # plt.grid()
    plt.tight_layout()
    pr_path = os.path.join(args.save_dir, "test_pr.png")
    plt.savefig(pr_path); plt.close()

    # ----------------------------- metric CSV ---------------------------------
    rows = []
    for name, pt, (lo, hi) in [
        ("precision", pt_precision, thr_ci["precision"]),
        ("recall",    pt_recall,    thr_ci["recall"]),
        ("specificity", pt_spec,    thr_ci["specificity"]),
        ("f1",        pt_f1,        thr_ci["f1"]),
        ("youden_j",  pt_youden,    thr_ci["youden"])
    ]:
        rows.append({"metric": name,
                     "point_estimate": pt,
                     f"ci_{(100 - args.ci) / 2:.1f}%": lo,
                     f"ci_{100 - (100 - args.ci) / 2:.1f}%": hi})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(os.path.join(args.save_dir, "threshold_metrics.csv"),
                      index=False)

    print(f"\n✅ Finished. Results saved in: {args.save_dir}\n"
          f"  • ROC curve → {roc_path}\n"
          f"  • PR curve  → {pr_path}\n"
          f"  • Metrics    → threshold_metrics.csv")

# ---------- entry-point CLI --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Patient-level bootstrap ROC/PR evaluation"
    )
    # parser.add_argument("--meta",       default="output_ext_test/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_0.001_decay_1e-05_10_10_1_1e-05_epochs_100_hflip_seed_0/test.csv")
    # parser.add_argument("--probs",      default="output_ext_test/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_0.001_decay_1e-05_10_10_1_1e-05_epochs_100_hflip_seed_0/test_probabilities.csv")
    # parser.add_argument("--save_dir",   default="output_ext_test/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_0.001_decay_1e-05_10_10_1_1e-05_epochs_100_hflip_seed_0/Bootstrap Outputs")
    # parser.add_argument("--meta",       default="CrossValidationOutputs/test_aggregated_folds.csv")
    # parser.add_argument("--probs",      default="CrossValidationOutputs/test_probabilities_aggregated_folds.csv")
    # parser.add_argument("--save_dir",   default="CrossValidationOutputs/Bootstrap Outputs")
    # parser.add_argument("--threshold",  type=float, default=0.085,
    #                     help="Decision threshold for metrics (default 0.5)")
    parser.add_argument("--meta",       default="fold_9/test.csv")
    parser.add_argument("--probs",      default="output_fold_9/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_1e-05_decay_0.001_10_10_1_1e-07_epochs_100_hflip_seed_0/test_probabilities.csv")
    parser.add_argument("--save_dir",   default="fold_9_bootstrap")
    parser.add_argument("--threshold",  type=float, default=0.450,
                        help="Decision threshold for metrics (default 0.5)")
    parser.add_argument("--n_boot",     type=int,   default=10000,
                        help="Number of bootstrap replicates (default 1000)")
    parser.add_argument("--ci",         type=float, default=95,
                        help="Confidence level in percent (default 95)")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random-seed for reproducibility")
    main(parser.parse_args())
