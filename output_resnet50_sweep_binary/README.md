# ResNet50 Binary Sweep Summary

This README summarizes the completed runs under `output_resnet50_sweep_binary`.

## Scope

- Completed runs found: 95
- Experiment groups found: 19
- Folds per experiment: 5
- Metrics summarized from each run: `val_summary.json`, `test_summary.json`, and `train_metadata.json`

Each experiment folder follows this pattern:

`<experiment>/fold_<k>/<run_name>/`

Inside each run folder, the most useful files are:

- `test_summary.json`
- `val_summary.json`
- `train_metadata.json`
- `test_zone_metrics.csv`
- `val_zone_metrics.csv`
- `history.png`

## Headline Takeaways

- Best mean test binary F1: `finetune_supervised_ce_unweighted_hflip_elastic_gnoise` at `0.6363 +- 0.0618`
- Best mean test accuracy and precision: `finetune_supervised_ce_unweighted_mixup_0p8` with accuracy `0.5785` and precision `0.5464`
- Best mean test specificity: `lin_eval_supervised_ce_unweighted_mixup_0p2` at `0.4681`
- Best lin-eval configuration by mean test binary F1: `lin_eval_supervised_focal_g2p0_unweighted` at `0.6038 +- 0.0710`
- `finetune` outperformed `lin_eval` on average mean test binary F1: `0.6038` vs `0.5859`
- Validation scores were much more optimistic for `finetune` than for `lin_eval`
- `fold_0` was the best test fold for 14 of 19 experiments

## Interpretation Notes

- The top F1 model, `finetune_supervised_ce_unweighted_hflip_elastic_gnoise`, achieves very high recall (`0.9655`) but extremely low specificity (`0.0773`). It appears to favor predicting the positive class aggressively.
- `finetune_supervised_ce_class_weighted` is a stronger balanced choice than the top-F1 model if you care about both F1 and classification balance. It has nearly the same mean test F1 (`0.6331`) with much better accuracy (`0.5727`) and specificity (`0.3607`).
- `finetune_supervised_ce_unweighted_mixup_0p2` and `mixup_0p8` were both competitive. `mixup_0p8` gave the best average accuracy, while `mixup_0p2` gave slightly better mean test F1.
- Adding `brightness_contrast` hurt performance in both protocols. Those variants landed at or near the bottom of the ranking.
- The most stable configuration across folds was `lin_eval_supervised_ce_unweighted_mixup_0p8`, with the lowest fold-to-fold test F1 standard deviation (`0.0195`).

## Protocol Comparison

| Protocol | Experiments | Mean val binary F1 | Mean test binary F1 | Mean val-test gap |
| --- | ---: | ---: | ---: | ---: |
| `finetune` | 9 | 0.6515 | 0.6038 | +0.0477 |
| `lin_eval` | 10 | 0.5923 | 0.5859 | +0.0064 |

This suggests the `lin_eval` runs were more stable from validation to test, while `finetune` usually scored better on test but also showed a larger validation-to-test drop.

## Cross-Fold Ranking

Sorted by mean test binary F1 across 5 folds.

| Rank | Experiment | Protocol | Mean test binary F1 | Mean test acc | Mean precision | Mean recall | Mean specificity | Mean val binary F1 | Val-test gap | Avg best epoch | Best fold |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `finetune_supervised_ce_unweighted_hflip_elastic_gnoise` | `finetune` | 0.6363 +- 0.0618 | 0.4959 | 0.4824 | 0.9655 | 0.0773 | 0.6293 | -0.0070 | 4.0 | `fold_0 (0.7338)` |
| 2 | `finetune_supervised_ce_class_weighted` | `finetune` | 0.6331 +- 0.1025 | 0.5727 | 0.5394 | 0.8214 | 0.3607 | 0.6543 | +0.0212 | 7.2 | `fold_0 (0.7670)` |
| 3 | `finetune_supervised_ce_unweighted_mixup_0p2` | `finetune` | 0.6318 +- 0.0873 | 0.5551 | 0.5231 | 0.8329 | 0.3129 | 0.6552 | +0.0234 | 9.2 | `fold_0 (0.7806)` |
| 4 | `finetune_supervised_ce_unweighted_mixup_0p8` | `finetune` | 0.6175 +- 0.0926 | 0.5785 | 0.5464 | 0.7625 | 0.4100 | 0.6555 | +0.0380 | 10.6 | `fold_0 (0.7797)` |
| 5 | `finetune_supervised_focal_g1p5_unweighted` | `finetune` | 0.6045 +- 0.1016 | 0.5479 | 0.5121 | 0.7731 | 0.3316 | 0.6508 | +0.0463 | 9.2 | `fold_0 (0.7603)` |
| 6 | `lin_eval_supervised_focal_g2p0_unweighted` | `lin_eval` | 0.6038 +- 0.0710 | 0.5595 | 0.5344 | 0.7407 | 0.4073 | 0.5962 | -0.0076 | 14.8 | `fold_0 (0.7105)` |
| 7 | `lin_eval_supervised_ce_unweighted_hflip_elastic_gnoise` | `lin_eval` | 0.6037 +- 0.0281 | 0.4631 | 0.4463 | 0.9546 | 0.0460 | 0.5925 | -0.0112 | 16.8 | `fold_1 (0.6435)` |
| 8 | `finetune_supervised_ce_weighted_sampling_unweighted` | `finetune` | 0.5998 +- 0.0894 | 0.5320 | 0.5006 | 0.8046 | 0.2634 | 0.6484 | +0.0486 | 8.0 | `fold_0 (0.7407)` |
| 9 | `lin_eval_supervised_ce_class_weighted` | `lin_eval` | 0.5979 +- 0.0564 | 0.5335 | 0.5112 | 0.7690 | 0.3311 | 0.6108 | +0.0130 | 8.6 | `fold_0 (0.6729)` |
| 10 | `lin_eval_supervised_focal_g1p5_unweighted` | `lin_eval` | 0.5902 +- 0.0519 | 0.5583 | 0.5360 | 0.7056 | 0.4432 | 0.5860 | -0.0042 | 17.6 | `fold_0 (0.6561)` |
| 11 | `lin_eval_supervised_ce_unweighted_mixup_0p8` | `lin_eval` | 0.5896 +- 0.0195 | 0.5523 | 0.5315 | 0.7127 | 0.4256 | 0.5909 | +0.0013 | 11.6 | `fold_3 (0.6165)` |
| 12 | `lin_eval_supervised_ce_weighted_sampling_unweighted` | `lin_eval` | 0.5884 +- 0.0575 | 0.5343 | 0.5204 | 0.7416 | 0.3685 | 0.5913 | +0.0029 | 8.0 | `fold_0 (0.6885)` |
| 13 | `lin_eval_supervised_ce_unweighted` | `lin_eval` | 0.5786 +- 0.0354 | 0.5515 | 0.5335 | 0.6860 | 0.4541 | 0.5868 | +0.0082 | 13.8 | `fold_1 (0.6265)` |
| 14 | `finetune_supervised_ce_unweighted_hflip` | `finetune` | 0.5782 +- 0.1043 | 0.5415 | 0.5171 | 0.7257 | 0.3637 | 0.6637 | +0.0854 | 4.2 | `fold_0 (0.7557)` |
| 15 | `lin_eval_supervised_ce_unweighted_mixup_0p2` | `lin_eval` | 0.5722 +- 0.0336 | 0.5520 | 0.5380 | 0.6681 | 0.4681 | 0.5982 | +0.0261 | 14.8 | `fold_0 (0.6012)` |
| 16 | `finetune_supervised_focal_g2p0_unweighted` | `finetune` | 0.5721 +- 0.0937 | 0.5356 | 0.5178 | 0.7182 | 0.3638 | 0.6654 | +0.0933 | 14.6 | `fold_0 (0.7322)` |
| 17 | `lin_eval_supervised_ce_unweighted_hflip` | `lin_eval` | 0.5676 +- 0.0574 | 0.5515 | 0.5293 | 0.6662 | 0.4550 | 0.5894 | +0.0218 | 12.2 | `fold_1 (0.6487)` |
| 18 | `lin_eval_supervised_ce_unweighted_hflip_brightness_contrast` | `lin_eval` | 0.5671 +- 0.0980 | 0.5331 | 0.5063 | 0.6891 | 0.3994 | 0.5805 | +0.0134 | 10.8 | `fold_1 (0.6609)` |
| 19 | `finetune_supervised_ce_unweighted_hflip_brightness_contrast` | `finetune` | 0.5605 +- 0.0989 | 0.5320 | 0.5075 | 0.6781 | 0.3780 | 0.6405 | +0.0800 | 9.4 | `fold_0 (0.7322)` |

## Best Candidates Depending on What You Care About

- Best overall mean test binary F1: `finetune_supervised_ce_unweighted_hflip_elastic_gnoise`
- Best balanced finetune choice: `finetune_supervised_ce_class_weighted`
- Best test accuracy and precision: `finetune_supervised_ce_unweighted_mixup_0p8`
- Best lin-eval choice: `lin_eval_supervised_focal_g2p0_unweighted`
- Most stable across folds: `lin_eval_supervised_ce_unweighted_mixup_0p8`

## Where To Inspect Individual Runs

Use any experiment row above, then open:

- `<experiment>/fold_<k>/<run_name>/test_summary.json`
- `<experiment>/fold_<k>/<run_name>/val_summary.json`
- `<experiment>/fold_<k>/<run_name>/test_zone_metrics.csv`
- `<experiment>/fold_<k>/<run_name>/history.png`

Example:

`finetune_supervised_ce_class_weighted/fold_0/finetune_resnet50_pretraining_supervised_CE_loss_batch_24_lr_0.001_decay_0.01_10_10_1_1e-05_epochs_100_zones10x2_seed_0/`

That run contains the per-split JSON summaries, per-zone CSV metrics, saved checkpoint, and training history plot for that fold.
