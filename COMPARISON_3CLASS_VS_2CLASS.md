# 3-Class vs 2-Class Sweep Comparison

This report compares the ResNet50 3-class sweep in `output_resnet50_sweep_remaining/` against the 2-class sweep in `output_resnet50_sweep_binary/`.

## What Was Compared

- 3-class sweep folder: `output_resnet50_sweep_remaining/`
- 2-class sweep folder: `output_resnet50_sweep_binary/`
- Matched experiments: `19`
- Matched fold pairs: `95`

The two sweeps do not store directly comparable headline metrics:

- The 3-class sweep stores `mean_macro_f1`
- The 2-class sweep stores `mean_binary_f1`

To make a fair comparison, the 3-class predictions were collapsed to binary using:

- `0 -> negative`
- `1 or 2 -> positive`

Then, for both sweeps, binary metrics were recomputed from each run's `test_predictions.csv`:

- per-zone binary accuracy
- per-zone precision
- per-zone recall
- per-zone specificity
- per-zone binary F1

The final score used here is the same style as the binary training script:

- compute binary F1 separately for each of the 10 zones
- average those 10 zone-level F1 values

## Overall Result

There is a clear jump in binary F1 when training directly for the 2-class problem.

| Metric | 3-class sweep collapsed to binary | 2-class sweep | Delta (2-class - 3-class) |
| --- | ---: | ---: | ---: |
| Mean binary F1 | 0.4653 | 0.5944 | +0.1291 |
| Mean accuracy | 0.5644 | 0.5411 | -0.0233 |
| Mean precision | 0.5428 | 0.5175 | -0.0252 |
| Mean recall | 0.4546 | 0.7587 | +0.3042 |
| Mean specificity | 0.6753 | 0.3505 | -0.3248 |

### Interpretation

- The 2-class models recover many more positives, which drives a much higher binary F1.
- That recall gain comes with a large drop in specificity.
- Accuracy stays in the same general range because the gain in true positives is partly offset by more false positives.

## Significance Checks

Using paired comparisons across matched folds:

- Binary F1 paired t-test: `t = 10.2330`, `p = 5.88e-17`
- Accuracy paired t-test: `t = -3.3764`, `p = 0.00107`

Using experiment-level means:

- Binary F1 paired t-test: `t = 6.9441`, `p = 1.73e-06`

These results support a real binary-F1 improvement for the 2-class sweep.

## Win/Loss Count

Across the 95 matched fold pairs:

- 2-class had higher binary F1 in `90` runs
- 3-class had higher binary F1 in `5` runs
- 2-class had higher accuracy in `27` runs
- 3-class had higher accuracy in `68` runs

This reinforces the main pattern:

- 2-class wins strongly on F1
- 3-class more often wins on accuracy

## Protocol-Level Comparison

| Protocol | 3-class binary F1 | 2-class binary F1 | Delta F1 | 3-class acc | 2-class acc | Delta acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `finetune` | 0.4661 | 0.6038 | +0.1376 | 0.5663 | 0.5435 | -0.0228 |
| `lin_eval` | 0.4645 | 0.5859 | +0.1214 | 0.5627 | 0.5389 | -0.0237 |

The binary-F1 gain appears in both protocols, slightly larger in `finetune`.

## Native Metric Reminder

The native stored metrics are still useful within each sweep, but they should not be compared directly across sweeps:

- 3-class native mean metric across matched runs: `mean_macro_f1 = 0.3610`
- 2-class native mean metric across matched runs: `mean_binary_f1 = 0.5944`

That difference is not an apples-to-apples comparison, because the underlying task definitions are different.

## Biggest Binary-F1 Gains by Experiment

| Experiment | 3-class binary F1 | 2-class binary F1 | Delta F1 | Delta acc |
| --- | ---: | ---: | ---: | ---: |
| `lin_eval_supervised_ce_unweighted_hflip_elastic_gnoise` | 0.2032 | 0.6037 | +0.4005 | -0.0191 |
| `finetune_supervised_ce_unweighted_hflip_elastic_gnoise` | 0.3620 | 0.6363 | +0.2744 | -0.0213 |
| `finetune_supervised_ce_unweighted_mixup_0p2` | 0.4672 | 0.6318 | +0.1645 | -0.0196 |
| `finetune_supervised_focal_g1p5_unweighted` | 0.4601 | 0.6045 | +0.1444 | -0.0376 |
| `finetune_supervised_ce_class_weighted` | 0.4911 | 0.6331 | +0.1421 | +0.0337 |
| `lin_eval_supervised_ce_unweighted_mixup_0p8` | 0.4607 | 0.5896 | +0.1289 | -0.0214 |
| `finetune_supervised_ce_weighted_sampling_unweighted` | 0.4791 | 0.5998 | +0.1207 | -0.0133 |
| `finetune_supervised_focal_g2p0_unweighted` | 0.4525 | 0.5721 | +0.1196 | -0.0361 |

## Smallest Binary-F1 Gains by Experiment

Even the weakest matched improvements were still positive on average.

| Experiment | 3-class binary F1 | 2-class binary F1 | Delta F1 | Delta acc |
| --- | ---: | ---: | ---: | ---: |
| `lin_eval_supervised_ce_weighted_sampling_unweighted` | 0.5239 | 0.5884 | +0.0645 | -0.0500 |
| `lin_eval_supervised_ce_unweighted` | 0.5119 | 0.5786 | +0.0667 | -0.0206 |
| `lin_eval_supervised_ce_unweighted_hflip` | 0.4934 | 0.5676 | +0.0742 | -0.0212 |
| `finetune_supervised_ce_unweighted_hflip_brightness_contrast` | 0.4820 | 0.5605 | +0.0785 | -0.0553 |
| `lin_eval_supervised_ce_unweighted_hflip_brightness_contrast` | 0.4828 | 0.5671 | +0.0843 | -0.0340 |
| `lin_eval_supervised_focal_g1p5_unweighted` | 0.5002 | 0.5902 | +0.0900 | -0.0094 |
| `finetune_supervised_ce_unweighted_hflip` | 0.4831 | 0.5782 | +0.0951 | -0.0293 |
| `lin_eval_supervised_ce_class_weighted` | 0.5026 | 0.5979 | +0.0952 | -0.0187 |

## Main Conclusion

If the target task is binary detection, the 2-class sweep is clearly better on binary F1.

If the target metric is plain accuracy, the gain is not there. In fact, the 3-class models collapsed to binary were slightly better on average accuracy.

The most likely reason is that the 2-class models are much more positive-sensitive:

- much higher recall
- much lower specificity
- much better binary F1
- roughly similar, slightly lower accuracy

## Files Used

This comparison was based on:

- `output_resnet50_sweep_remaining/*/fold_*/<run>/test_predictions.csv`
- `output_resnet50_sweep_remaining/*/fold_*/<run>/test_summary.json`
- `output_resnet50_sweep_binary/*/fold_*/<run>/test_predictions.csv`
- `output_resnet50_sweep_binary/*/fold_*/<run>/test_summary.json`
