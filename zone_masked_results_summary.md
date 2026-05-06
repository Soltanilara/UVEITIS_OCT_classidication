# Zone-Masked Results Summary

This note compares the results in:

- `output_zone_masked_ce`
- `output_zone_masked_canonical`

## What was compared

`output_zone_masked_ce` contains 5 runs total:

- 1 run per fold
- CE class-weighted loss
- `batch_4`
- `seed_0`

`output_zone_masked_canonical` contains 45 runs total:

- 3 experiment families
- 5 folds
- 3 seeds per fold
- `batch_24`

Canonical experiment families:

- `ce_class_weighted`
- `ce_label_smoothing_0p05_class_weighted`
- `focal_g1p5_class_weighted`

Because the two directories differ in both batch size and number of seeds, this comparison is directionally useful but not perfectly apples-to-apples.

## Main takeaways

`output_zone_masked_canonical` is stronger than `output_zone_masked_ce` on zone-level `mean_binary_f1` overall.

- `output_zone_masked_ce`: mean test `mean_binary_f1 = 0.5147`
- Best canonical family after validation-based seed selection: `ce_class_weighted` with mean test `mean_binary_f1 = 0.5981`

That improvement comes with a tradeoff. The canonical runs generally push recall much higher, but specificity drops.

- `output_zone_masked_ce`: mean recall `0.6701`, mean specificity `0.3730`
- Best canonical family (`ce_class_weighted`, selected by validation F1): mean recall `0.8100`, mean specificity `0.2660`

So the canonical runs are better at finding positives, but they are also more prone to false positives.

At the visit-level `any_positive_visit` task, the story is flatter:

- `output_zone_masked_ce`: mean visit F1 `0.8286`, mean PR AUC `0.7369`
- Canonical `ce_class_weighted` selected by validation F1: mean visit F1 `0.8301`, mean PR AUC `0.7943`

Visit-level F1 barely changes, but canonical improves ranking quality more clearly through PR AUC.

## Aggregate comparison

Test-set averages are shown below.

| Setting | Mean Accuracy | Mean Binary F1 | Precision | Recall | Specificity | Visit F1 | Visit PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `output_zone_masked_ce` | 0.5055 | 0.5147 | 0.4891 | 0.6701 | 0.3730 | 0.8286 | 0.7369 |
| Canonical `ce_class_weighted` all 15 runs | 0.5121 | 0.5797 | 0.4886 | 0.7974 | 0.2687 | 0.8301 | 0.7808 |
| Canonical `ce_class_weighted` selected by best validation seed per fold | 0.5232 | 0.5981 | 0.5014 | 0.8100 | 0.2660 | 0.8301 | 0.7943 |
| Canonical `ce_label_smoothing_0p05_class_weighted` selected by best validation seed per fold | 0.5248 | 0.5736 | 0.4930 | 0.7578 | 0.3212 | 0.8243 | 0.8067 |
| Canonical `focal_g1p5_class_weighted` selected by best validation seed per fold | 0.5012 | 0.5869 | 0.4910 | 0.8160 | 0.2364 | 0.8301 | 0.7627 |

## Best canonical family

Using validation `mean_binary_f1` to pick one seed per fold, the strongest canonical family on zone-level F1 is:

- `ce_class_weighted`

Its selected seeds were:

- `fold_0 -> seed_1`
- `fold_1 -> seed_0`
- `fold_2 -> seed_1`
- `fold_3 -> seed_1`
- `fold_4 -> seed_1`

Its mean test metrics were:

- `mean_accuracy = 0.5232`
- `mean_binary_f1 = 0.5981`
- `mean_precision = 0.5014`
- `mean_recall = 0.8100`
- `mean_specificity = 0.2660`
- `any_positive_visit_f1 = 0.8301`
- `any_positive_visit_pr_auc = 0.7943`

Interpretation:

- If zone-level sensitivity is the priority, this is the best overall choice among the canonical runs.
- If a better precision-specificity balance matters, `ce_label_smoothing_0p05_class_weighted` is worth considering because it gives up some zone F1 but recovers specificity.

## Fold-level comparison

`output_zone_masked_ce` test results by fold:

| Fold | Mean Accuracy | Mean Binary F1 | Precision | Recall | Specificity | Visit PR AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fold_0` | 0.5278 | 0.5545 | 0.6040 | 0.5895 | 0.4529 | 0.7748 |
| `fold_1` | 0.5573 | 0.5202 | 0.5164 | 0.5839 | 0.5207 | 0.7551 |
| `fold_2` | 0.4528 | 0.5430 | 0.4274 | 0.8450 | 0.1461 | 0.5472 |
| `fold_3` | 0.5474 | 0.4305 | 0.4951 | 0.4859 | 0.5630 | 0.8183 |
| `fold_4` | 0.4423 | 0.5252 | 0.4026 | 0.8464 | 0.1826 | 0.7891 |

Canonical `ce_class_weighted` after validation-based seed selection:

| Fold | Seed | Mean Accuracy | Mean Binary F1 | Precision | Recall | Specificity | Visit PR AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fold_0` | `seed_1` | 0.5889 | 0.7016 | 0.6046 | 0.8474 | 0.2424 | 0.8178 |
| `fold_1` | `seed_0` | 0.5326 | 0.6480 | 0.5006 | 0.9219 | 0.1776 | 0.8502 |
| `fold_2` | `seed_1` | 0.4972 | 0.5495 | 0.4566 | 0.7213 | 0.3281 | 0.6045 |
| `fold_3` | `seed_1` | 0.5629 | 0.5284 | 0.5323 | 0.6361 | 0.4821 | 0.8569 |
| `fold_4` | `seed_1` | 0.4346 | 0.5631 | 0.4128 | 0.9234 | 0.0998 | 0.8421 |

Observations:

- `fold_0` shows the biggest gain in zone-level F1 for canonical CE.
- `fold_3` remains relatively modest even after seed selection.
- `fold_4` improves recall substantially but specificity collapses the most.

## Best single runs in canonical

Best single test run in each canonical family by zone-level `mean_binary_f1`:

- `ce_class_weighted`: `fold_0/seed_0`, `mean_binary_f1 = 0.7259`, `mean_accuracy = 0.5733`, `specificity = 0.0091`, `visit PR AUC = 0.8675`
- `ce_label_smoothing_0p05_class_weighted`: `fold_0/seed_1`, `mean_binary_f1 = 0.7060`, `mean_accuracy = 0.5856`, `specificity = 0.1773`, `visit PR AUC = 0.8523`
- `focal_g1p5_class_weighted`: `fold_0/seed_2`, `mean_binary_f1 = 0.6940`, `mean_accuracy = 0.5911`, `specificity = 0.2806`, `visit PR AUC = 0.8462`

These best single runs are strong, but some of them achieve that by almost always predicting positive. For that reason, the fold-wise validation-selected summaries above are the safer headline comparison.

## Bottom line

If the goal is the best zone-level detection performance, `output_zone_masked_canonical/ce_class_weighted` is the strongest overall result among the folders reviewed.

If the goal is a more balanced operating point, `ce_label_smoothing_0p05_class_weighted` looks like the most reasonable compromise:

- lower zone F1 than canonical CE
- better specificity than canonical CE
- strongest visit-level PR AUC among the validation-selected canonical families
