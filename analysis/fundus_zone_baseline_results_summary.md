# Corrected ResNet50 Baseline Results Summary

## Scope

- Experiment root: `output_fundus_zone_baseline/`
- Architectures in this report: corrected full-image `ResNet50` baseline only
- Variants compared:
  - `ce_class_weighted`
  - `focal_g1p5_class_weighted`
  - `ce_label_smoothing_0p05_class_weighted`
- Sweep size: `5 folds x 3 seeds = 15 runs per variant`, `45 runs total`
- Primary endpoint: mean zone binary F1
- Secondary guardrail: mean zone accuracy
- Derived clinical endpoint: any-positive visit metrics

## Test Summary

| Variant | Mean zone F1 | Mean zone acc | Any-positive visit F1 | Any-positive visit acc | Any-positive visit specificity | Mean best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `focal_g1p5_class_weighted` | `0.6216 +- 0.0708` | `0.5520 +- 0.0620` | `0.8056 +- 0.0541` | `0.6898 +- 0.0719` | `0.1392 +- 0.1537` | `7.8` |
| `ce_label_smoothing_0p05_class_weighted` | `0.6039 +- 0.0998` | `0.5524 +- 0.0836` | `0.7892 +- 0.0657` | `0.6755 +- 0.0735` | `0.1673 +- 0.2113` | `9.1` |
| `ce_class_weighted` | `0.5911 +- 0.0763` | `0.5601 +- 0.0869` | `0.7710 +- 0.0687` | `0.6636 +- 0.0746` | `0.3195 +- 0.2868` | `8.0` |

## Promotion Decision Against The Agreed Criteria

Reference baseline: `ce_class_weighted`

| Variant | Delta mean zone F1 | Delta mean zone acc | Delta any-positive specificity | Promote? |
| --- | ---: | ---: | ---: | --- |
| `ce_label_smoothing_0p05_class_weighted` | `+0.0128` | `-0.0077` | `-0.1522` | `No` |
| `focal_g1p5_class_weighted` | `+0.0304` | `-0.0081` | `-0.1804` | `No` |

Neither candidate clears the full promotion rule:

- both improve mean zone F1 by more than `0.01`
- both fail the mean zone accuracy guardrail of no worse than `-0.005`
- both also collapse derived visit-level specificity far beyond the allowed `-0.05`

Conclusion: **the corrected CE class-weighted baseline remains the safest reference model for now**.

## Best Single Run

Best single test run by mean zone F1:

- Variant: `ce_label_smoothing_0p05_class_weighted`
- Fold / seed: `fold_0 / seed_2`
- Mean zone F1: `0.7589`
- Mean zone accuracy: `0.6374`
- Any-positive visit F1: `0.8662`
- Any-positive visit specificity: `0.0870`

This is useful as an upper-bound example, but it is not the right model to promote because its specificity is still very poor and fold-level variation is substantial.

## Main Insights

### 1. The corrected baseline is stronger than the old impression of the task

Even the corrected full-image baseline is reasonably competitive at the zone level:

- `ce_class_weighted` reaches `0.5911` mean test zone F1
- `focal_g1p5_class_weighted` reaches `0.6216`
- `label smoothing` reaches `0.6039`

So the zone-first framing is learnable with the existing dataset, but the operating point is still too recall-heavy for the derived clinical endpoint.

### 2. Focal loss gives the best raw F1, but mainly by pushing recall harder

Compared with `ce_class_weighted`, focal loss:

- raises mean zone F1 by `+0.0304`
- raises any-positive visit F1 by `+0.0346`
- drops mean zone accuracy by `-0.0081`
- drops derived visit specificity by `-0.1804`

Interpretation:

- focal improves sensitivity / recall-oriented detection
- the gain is bought by a meaningful increase in false positives
- that is especially risky for a medical screening endpoint where specificity still matters

### 3. Label smoothing helps some zones, but does not solve the clinical operating point

Compared with `ce_class_weighted`, label smoothing:

- raises mean zone F1 by `+0.0128`
- slightly lowers mean zone accuracy by `-0.0077`
- drops any-positive specificity by `-0.1522`

It looks like a softer version of the same tradeoff:

- modest F1 improvement
- weaker specificity
- still not acceptable as a promoted replacement under the agreed rules

### 4. The derived any-positive visit endpoint is still too non-specific

Mean any-positive visit specificity:

- `ce_class_weighted`: `0.3195`
- `ce_label_smoothing_0p05_class_weighted`: `0.1673`
- `focal_g1p5_class_weighted`: `0.1392`

That is the clearest warning sign in this experiment. The models are good at finding positives somewhere, but too often at the cost of overcalling the visit as positive.

### 5. Fold effects are large, so architecture changes should be judged across all folds, not by the best run

Mean test zone F1 by fold:

- `ce_class_weighted`: `fold_0 0.7037`, `fold_4 0.5203`
- `ce_label_smoothing_0p05_class_weighted`: `fold_0 0.7531`, `fold_4 0.5632`
- `focal_g1p5_class_weighted`: `fold_0 0.7336`, `fold_4 0.5469`

This says:

- `fold_0` is consistently easier
- `fold_4` is consistently hardest
- we should avoid overinterpreting any one fold / seed result

### 6. Focal loss appears slightly better calibrated from validation to test than CE

Mean validation-to-test gap in mean zone F1:

- `ce_class_weighted`: `+0.0763`
- `ce_label_smoothing_0p05_class_weighted`: `+0.0384`
- `focal_g1p5_class_weighted`: `+0.0184`

This is one positive signal for focal: it generalizes a bit more consistently, even though its final operating point is too aggressive on positives.

## Zone-Level Insights

### Stronger zones in the baseline

Best zones for `ce_class_weighted` by mean test zone F1:

- `Zone 8`: `0.6364`
- `Zone 7`: `0.6017`
- `Zone 1`: `0.6009`

### Harder zones in the baseline

Weakest zones for `ce_class_weighted` by mean test zone F1:

- `Zone 5`: `0.5706`
- `Zone 10`: `0.5765`
- `Zone 3`: `0.5771`

### Where label smoothing helped most

Largest mean zone F1 gains vs baseline:

- `Zone 9`: `+0.0539`
- `Zone 6`: `+0.0413`
- `Zone 7`: `+0.0333`

But it hurt:

- `Zone 10`: `-0.0345`
- `Zone 3`: `-0.0013`

### Where focal helped most

Largest mean zone F1 gains vs baseline:

- `Zone 10`: `+0.0576`
- `Zone 5`: `+0.0560`
- `Zone 6`: `+0.0553`
- `Zone 9`: `+0.0490`

But focal also reduced specificity in every zone, with especially large drops in:

- `Zone 4`: `-0.2132`
- `Zone 5`: `-0.1777`
- `Zone 3`: `-0.1633`
- `Zone 10`: `-0.1449`

Interpretation: focal is rescuing the harder positive zones, but mostly by shifting the classifier toward more positive calls overall.

## Recommended Next Step

Based on this experiment alone:

- keep `ce_class_weighted` as the corrected full-image reference baseline
- do **not** promote focal or label smoothing as the new default full-image model
- move next to the planned `zone_crops_shared` and `hybrid_global_plus_zone` experiments
- when those finish, compare them against this CE class-weighted reference using the same promotion rule

## Files Used

- `output_fundus_zone_baseline/test_experiment_ranking.csv`
- `output_fundus_zone_baseline/test_run_ranking.csv`
- `output_fundus_zone_baseline/val_experiment_ranking.csv`
- `output_fundus_zone_baseline/val_run_ranking.csv`
- per-run `test_summary.json`
- per-run `val_summary.json`
- per-run `test_zone_metrics.csv`
