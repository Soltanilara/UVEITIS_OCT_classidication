| Rank | Experiment | Folds | Test Macro-F1 (mean±std) | Test Acc (mean±std) | Test Loss | Val Macro-F1 | Val Acc | Val Loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | finetune_supervised_ce_unweighted_mixup_0p8 | 5 | 0.4166 ± 0.0539 | 0.5387 ± 0.0413 | 1.0314 | 0.4150 | 0.5689 | 0.9511 |
| 2 | lin_eval_supervised_ce_weighted_sampling_unweighted | 5 | 0.4048 ± 0.0177 | 0.5044 ± 0.0220 | 1.0522 | 0.4095 | 0.5173 | 1.0574 |
| 3 | lin_eval_supervised_ce_unweighted | 5 | 0.3924 ± 0.0339 | 0.4974 ± 0.0524 | 1.0615 | 0.3970 | 0.5182 | 1.0448 |
| 4 | lin_eval_supervised_focal_g2p0_unweighted | 5 | 0.3858 ± 0.0336 | 0.4999 ± 0.0337 | 0.4928 | 0.4032 | 0.5170 | 0.4908 |
| 5 | lin_eval_supervised_ce_unweighted_hflip | 5 | 0.3835 ± 0.0352 | 0.4971 ± 0.0322 | 1.0579 | 0.3875 | 0.5110 | 1.0401 |
| 6 | lin_eval_supervised_ce_unweighted_mixup_0p8 | 5 | 0.3833 ± 0.0213 | 0.5156 ± 0.0203 | 1.0017 | 0.3956 | 0.5459 | 0.9869 |
| 7 | lin_eval_supervised_ce_unweighted_mixup_0p2 | 5 | 0.3793 ± 0.0274 | 0.5078 ± 0.0312 | 1.0275 | 0.4039 | 0.5431 | 1.0074 |
| 8 | lin_eval_supervised_ce_unweighted_hflip_brightness_contrast | 5 | 0.3790 ± 0.0251 | 0.4976 ± 0.0374 | 1.0299 | 0.3933 | 0.5303 | 1.0117 |
| 9 | lin_eval_supervised_focal_g1p5_unweighted | 5 | 0.3759 ± 0.0354 | 0.4878 ± 0.0590 | 0.6282 | 0.4006 | 0.5196 | 0.5867 |
| 10 | finetune_supervised_focal_g2p0_unweighted | 5 | 0.3740 ± 0.0594 | 0.5050 ± 0.0515 | 0.5807 | 0.4157 | 0.5774 | 0.5114 |
| 11 | finetune_supervised_focal_g1p5_unweighted | 5 | 0.3737 ± 0.0504 | 0.5160 ± 0.0512 | 0.6397 | 0.4061 | 0.5801 | 0.5466 |
| 12 | finetune_supervised_ce_unweighted_hflip_brightness_contrast | 5 | 0.3729 ± 0.0701 | 0.5200 ± 0.0403 | 1.1318 | 0.3885 | 0.5547 | 1.0682 |
| 13 | lin_eval_supervised_ce_class_weighted | 5 | 0.3705 ± 0.0309 | 0.4543 ± 0.0464 | 1.1426 | 0.4087 | 0.5206 | 1.1059 |
| 14 | finetune_supervised_ce_unweighted_mixup_0p2 | 5 | 0.3572 ± 0.0597 | 0.4969 ± 0.0602 | 1.1968 | 0.4211 | 0.5764 | 1.1038 |
| 15 | finetune_supervised_ce_unweighted_hflip | 5 | 0.3489 ± 0.0679 | 0.4832 ± 0.0518 | 1.2568 | 0.3962 | 0.5731 | 1.2176 |
| 16 | finetune_supervised_ce_class_weighted | 5 | 0.3457 ± 0.0590 | 0.4289 ± 0.0821 | 1.4587 | 0.4284 | 0.5420 | 1.2052 |
| 17 | finetune_supervised_ce_weighted_sampling_unweighted | 5 | 0.3335 ± 0.0327 | 0.4553 ± 0.0806 | 1.1764 | 0.4083 | 0.5479 | 1.0152 |
| 18 | finetune_supervised_ce_unweighted_hflip_elastic_gnoise | 5 | 0.2809 ± 0.0427 | 0.4382 ± 0.0626 | 1.2941 | 0.3332 | 0.4849 | 1.1571 |
| 19 | lin_eval_supervised_ce_unweighted_hflip_elastic_gnoise | 5 | 0.2004 ± 0.0243 | 0.4292 ± 0.0767 | 2.6268 | 0.2190 | 0.4918 | 2.3431 |

## Scratch Baseline Comparison (folds 0-4)

Scratch baseline is from `output_fold_*_zones` using:
`--protocol scratch --model resnet50 --loss CE --unweighted --image_size 224 --batch_size 24 --num_epochs 100 --earlystop`.

| Baseline | Folds | Test Macro-F1 (mean±std) | Test Acc (mean±std) | Test Loss | Val Macro-F1 | Val Acc | Val Loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| scratch_resnet50_ce_unweighted | 5 | 0.3087 ± 0.0620 | 0.4447 ± 0.0575 | 1.1627 | 0.3502 | 0.5099 | 1.0899 |

### Top Configs vs Scratch (delta = config - scratch)

| Rank | Experiment | Δ Test Macro-F1 | Δ Test Acc | Δ Test Loss |
|---:|---|---:|---:|---:|
| 1 | finetune_supervised_ce_unweighted_mixup_0p8 | +0.1079 | +0.0940 | -0.1313 |
| 2 | lin_eval_supervised_ce_weighted_sampling_unweighted | +0.0960 | +0.0597 | -0.1105 |
| 3 | lin_eval_supervised_ce_unweighted | +0.0836 | +0.0527 | -0.1012 |
| 4 | lin_eval_supervised_focal_g2p0_unweighted | +0.0771 | +0.0552 | -0.6699 |
| 5 | lin_eval_supervised_ce_unweighted_hflip | +0.0747 | +0.0524 | -0.1048 |
