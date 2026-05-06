# FA Path Auto-Fix Report

- Source spreadsheet: `UWFAFP_Annotations_Mo_4.5.2026 (Uveitis).xlsx`
- Unique FA paths reviewed: `741`
- Resolved after mapping correction: `740`
- Unresolved after mapping correction: `1`

## Correction Counts

- `exact_exists`: `495`
- `swap_0000_to_0001`: `240`
- `extension_to_png`: `2`
- `swap_0001_to_0000`: `1`
- `truncated_to_0001`: `1`
- `unresolved`: `1`
- `single_same_eye_fa`: `1`

## Completed Rerun Summary

- Recoverable corrected paths targeted: `243`
- Successfully recovered: `240`
- Remaining rerun errors: `3`
- Real extractor errors after correction: `3`
- Prior disk-space errors now remaining: `0`

## Remaining Unresolved Path

- `Patient083/20250204/Patient083_20250204_OD_FA_0001.png`: available_fa_files=['Patient083_20250204_OS_FA_0001.png', 'Patient083_20250204_OS_FA_0001_masks.npy']

## Remaining Extractor Failures After Correction

- `Could not detect zone circles from yellow overlay.`: `1`
- `Could not isolate the optic-disc circle.`: `2`
