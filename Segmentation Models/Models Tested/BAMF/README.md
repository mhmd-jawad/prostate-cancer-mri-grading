# BAMF/AIMI Prostate MRI Evaluation

This workspace contains the full workflow used to evaluate the BAMF/AIMI prostate MR segmentation pipeline on the PROMISE12 prostate MRI dataset.

## Model

- Repository: `bamf-health/aimi-prostate-mr`
- Inference backend used: `nnU-Net v1`
- Released pretrained task: `Task788_ProstateX`
- Domain: `PROMISE12`
- Intended input: axial T2-weighted prostate MRI
- Official repository: <https://github.com/bamf-health/aimi-prostate-mr>
- Official dataset page: <https://promise12.grand-challenge.org/Details/>

## What Was Done

The PROMISE12 cases were provided as medical image files that needed to be standardized before BAMF inference. The workflow in this folder:

1. Downloaded the PROMISE12 dataset archive from a direct URL
2. Discovered and matched MRI volumes with their ground-truth segmentations
3. Converted the PROMISE12 files to standardized `.nii.gz`
4. Verified image-label consistency, including shape and spacing checks
5. Downloaded and installed the official BAMF/AIMI pretrained prostate MRI model weights
6. Ran BAMF/AIMI inference through the released nnU-Net pipeline
7. Computed Dice score between predicted masks and the PROMISE12 ground-truth masks
8. Generated report-ready figures, CSV metrics, and LaTeX assets

## Main Files

- `bamf_promise12_colab_inference_evaluation.ipynb`  
  Colab notebook for:
  - dataset download
  - preprocessing and NIfTI conversion
  - BAMF/AIMI model setup
  - inference
  - Dice evaluation
  - report asset generation

- `per_case_dice_scores.csv`  
  Per-case Dice scores for the evaluated PROMISE12 cases

- `summary_metrics.csv`  
  Summary metrics including:
  - number of evaluated cases
  - mean Dice
  - standard deviation
  - minimum Dice
  - maximum Dice

- `report_assets/quantitative_results_table.tex`  
  LaTeX table containing the per-case whole-gland Dice results

- `report_assets/report.tex`  
  Final LaTeX report source

## Important Output Files

- `per_case_dice_scores.csv`
  Final per-case Dice scores

- `summary_metrics.csv`
  Final summary statistics

- `report_assets/quantitative_results_table.tex`
  LaTeX-ready Dice table

- `report_assets/case00_compare.png`
- `report_assets/case09_compare.png`
- `report_assets/case23_compare.png`
  Representative qualitative comparison figures

- `VIPP Report.pdf`
  Final PDF report

## Final Result

- Evaluated cases: `case00` to `case29`
- Geometry validation: passed
- Mean whole-gland Dice score: `0.916301`

## Notes

- The BAMF/AIMI released pipeline was run through the associated nnU-Net inference workflow rather than a custom reimplementation.
- A PyTorch version adjustment was required in Colab to make the older nnU-Net v1 checkpoint loading compatible with the current runtime.
- PROMISE12 files were converted to `.nii.gz` before inference so the workflow remained stable and reproducible.
- This evaluation was run on the labeled PROMISE12 cases available in the downloaded test-set package used for the project.

## Suggested GitHub Contents

Recommended to keep in the repository:

- notebook
- final CSV metrics
- report source and final report
- qualitative report figures
- this README

Recommended to keep out of the repository:

- raw PROMISE12 archives
- extracted medical image folders
- standardized `.nii.gz` volumes
- downloaded model zip files
- temporary nnU-Net prediction folders

Those files are large and are better stored outside the repo or with Git LFS if needed.
