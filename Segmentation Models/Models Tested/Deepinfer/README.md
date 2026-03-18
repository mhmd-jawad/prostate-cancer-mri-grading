# DeepInfer Prostate MRI Evaluation

This workspace contains the full workflow used to evaluate the DeepInfer prostate segmentation model on the PROMISE12 prostate MRI dataset.

## Model

- Docker image: `deepinfer/prostate`
- Model name: `prostate-segmenter`
- Domain: `PROMISE12`
- Intended input: pelvic axial T2-weighted prostate MRI
- Official model page: <https://www.deepinfer.org/models/prostate-segmenter/>
- Official dataset page: <https://promise12.grand-challenge.org/Details/>

## What Was Done

The original dataset was provided as MetaImage pairs (`.mhd` + `.raw`), but the model required `.nrrd` input. The workflow in this folder:

1. Converted the MRI volumes and ground-truth segmentations from `.mhd/.raw` to `.nrrd`
2. Verified that the converted `.nrrd` geometry matched the original `.mhd` geometry
3. Ran DeepInfer segmentation for all 50 PROMISE12 cases
4. Computed Dice score between the predicted masks and the ground-truth masks
5. Generated report files in HTML, PDF, and LaTeX

## Main Scripts

- `convert_mhd_to_nrrd.py`
  Converts `training_data` MetaImage files into:
  - `nrrd_images`
  - `nrrd_labels`

- `run_deepinfer_batch.ps1`
  Runs the DeepInfer Docker container in batch mode over the `.nrrd` MRI cases and writes predictions to the output folder.

- `compute_dice_scores.py`
  Reads ground-truth and predicted `.nrrd` label maps, checks geometry, and computes Dice score for each case plus the dataset mean.

- `compare_slicer_geometry.py`
  Loads files in 3D Slicer and compares geometry to confirm the converted `.nrrd` files match the original images.

## Important Output Files

- `dice_scores_all50.csv`
  Final Dice scores for all 50 cases

- `DeepInfer_PROMISE12_Report.tex`
  Final LaTeX report source

- `DeepInfer_PROMISE12_Report.pdf`
  Generated PDF report

- `DeepInfer_PROMISE12_Report.html`
  Generated HTML report

## Final Result

- Evaluated cases: `Case00` to `Case49`
- Geometry validation: passed
- Mean whole-gland Dice score: `0.848583`

## Notes

- The DeepInfer container accepted GPU passthrough from Docker, but the model runtime inside the container did not expose a usable GPU device. In practice, inference ran on CPU.
- The model should be treated as a T2-weighted MRI model, because the DeepInfer documentation and the PROMISE12 dataset are T2-based.

## Suggested GitHub Contents

Recommended to keep in the repository:

- scripts
- report source and final report
- final Dice CSV
- this README

Recommended to keep out of the repository:

- `training_data/`
- `nrrd_images/`
- `nrrd_labels/`
- `deepinfer_results_first25/`

Those folders contain large medical image files and model outputs and are better stored outside the repo or with Git LFS if needed.
