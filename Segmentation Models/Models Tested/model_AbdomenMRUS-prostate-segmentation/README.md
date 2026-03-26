# Prostate158 Evaluation Workspace

## Overview

This workspace contains a complete local evaluation of the **DIAGNijmegen AbdomenMRUS Prostate Segmentation** model on the **Prostate158** public test subset.

The evaluated task is:

- **Whole-gland prostate segmentation**

The final deliverables produced in this workspace are:

- a prepared inference workspace for all 19 Prostate158 test cases
- model predictions for all 19 cases
- provided labels preserved separately
- binarized whole-gland reference labels for both readers
- 3D Slicer-ready segmentation files
- Dice score results against both readers
- helper scripts used to prepare data, run inference, calculate Dice, and export Slicer segmentations

## Official Model Information

### Model name

**Prostate Segmentation in MRI**

### Official links

- Grand Challenge algorithm page: <https://grand-challenge.org/algorithms/prostate-segmentation/>
- Source code repository: <https://github.com/DIAGNijmegen/AbdomenMRUS-prostate-segmentation>

### What the model does

This model predicts a **binary whole-prostate gland segmentation mask** from **biparametric prostate MRI (bpMRI)**.

It is intended for:

- whole-gland prostate segmentation
- prostate volume estimation support
- research use

It is **not** a tumor segmentation model and **not** a zonal segmentation model.

### Model architecture and training

According to the official model documentation:

- the model is an **ensemble of 5 nnU-Net models**
- training used **Cross-Entropy + Focal Loss**
- it was trained on **438 bpMRI scans** with manual prostate gland segmentations
- the training cohort included:
  - **299 cases from Radboudumc**
  - **139 cases from Prostate158**
- the reported cross-validation performance is:
  - **Dice similarity coefficient: 0.8968 +- 0.0547**
  - **Jaccard index: 0.8169 +- 0.0820**

### Official input requirement

The model requires **three aligned MRI inputs** per case:

- **T2W**
- **ADC**
- **HBV**

The expected Docker-style input structure is:

```text
input/
\-- images/
    +-- transverse-t2-prostate-mri/
    |   \-- <case>_t2w.mha
    +-- transverse-adc-prostate-mri/
    |   \-- <case>_adc.mha
    \-- transverse-hbv-prostate-mri/
        \-- <case>_hbv.mha
```

### Official output

The model writes one output mask per case:

```text
output/
\-- images/
    \-- transverse-whole-prostate-mri/
        \-- prostate_gland.mha
```

This output is a **binary whole-gland mask**.

## Important Usage Limitations

Based on the official model page and repository documentation, this model should be treated as:

- a **bpMRI model**, not a general prostate MRI model
- a model intended for **aligned T2W + ADC + HBV**
- a model developed primarily on **Siemens scanner data**
- a **research-use** model

It is **not officially validated** for:

- T1-weighted MRI
- single-sequence-only inference
- unrelated MRI domains
- severe inter-sequence misalignment
- previously treated prostates
- generalized clinical deployment without local validation

## Dataset Used in This Workspace

### Dataset

The evaluated dataset is the **Prostate158** public test release.

### References

- Prostate158 paper: <https://doi.org/10.1016/j.compbiomed.2022.105817>
- Prostate158 public data release: <https://doi.org/10.5281/zenodo.6592345>

### Cases used

This workspace contains **19 test cases**:

- `001` through `019`

### Original files provided by the dataset

Each case originally contains:

- `t2.nii.gz`
- `adc.nii.gz`
- `dwi.nii.gz`
- `t2_anatomy_reader1.nii.gz`
- `t2_anatomy_reader2.nii.gz`
- `t2_tumor_reader1.nii.gz`
- `adc_tumor_reader1.nii.gz`
- `adc_tumor_reader2.nii.gz`

### Label meaning

The relevant provided anatomy labels are:

- `t2_anatomy_reader1`
- `t2_anatomy_reader2`

These anatomy labels are multi-class and contain values:

- `0` = background
- `1` and `2` = anatomical prostate regions

For whole-gland evaluation, these were converted to **binary whole-gland masks** using:

- `mask > 0`

The tumor labels were preserved, but they were **not used for Dice scoring**, because the evaluated model predicts only the whole gland.

## How the Data Were Adapted for This Model

The Prostate158 public test data do not provide a file explicitly named `hbv.nii.gz`. Instead, they provide:

- `dwi.nii.gz`

To match the model's required three-input interface, the prepared input workspace maps:

- `t2.nii.gz` -> T2W input
- `adc.nii.gz` -> ADC input
- `dwi.nii.gz` -> HBV input slot

This mapping was used only to satisfy the model interface for local inference in this workspace. It should not be interpreted as official evidence that arbitrary diffusion inputs are validated outside the documented model setting.

## Root Workspace Structure

This is the current top-level structure of the workspace:

```text
model1/
+-- modell_AbdomenMRUS-prostate-segmentation/
+-- model_AbdomenMRUS-prostate-segmentation/
+-- prostate158_model_workspace/
+-- prostate158_test/
\-- README.md
```

### Meaning of the top-level folders

- `modell_AbdomenMRUS-prostate-segmentation/`
  - working model copy used for the preparation, inference, Dice, and export scripts
- `model_AbdomenMRUS-prostate-segmentation/`
  - second copy of the same model repository kept in the workspace
- `prostate158_test/`
  - original Prostate158 test data as provided
- `prostate158_model_workspace/`
  - organized evaluation workspace produced for this project

## Which Model Copy Was Used

There are two nearly identical model folders in the root:

- `modell_AbdomenMRUS-prostate-segmentation`
- `model_AbdomenMRUS-prostate-segmentation`

The evaluation used the **working copy**:

- `modell_AbdomenMRUS-prostate-segmentation/AbdomenMRUS-prostate-segmentation`

This copy was used because it already contained local testing artifacts and was the safer place to add preparation and evaluation scripts.

## Prepared Evaluation Workspace Structure

The folder `prostate158_model_workspace/` contains the organized inputs, outputs, labels, Slicer exports, and score files.

```text
prostate158_model_workspace/
+-- dice_scores.csv
+-- manifest.csv
+-- slicer_segmentations_manifest.csv
+-- inputs/
|   +-- 001/
|   +-- 002/
|   +-- ...
|   \-- 019/
+-- outputs/
|   +-- 001/
|   +-- 002/
|   +-- ...
|   \-- 019/
+-- labels/
|   +-- model_predictions/
|   +-- provided_original/
|   \-- provided_whole_gland/
\-- slicer_segmentations/
    +-- 001/
    +-- 002/
    +-- ...
    \-- 019/
```

## Detailed Data Structure

### Prepared inputs

Each case under `inputs/` has the format:

```text
inputs/<case>/images/
+-- transverse-t2-prostate-mri/
|   \-- <case>_t2w.mha
+-- transverse-adc-prostate-mri/
|   \-- <case>_adc.mha
\-- transverse-hbv-prostate-mri/
    \-- <case>_hbv.mha
```

These are the files used as the actual model inputs.

### Model outputs

Each case under `outputs/` has the format:

```text
outputs/<case>/images/transverse-whole-prostate-mri/prostate_gland.mha
```

These are the direct per-case inference outputs from the model container.

### Preserved provided labels

The original labels from the dataset were preserved here:

```text
labels/provided_original/<case>/
+-- t2_anatomy_reader1.nii.gz
+-- t2_anatomy_reader2.nii.gz
+-- t2_tumor_reader1.nii.gz
+-- adc_tumor_reader1.nii.gz
\-- adc_tumor_reader2.nii.gz
```

### Whole-gland references used for Dice scoring

The anatomy labels were binarized to whole-gland masks and written here:

```text
labels/provided_whole_gland/<case>/
+-- <case>_reader1_prostate_gland.mha
\-- <case>_reader2_prostate_gland.mha
```

These are the reference masks used for final Dice scoring.

### Collected model predictions

The model outputs were also collected into a single folder for easier access:

```text
labels/model_predictions/
+-- 001_prostate_gland.mha
+-- 002_prostate_gland.mha
+-- ...
\-- 019_prostate_gland.mha
```

These files are the same model masks as the per-case output masks, just copied into one place.

### 3D Slicer-ready segmentation exports

Each case has direct-import `.seg.nrrd` files here:

```text
slicer_segmentations/<case>/
+-- <case>_model_prediction.seg.nrrd
+-- <case>_reader1.seg.nrrd
\-- <case>_reader2.seg.nrrd
```

These were created so that 3D Slicer loads them as **segmentations**, not as plain volumes.

## Files That Matter Most

If only the key outputs need to be checked, use these files:

- final score table:
  - `prostate158_model_workspace/dice_scores.csv`
- per-case input/output manifest:
  - `prostate158_model_workspace/manifest.csv`
- per-case Slicer manifest:
  - `prostate158_model_workspace/slicer_segmentations_manifest.csv`
- direct-import Slicer segmentations:
  - `prostate158_model_workspace/slicer_segmentations/`

## Quantitative Results

### Final dataset-level results

The final Dice scores are:

- **Mean Dice vs reader 1:** `0.917437`
- **Mean Dice vs reader 2:** `0.902702`
- **Mean Dice across both readers:** `0.910070`
- **Standard deviation across both-reader mean:** `0.019989`

### Per-case Dice scores

| Case | Dice vs Reader 1 | Dice vs Reader 2 | Mean Dice |
|---|---:|---:|---:|
| 001 | 0.887849 | 0.862919 | 0.875384 |
| 002 | 0.936434 | 0.911647 | 0.924041 |
| 003 | 0.934623 | 0.924725 | 0.929674 |
| 004 | 0.911736 | 0.907722 | 0.909729 |
| 005 | 0.938705 | 0.929769 | 0.934237 |
| 006 | 0.937451 | 0.937686 | 0.937569 |
| 007 | 0.919771 | 0.907988 | 0.913879 |
| 008 | 0.922208 | 0.893018 | 0.907613 |
| 009 | 0.916447 | 0.869973 | 0.893210 |
| 010 | 0.883255 | 0.895653 | 0.889454 |
| 011 | 0.934122 | 0.927360 | 0.930741 |
| 012 | 0.928180 | 0.925739 | 0.926959 |
| 013 | 0.897648 | 0.907047 | 0.902348 |
| 014 | 0.925016 | 0.900903 | 0.912959 |
| 015 | 0.913905 | 0.925671 | 0.919788 |
| 016 | 0.919881 | 0.827593 | 0.873737 |
| 017 | 0.928022 | 0.884863 | 0.906443 |
| 018 | 0.870039 | 0.880736 | 0.875388 |
| 019 | 0.926005 | 0.930334 | 0.928170 |

## Validation and Integrity Checks Performed

The following checks were completed in this workspace:

- all 19 cases were found and prepared successfully
- all required input modalities were present for every case
- geometry matched within each case across the input modalities and labels
- the model output masks were verified to be binary
- the model output masks were verified to align with the prepared T2 images
- the binarized whole-gland references were verified to align with the prepared T2 images
- the `.seg.nrrd` exports for 3D Slicer were verified to preserve:
  - voxel arrays
  - size
  - spacing
  - origin
  - direction

### Special note

One original label file had an unusual storage type:

- case `019`, `t2_anatomy_reader1.nii.gz`

This file was stored as `float32`, but its values were still valid for anatomy labeling:

- `{0, 1, 2}`

It was therefore still valid for binarization and evaluation.

## 3D Slicer Usage

For direct review in 3D Slicer, the recommended files for each case are:

- background volume:
  - `inputs/<case>/images/transverse-t2-prostate-mri/<case>_t2w.mha`
- model segmentation:
  - `slicer_segmentations/<case>/<case>_model_prediction.seg.nrrd`
- reader 1 segmentation:
  - `slicer_segmentations/<case>/<case>_reader1.seg.nrrd`
- reader 2 segmentation:
  - `slicer_segmentations/<case>/<case>_reader2.seg.nrrd`

Example for case `001`:

```text
prostate158_model_workspace/
+-- inputs/001/images/transverse-t2-prostate-mri/001_t2w.mha
\-- slicer_segmentations/001/
    +-- 001_model_prediction.seg.nrrd
    +-- 001_reader1.seg.nrrd
    \-- 001_reader2.seg.nrrd
```

Color convention used in the Slicer exports:

- **reader 1** = green
- **reader 2** = yellow
- **model prediction** = red

## Scripts Added for This Evaluation

The following helper scripts were added under the working model copy:

```text
modell_AbdomenMRUS-prostate-segmentation/
\-- AbdomenMRUS-prostate-segmentation/
    \-- local_test/
        +-- prepare_prostate158_dataset.py
        +-- run_prostate158_inference.ps1
        +-- calculate_prostate158_dice.py
        \-- create_slicer_seg_nrrd.py
```

### Script purposes

- `prepare_prostate158_dataset.py`
  - reorganizes Prostate158 into the required model input/output/label workspace
- `run_prostate158_inference.ps1`
  - prepares the workspace and runs batched Docker inference, using GPU if available
- `calculate_prostate158_dice.py`
  - computes Dice scores against both binarized reader references
- `create_slicer_seg_nrrd.py`
  - converts masks into Slicer-native `.seg.nrrd` files without changing voxel data or geometry

## Inference Environment Notes

### Runtime

Inference was executed with the Docker image:

- `joeranbosma/picai_prostate_segmentation_processor:latest`

### GPU usage

GPU inference was tested and confirmed available in Docker on this workstation.

The final 19-case batch inference was run successfully using GPU.

## Practical Conclusions

- the model works as a **whole-gland bpMRI segmentation model**
- this workspace successfully adapted the Prostate158 test set to the model's required three-input format
- the final results are strong for same-domain whole-gland segmentation
- the most appropriate single-number summary from this workspace is:
  - **mean Dice across both readers = 0.910070**

## Recommended Citation / Reference Section for a Report

If this workspace is summarized in a report, the most important references are:

- Grand Challenge model page: <https://grand-challenge.org/algorithms/prostate-segmentation/>
- GitHub repository: <https://github.com/DIAGNijmegen/AbdomenMRUS-prostate-segmentation>
- Prostate158 paper: <https://doi.org/10.1016/j.compbiomed.2022.105817>
- Prostate158 data release: <https://doi.org/10.5281/zenodo.6592345>

## Minimal Checklist

If someone opens this workspace later and only wants the essentials, this is the shortest path:

1. Read `prostate158_model_workspace/dice_scores.csv`
2. Open `prostate158_model_workspace/slicer_segmentations_manifest.csv`
3. Load one case in 3D Slicer using:
   - `inputs/<case>/.../<case>_t2w.mha`
   - `slicer_segmentations/<case>/<case>_model_prediction.seg.nrrd`
   - `slicer_segmentations/<case>/<case>_reader1.seg.nrrd`
   - `slicer_segmentations/<case>/<case>_reader2.seg.nrrd`
