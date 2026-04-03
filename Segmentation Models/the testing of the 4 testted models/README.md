# Testing of the 4 Tested Models

This folder is intended for the benchmark notebook used to evaluate four fixed prostate MRI segmentation models on the PROMISE12 whole-gland test set.

The notebook:
- downloads the test data and required model assets
- pairs MRI volumes with ground-truth masks
- standardizes predictions into a unified whole-gland format
- reports Dice, Tversky (`alpha=0.3`, `beta=0.7`), and non-intersecting prediction percentage
- compares Dzaridis, DeepInfer, BAMF AIMI, and MONAI `prostate_mri_anatomy`
- exports summary tables, qualitative figures, and a PDF report

Notebook title: `PROMISE12 Whole-Gland Benchmark Notebook`
