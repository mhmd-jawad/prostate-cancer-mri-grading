import glob, os
import SimpleITK as sitk

raw_dir = os.path.abspath(os.path.join("..", "PROMISE12_raw"))
out_dir = os.path.abspath(os.path.join("..", "PROMISE12_nifti"))
os.makedirs(out_dir, exist_ok=True)

mhd_files = sorted(glob.glob(os.path.join(raw_dir, "Case*.mhd")))
if not mhd_files:
    raise SystemExit(f"No Case*.mhd found in: {raw_dir}")

for f in mhd_files:
    img = sitk.ReadImage(f)
    out = os.path.join(out_dir, os.path.basename(f).replace(".mhd", ".nii.gz"))
    sitk.WriteImage(img, out)
    print("Wrote:", out)