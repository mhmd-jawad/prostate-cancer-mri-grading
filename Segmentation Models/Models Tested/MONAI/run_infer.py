import argparse
import statistics
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import torch
from monai.data import DataLoader, Dataset, decollate_batch, set_track_meta
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    SaveImaged,
    ScaleIntensityd,
    Spacingd,
)

try:
    import SimpleITK as sitk  # noqa: N813
except Exception:  # pragma: no cover - only needed for MHD/cropping paths
    sitk = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress bar
    tqdm = None


warnings.filterwarnings("ignore")


LABEL_SUFFIX = "_segmentation"


@dataclass
class CaseRecord:
    case_id: str
    image: Path
    label: Optional[Path] = None
    source_image: Optional[Path] = None
    source_label: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.source_image is None:
            self.source_image = self.image
        if self.source_label is None:
            self.source_label = self.label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prostate MRI zonal segmentation inference (MONAI bundle)."
    )
    parser.add_argument(
        "--input",
        required=False,
        help=(
            "Input file or directory. Supported: .nii, .nii.gz, .mhd, .mha. "
            "If omitted, runs on the first 25 PROMISE12_nifti images."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for segmentation outputs.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model weights (.pt) or TorchScript (.ts).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--torchscript",
        action="store_true",
        help="Force loading the model as TorchScript.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (CUDA only).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--roi-size",
        default="96,96,96",
        help="Sliding window ROI size, comma-separated.",
    )
    parser.add_argument(
        "--sw-batch-size",
        type=int,
        default=4,
        help="Sliding window batch size.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Sliding window overlap.",
    )
    parser.add_argument(
        "--output-postfix",
        default="pred",
        help="Postfix for saved segmentations.",
    )
    parser.add_argument(
        "--center-crop",
        default=None,
        help=(
            "Optional center-crop margin for X/Y axes. "
            "Provide a float (percentage per side) or int (voxels per side)."
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary converted/cropped files.",
    )
    return parser.parse_args()


def _parse_roi_size(text: str) -> Tuple[int, int, int]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("roi-size must have 3 comma-separated integers")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _flatten(t: Iterable[Iterable[Union[int, float]]]) -> List[Union[int, float]]:
    return [item for sublist in t for item in sublist]


def _path_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _case_id_from_path(path: Path) -> str:
    stem = _path_stem(path)
    if stem.lower().endswith(LABEL_SUFFIX):
        return stem[: -len(LABEL_SUFFIX)]
    return stem


def _is_label_path(path: Path) -> bool:
    return _path_stem(path).lower().endswith(LABEL_SUFFIX)


def _find_label_for_image(image_path: Path) -> Optional[Path]:
    case_id = _case_id_from_path(image_path)
    for ext in (".nii.gz", ".nii", ".mhd", ".mha"):
        candidate = image_path.with_name(f"{case_id}{LABEL_SUFFIX}{ext}")
        if candidate.exists():
            return candidate
    return None


def _as_text(value: object) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Expected a non-empty case identifier.")
        return str(value[0])
    return str(value)


def _center_crop_sitk(
    image: "sitk.Image", margin: Union[int, float]
) -> "sitk.Image":
    if isinstance(margin, (list, tuple)):
        if len(margin) != 3:
            raise ValueError("margin list must have length 3")
    else:
        if not isinstance(margin, (int, float)):
            raise ValueError("margin must be int or float")
        margin = [margin, margin, 0.0]  # crop X/Y only, keep Z

    margin = [m if isinstance(m, (tuple, list)) else [m, m] for m in margin]
    old_size = image.GetSize()

    if all(isinstance(m, float) for m in _flatten(margin)):
        if not all(0 <= m < 0.5 for m in _flatten(margin)):
            raise ValueError("float margins must be between 0 and 0.5")
        to_crop = [[int(sz * _m) for _m in m] for sz, m in zip(old_size, margin)]
    elif all(isinstance(m, int) for m in _flatten(margin)):
        to_crop = margin
    else:
        raise ValueError("margin types must be all int or all float")

    new_size = [sz - sum(c) for sz, c in zip(old_size, to_crop)]
    new_origin = image.TransformIndexToPhysicalPoint([c[0] for c in to_crop])

    ref_image = sitk.Image(new_size, image.GetPixelIDValue())
    ref_image.SetSpacing(image.GetSpacing())
    ref_image.SetOrigin(new_origin)
    ref_image.SetDirection(image.GetDirection())
    return sitk.Resample(image, ref_image, interpolator=sitk.sitkLinear)


def _parse_margin(margin_text: Optional[str]) -> Optional[Union[int, float]]:
    if margin_text is None:
        return None
    margin_text = margin_text.strip()
    if margin_text == "":
        return None
    if "." in margin_text:
        return float(margin_text)
    return int(margin_text)


def _ensure_sitk() -> None:
    if sitk is None:
        raise RuntimeError(
            "SimpleITK is required for .mhd/.mha conversion or center cropping."
        )


def convert_mhd_to_nifti(src: Path, dst: Path) -> None:
    _ensure_sitk()
    img = sitk.ReadImage(str(src))
    sitk.WriteImage(img, str(dst))


def center_crop_file(
    src: Path,
    dst: Path,
    margin: Union[int, float],
    interpolator: Optional[int] = None,
) -> None:
    _ensure_sitk()
    img = sitk.ReadImage(str(src))
    cropped = _center_crop_sitk(img, margin)
    if interpolator is not None:
        ref_image = sitk.Image(cropped.GetSize(), img.GetPixelIDValue())
        ref_image.SetSpacing(cropped.GetSpacing())
        ref_image.SetOrigin(cropped.GetOrigin())
        ref_image.SetDirection(cropped.GetDirection())
        cropped = sitk.Resample(img, ref_image, interpolator=interpolator)
    sitk.WriteImage(cropped, str(dst))


def collect_inputs(input_path: Path) -> List[CaseRecord]:
    if input_path.is_file():
        name = input_path.name.lower()
        if (
            input_path.suffix.lower() in {".nii", ".mhd", ".mha"}
            or name.endswith(".nii.gz")
        ):
            if _is_label_path(input_path):
                raise ValueError(
                    f"Input file {input_path} looks like a label. Please pass the MRI image."
                )
            label_path = _find_label_for_image(input_path)
            return [
                CaseRecord(
                    case_id=_case_id_from_path(input_path),
                    image=input_path,
                    label=label_path,
                )
            ]
        raise ValueError(f"Unsupported input file: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    exts = {".nii", ".mhd", ".mha"}
    images = {}
    labels = {}
    for p in sorted(input_path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts or p.name.lower().endswith(".nii.gz"):
            case_id = _case_id_from_path(p)
            if _is_label_path(p):
                labels[case_id] = p
            else:
                images[case_id] = p
    return [
        CaseRecord(case_id=case_id, image=image_path, label=labels.get(case_id))
        for case_id, image_path in images.items()
    ]


def preprocess_inputs(
    cases: List[CaseRecord],
    output_dir: Path,
    margin: Optional[Union[int, float]],
    keep_temp: bool,
) -> Tuple[List[CaseRecord], Optional[tempfile.TemporaryDirectory]]:
    temp_dir = None
    needs_temp = margin is not None or any(
        case.image.suffix.lower() in {".mhd", ".mha"}
        or (case.label is not None and case.label.suffix.lower() in {".mhd", ".mha"})
        for case in cases
    )
    if needs_temp:
        if keep_temp:
            temp_root = output_dir / "_tmp"
            temp_root.mkdir(parents=True, exist_ok=True)
            temp_dir_path = temp_root
        else:
            temp_dir = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir.name)
        converted_cases = []
        for case in cases:
            cur_image = case.image
            cur_label = case.label
            if case.image.suffix.lower() in {".mhd", ".mha"}:
                out_path = temp_dir_path / f"{_path_stem(case.image)}.nii.gz"
                convert_mhd_to_nifti(case.image, out_path)
                cur_image = out_path
            if case.label is not None and case.label.suffix.lower() in {".mhd", ".mha"}:
                out_path = temp_dir_path / f"{_path_stem(case.label)}.nii.gz"
                convert_mhd_to_nifti(case.label, out_path)
                cur_label = out_path
            if margin is not None:
                image_out = temp_dir_path / f"{_path_stem(cur_image)}_cropped.nii.gz"
                center_crop_file(cur_image, image_out, margin, interpolator=sitk.sitkLinear)
                cur_image = image_out
                if cur_label is not None:
                    label_out = temp_dir_path / f"{_path_stem(cur_label)}_cropped.nii.gz"
                    center_crop_file(
                        cur_label,
                        label_out,
                        margin,
                        interpolator=sitk.sitkNearestNeighbor,
                    )
                    cur_label = label_out
            converted_cases.append(
                CaseRecord(
                    case_id=case.case_id,
                    image=cur_image,
                    label=cur_label,
                    source_image=case.source_image,
                    source_label=case.source_label,
                )
            )
        return converted_cases, temp_dir
    return cases, None


def build_network(device: torch.device) -> torch.nn.Module:
    net = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2, 2),
        num_res_units=4,
        norm="batch",
        act="prelu",
        dropout=0.15,
    )
    return net.to(device)


def load_model(model_path: Path, device: torch.device, force_ts: bool) -> torch.nn.Module:
    if force_ts or model_path.suffix.lower() == ".ts":
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        return model
    model = build_network(device)
    state = torch.load(str(model_path), map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def _copy_if_present(src: Optional[Path], dst: Path) -> None:
    if src is None:
        return
    shutil.copy2(src, dst)


def _case_output_dir(output_dir: Path, case_id: str) -> Path:
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _save_case_artifact(item: dict, key: str, output_dir: Path, postfix: str) -> None:
    if key not in item:
        return
    value = item[key]
    if hasattr(value, "meta"):
        meta_source = item.get(f"{key}_source")
        if meta_source:
            value.meta["filename_or_obj"] = str(meta_source)
    SaveImaged(
        keys=key,
        output_dir=str(output_dir),
        output_postfix=postfix,
        output_ext=".nii.gz",
        separate_folder=False,
        resample=False,
        allow_missing_keys=True,
    )(item)


def _load_reference_label(label_path: Path) -> torch.Tensor:
    label = LoadImaged(keys="label")({"label": str(label_path)})["label"]
    label = EnsureChannelFirstd(keys="label")({"label": label})["label"]
    return label


def _label_stats(tensor: torch.Tensor) -> Tuple[List[int], dict]:
    data = torch.as_tensor(tensor).cpu().squeeze()
    unique_labels = [int(v) for v in torch.unique(data).tolist()]
    counts = {label: int((data == label).sum().item()) for label in unique_labels}
    return unique_labels, counts


def _dice_for_class(pred: torch.Tensor, target: torch.Tensor, class_value: int) -> float:
    pred_mask = torch.as_tensor(pred).cpu().squeeze() == class_value
    target_mask = torch.as_tensor(target).cpu().squeeze() == class_value
    pred_sum = int(pred_mask.sum().item())
    target_sum = int(target_mask.sum().item())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    if pred_sum + target_sum == 0:
        return 0.0
    intersection = int((pred_mask & target_mask).sum().item())
    return (2.0 * intersection) / (pred_sum + target_sum)


def _affine_match(pred: torch.Tensor, label: torch.Tensor) -> bool:
    pred_affine = getattr(pred, "affine", None)
    label_affine = getattr(label, "affine", None)
    if pred_affine is None or label_affine is None:
        return False
    return bool(torch.allclose(torch.as_tensor(pred_affine), torch.as_tensor(label_affine)))


def _compute_case_metrics(case_id: str, pred: torch.Tensor, label: torch.Tensor) -> dict:
    pred_data = torch.as_tensor(pred).cpu().squeeze()
    label_data = torch.as_tensor(label).cpu().squeeze()
    gt_unique, gt_counts = _label_stats(label_data)
    pred_unique, pred_counts = _label_stats(pred_data)
    shape_match = tuple(pred_data.shape) == tuple(label_data.shape)
    binary_gt = set(gt_unique).issubset({0, 1}) and 1 in gt_unique and 2 not in gt_unique

    return {
        "case_id": case_id,
        "eval_mode": "whole-gland binary GT" if binary_gt else "zonal multiclass GT",
        "gt_unique": gt_unique,
        "pred_unique": pred_unique,
        "gt_counts": gt_counts,
        "pred_counts": pred_counts,
        "shape_match": shape_match,
        "affine_match": _affine_match(pred, label),
        "dice_whole_gland": _dice_for_class(pred_data > 0, label_data > 0, True),
        "dice_central_gland": None if binary_gt else _dice_for_class(pred_data, label_data, 1),
        "dice_peripheral_zone": None if binary_gt else _dice_for_class(pred_data, label_data, 2),
    }


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"


def _summary_row(rows: List[dict]) -> dict:
    whole_gland_rows = [row["dice_whole_gland"] for row in rows if row["dice_whole_gland"] is not None]
    cg_rows = [row["dice_central_gland"] for row in rows if row["dice_central_gland"] is not None]
    pz_rows = [row["dice_peripheral_zone"] for row in rows if row["dice_peripheral_zone"] is not None]
    return {
        "case_id": "Dataset mean",
        "eval_mode": ",".join(sorted({row["eval_mode"] for row in rows})),
        "dice_whole_gland": statistics.fmean(whole_gland_rows) if whole_gland_rows else None,
        "dice_central_gland": statistics.fmean(cg_rows) if cg_rows else None,
        "dice_peripheral_zone": statistics.fmean(pz_rows) if pz_rows else None,
    }


def _write_results_text(output_dir: Path, rows: List[dict]) -> None:
    if not rows:
        return
    summary = _summary_row(rows)
    lines = [
        "Per-case debug and metric report",
        "case_id\teval_mode\tgt_unique\tpred_unique\tshape_match\taffine_match\t"
        "dice_whole_gland\tdice_central_gland\tdice_peripheral_zone",
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row["case_id"],
                    row["eval_mode"],
                    str(row["gt_unique"]),
                    str(row["pred_unique"]),
                    str(row["shape_match"]),
                    str(row["affine_match"]),
                    _format_metric(row["dice_whole_gland"]),
                    _format_metric(row["dice_central_gland"]),
                    _format_metric(row["dice_peripheral_zone"]),
                ]
            )
        )
        lines.append(f"  gt_counts={row['gt_counts']}")
        lines.append(f"  pred_counts={row['pred_counts']}")
    lines.extend(
        [
            "",
            "Dataset summary",
            f"evaluation_modes={summary['eval_mode']}",
            "\t".join(
                [
                    summary["case_id"],
                    _format_metric(summary["dice_whole_gland"]),
                    _format_metric(summary["dice_central_gland"]),
                    _format_metric(summary["dice_peripheral_zone"]),
                ]
            ),
            "",
            "Note:",
            "If eval_mode is whole-gland binary GT, the ground truth contains only labels [0, 1].",
            "In that case, zonal Dice for central/peripheral gland is not valid on this dataset.",
            "Whole-gland Dice is computed from pred > 0 versus gt > 0 instead.",
        ]
    )
    (output_dir / "results.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_results_tex(output_dir: Path, rows: List[dict]) -> None:
    if not rows:
        return
    summary = _summary_row(rows)
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Case & Mode & Whole Gland Dice & Dice (Central Gland) & Dice (Peripheral Zone) \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['case_id']} & "
            f"{row['eval_mode']} & "
            f"{_format_metric(row['dice_whole_gland'])} & "
            f"{_format_metric(row['dice_central_gland'])} & "
            f"{_format_metric(row['dice_peripheral_zone'])} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            f"{summary['case_id']} & "
            f"{summary['eval_mode']} & "
            f"{_format_metric(summary['dice_whole_gland'])} & "
            f"{_format_metric(summary['dice_central_gland'])} & "
            f"{_format_metric(summary['dice_peripheral_zone'])} \\\\",
            r"\hline",
            r"\end{tabular}",
            "",
        ]
    )
    (output_dir / "results.tex").write_text("\n".join(lines), encoding="ascii")


def _save_case_inputs(item: dict, case: CaseRecord, output_dir: Path) -> Path:
    case_dir = _case_output_dir(output_dir, case.case_id)
    _copy_if_present(case.source_image, case_dir / case.source_image.name)
    if case.source_label is not None:
        _copy_if_present(case.source_label, case_dir / case.source_label.name)
    return case_dir


def run_inference(
    model: torch.nn.Module,
    cases: List[CaseRecord],
    output_dir: Path,
    device: torch.device,
    roi_size: Tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    output_postfix: str,
    amp: bool,
    num_workers: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []

    preprocessing = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS", labels=None),
            Spacingd(keys="image", pixdim=(0.5, 0.5, 0.5), mode="bilinear"),
            ScaleIntensityd(keys="image", minv=0, maxv=1),
            NormalizeIntensityd(keys="image"),
            EnsureTyped(keys="image", track_meta=True),
        ]
    )

    postprocessing = Compose(
        [
            AsDiscreted(keys="pred", argmax=True),
            KeepLargestConnectedComponentd(keys="pred", applied_labels=[1, 2]),
            Invertd(
                keys="pred",
                transform=preprocessing,
                orig_keys="image",
                meta_key_postfix="meta_dict",
                nearest_interp=True,
                to_tensor=True,
            ),
        ]
    )

    data = []
    for case in cases:
        data.append({"case_id": case.case_id, "image": str(case.image)})
    ds = Dataset(data=data, transform=preprocessing)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
    case_lookup = {case.case_id: case for case in cases}

    inferer = SlidingWindowInferer(
        roi_size=roi_size, sw_batch_size=sw_batch_size, overlap=overlap
    )

    iterable = loader
    if tqdm is not None:
        iterable = tqdm(loader, desc="Inference", unit="case")

    for batch_data in iterable:
        inputs = batch_data["image"].to(device)
        with torch.no_grad():
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    pred = inferer(inputs, model)
            else:
                pred = inferer(inputs, model)
        batch_data["pred"] = pred.detach().cpu()
        for item in decollate_batch(batch_data):
            case_id = _as_text(item["case_id"])
            case = case_lookup[case_id]
            case_dir = _save_case_inputs(item, case, output_dir)
            item["pred_source"] = str(case.source_image)
            item = postprocessing(item)
            _save_case_artifact(item, "pred", case_dir, output_postfix)
            if case.source_label is not None:
                label = _load_reference_label(case.source_label)
                metric_rows.append(_compute_case_metrics(case_id, item["pred"], label))
    _write_results_text(output_dir, metric_rows)
    _write_results_tex(output_dir, metric_rows)


def main() -> None:
    args = parse_args()

    set_track_meta(True)
    base_dir = Path(__file__).resolve().parent
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        cases = collect_inputs(input_path)
    else:
        default_dir = (base_dir.parent / "PROMISE12_nifti").resolve()
        if not default_dir.exists():
            raise RuntimeError(
                "No --input provided and default PROMISE12_nifti folder not found."
            )
        cases = collect_inputs(default_dir)
        if len(cases) < 25:
            raise RuntimeError("Need at least 25 non-segmentation images to run.")
        cases = cases[:25]

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (base_dir.parent / "pred").resolve()
    )
    model_path = (
        Path(args.model).expanduser().resolve()
        if args.model
        else (base_dir / "models" / "model.pt").resolve()
    )
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    margin = _parse_margin(args.center_crop)
    if margin is not None and sitk is None:
        raise RuntimeError("Center cropping requires SimpleITK to be installed.")

    if not cases:
        raise RuntimeError("No supported input files found.")

    processed_cases, temp_dir = preprocess_inputs(
        cases, output_dir, margin=margin, keep_temp=args.keep_temp
    )

    device = torch.device(args.device)
    model = load_model(model_path, device, args.torchscript)

    run_inference(
        model=model,
        cases=processed_cases,
        output_dir=output_dir,
        device=device,
        roi_size=_parse_roi_size(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        output_postfix=args.output_postfix,
        amp=args.amp,
        num_workers=args.num_workers,
    )

    if temp_dir is not None and not args.keep_temp:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
