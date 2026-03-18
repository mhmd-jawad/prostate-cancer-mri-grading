from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TYPE_MAP = {
    "signed char": np.int8,
    "int8": np.int8,
    "char": np.int8,
    "uchar": np.uint8,
    "unsigned char": np.uint8,
    "uint8": np.uint8,
    "short": np.int16,
    "short int": np.int16,
    "signed short": np.int16,
    "signed short int": np.int16,
    "ushort": np.uint16,
    "unsigned short": np.uint16,
    "unsigned short int": np.uint16,
    "int": np.int32,
    "signed int": np.int32,
    "uint": np.uint32,
    "unsigned int": np.uint32,
    "float": np.float32,
    "double": np.float64,
}

CASE_PATTERN = re.compile(r"(Case\d+)", re.IGNORECASE)


@dataclass
class NrrdImage:
    path: Path
    data: np.ndarray
    sizes: tuple[int, ...]
    space_directions: tuple[tuple[float, ...], ...] | None
    space_origin: tuple[float, ...] | None


def read_nrrd(path: Path) -> NrrdImage:
    header, payload = read_nrrd_header_and_payload(path)
    sizes = tuple(int(value) for value in header["sizes"].split())
    dtype = build_dtype(header)
    encoding = header.get("encoding", "raw").strip().lower()

    if encoding in {"gzip", "gz"}:
        payload = gzip.decompress(payload)
    elif encoding != "raw":
        raise ValueError(f"{path}: unsupported NRRD encoding '{encoding}'")

    voxel_count = math.prod(sizes)
    expected_bytes = voxel_count * dtype.itemsize
    if len(payload) != expected_bytes:
        raise ValueError(
            f"{path}: expected {expected_bytes} bytes from header, found {len(payload)}"
        )

    data = np.frombuffer(payload, dtype=dtype, count=voxel_count).copy()
    data = data.reshape(tuple(reversed(sizes)))

    return NrrdImage(
        path=path,
        data=data,
        sizes=sizes,
        space_directions=parse_vectors(header.get("space directions")),
        space_origin=parse_vector(header.get("space origin")),
    )


def read_nrrd_header_and_payload(path: Path) -> tuple[dict[str, str], bytes]:
    header_lines: list[str] = []
    with path.open("rb") as stream:
        magic = stream.readline().decode("ascii").strip()
        if not magic.startswith("NRRD"):
            raise ValueError(f"{path}: not a NRRD file")
        header_lines.append(magic)

        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: incomplete NRRD header")
            if line.strip() == b"":
                break
            header_lines.append(line.decode("ascii").rstrip("\r\n"))

        payload = stream.read()

    header: dict[str, str] = {}
    for line in header_lines[1:]:
        if not line or line.startswith("#") or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        header[key.strip().lower()] = value.strip()

    required = {"type", "dimension", "sizes"}
    missing = required - header.keys()
    if missing:
        raise ValueError(f"{path}: missing NRRD fields {sorted(missing)}")

    return header, payload


def build_dtype(header: dict[str, str]) -> np.dtype:
    type_name = header["type"].strip().lower()
    if type_name not in TYPE_MAP:
        raise ValueError(f"Unsupported NRRD type: {header['type']}")

    dtype = np.dtype(TYPE_MAP[type_name])
    if dtype.itemsize == 1:
        return dtype

    endian = header.get("endian", "").strip().lower()
    if endian == "little":
        return dtype.newbyteorder("<")
    if endian == "big":
        return dtype.newbyteorder(">")
    raise ValueError("NRRD endian field is required for multi-byte voxel types")


def parse_vectors(value: str | None) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    parts = value.split(") (")
    normalized = []
    for part in parts:
        if not part.startswith("("):
            part = "(" + part
        if not part.endswith(")"):
            part = part + ")"
        normalized.append(parse_vector(part))
    return tuple(normalized)


def parse_vector(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    text = value.strip()
    if text.lower() == "none":
        return None
    if not (text.startswith("(") and text.endswith(")")):
        raise ValueError(f"Cannot parse vector: {value}")
    return tuple(float(item) for item in text[1:-1].split(","))


def dice_score(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    gt = ground_truth > 0
    pred = prediction > 0

    gt_sum = int(np.count_nonzero(gt))
    pred_sum = int(np.count_nonzero(pred))
    if gt_sum == 0 and pred_sum == 0:
        return 1.0

    intersection = int(np.count_nonzero(gt & pred))
    return (2.0 * intersection) / (gt_sum + pred_sum)


def case_id(path: Path) -> str:
    match = CASE_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Cannot extract case id from filename: {path.name}")
    return match.group(1)


def geometry_matches(
    image_a: NrrdImage, image_b: NrrdImage, tolerance: float
) -> tuple[bool, str]:
    if image_a.sizes != image_b.sizes:
        return False, f"size mismatch: {image_a.sizes} vs {image_b.sizes}"

    if image_a.space_origin is not None and image_b.space_origin is not None:
        if not all(
            math.isclose(a, b, abs_tol=tolerance)
            for a, b in zip(image_a.space_origin, image_b.space_origin)
        ):
            return False, "space origin mismatch"

    if image_a.space_directions is not None and image_b.space_directions is not None:
        for vec_a, vec_b in zip(image_a.space_directions, image_b.space_directions):
            if vec_a is None or vec_b is None:
                continue
            if not all(
                math.isclose(a, b, abs_tol=tolerance) for a, b in zip(vec_a, vec_b)
            ):
                return False, "space directions mismatch"

    return True, "ok"


def collect_cases(directory: Path) -> dict[str, Path]:
    return {case_id(path): path for path in sorted(directory.glob("*.nrrd"))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Dice scores between ground-truth and predicted NRRD masks."
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("nrrd_labels"),
        help="Directory that contains ground-truth label NRRD files.",
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("deepinfer_results_first25"),
        help="Directory that contains predicted label NRRD files.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path for saving the per-case Dice scores.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance used for geometry comparison.",
    )
    args = parser.parse_args()

    gt_dir = args.ground_truth_dir.resolve()
    pred_dir = args.prediction_dir.resolve()

    gt_cases = collect_cases(gt_dir)
    pred_cases = collect_cases(pred_dir)
    common_cases = sorted(set(gt_cases) & set(pred_cases))

    if not common_cases:
        raise SystemExit("No matching cases found between the two directories.")

    rows: list[dict[str, object]] = []
    for case in common_cases:
        gt_image = read_nrrd(gt_cases[case])
        pred_image = read_nrrd(pred_cases[case])

        geometry_ok, geometry_note = geometry_matches(gt_image, pred_image, args.tolerance)
        score = dice_score(gt_image.data, pred_image.data)

        rows.append(
            {
                "case": case,
                "dice": score,
                "geometry_ok": geometry_ok,
                "geometry_note": geometry_note,
                "gt_voxels": int(np.count_nonzero(gt_image.data)),
                "pred_voxels": int(np.count_nonzero(pred_image.data)),
            }
        )

    mean_dice = float(np.mean([row["dice"] for row in rows]))

    print("Case\tDice\tGeometryOK\tGroundTruthVoxels\tPredictedVoxels")
    for row in rows:
        print(
            f"{row['case']}\t{row['dice']:.6f}\t{row['geometry_ok']}\t"
            f"{row['gt_voxels']}\t{row['pred_voxels']}"
        )
    print(f"Mean\t{mean_dice:.6f}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "case",
                    "dice",
                    "geometry_ok",
                    "geometry_note",
                    "gt_voxels",
                    "pred_voxels",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(
                {
                    "case": "Mean",
                    "dice": mean_dice,
                    "geometry_ok": "",
                    "geometry_note": "",
                    "gt_voxels": "",
                    "pred_voxels": "",
                }
            )


if __name__ == "__main__":
    main()
