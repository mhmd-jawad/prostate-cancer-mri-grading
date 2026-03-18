from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np


TYPE_INFO = {
    "MET_CHAR": {"nrrd_type": "signed char", "bytes": 1, "numpy": np.int8},
    "MET_UCHAR": {"nrrd_type": "uchar", "bytes": 1, "numpy": np.uint8},
    "MET_SHORT": {"nrrd_type": "short", "bytes": 2, "numpy": np.int16},
    "MET_USHORT": {"nrrd_type": "ushort", "bytes": 2, "numpy": np.uint16},
    "MET_INT": {"nrrd_type": "int", "bytes": 4, "numpy": np.int32},
    "MET_UINT": {"nrrd_type": "uint", "bytes": 4, "numpy": np.uint32},
    "MET_FLOAT": {"nrrd_type": "float", "bytes": 4, "numpy": np.float32},
    "MET_DOUBLE": {"nrrd_type": "double", "bytes": 8, "numpy": np.float64},
}


def parse_mhd(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        meta[key] = value
    required = {
        "DimSize",
        "ElementSpacing",
        "ElementType",
        "ElementDataFile",
        "BinaryDataByteOrderMSB",
    }
    missing = required - meta.keys()
    if missing:
        raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
    return meta


def parse_numbers(text: str, cast):
    return [cast(value) for value in text.split()]


def compute_space_directions(meta: dict[str, str]) -> list[tuple[float, float, float]] | None:
    spacing = parse_numbers(meta["ElementSpacing"], float)
    if "TransformMatrix" not in meta:
        return None
    matrix_values = parse_numbers(meta["TransformMatrix"], float)
    dims = len(spacing)
    if len(matrix_values) != dims * dims:
        raise ValueError("TransformMatrix size does not match spacing dimensionality")
    directions = []
    for axis in range(dims):
        vector = []
        # MetaImage stores TransformMatrix in row-major order, and each row
        # describes the physical direction of one image axis.
        row_start = axis * dims
        for column in range(dims):
            vector.append(matrix_values[row_start + column] * spacing[axis])
        directions.append(tuple(vector))
    return directions


def format_vector(vector: tuple[float, ...]) -> str:
    return "(" + ",".join(f"{value:.17g}" for value in vector) + ")"


def build_header(meta: dict[str, str], nrrd_type: str) -> bytes:
    sizes = " ".join(meta["DimSize"].split())
    dimension = len(meta["DimSize"].split())
    endian = "big" if meta["BinaryDataByteOrderMSB"].lower() == "true" else "little"

    header_lines = [
        "NRRD0005",
        "# Converted from MetaImage (.mhd/.raw) without changing voxel data ordering.",
        f"type: {nrrd_type}",
        f"dimension: {dimension}",
        "space: left-posterior-superior",
        f"sizes: {sizes}",
    ]

    space_directions = compute_space_directions(meta)
    if space_directions:
        header_lines.append(
            "space directions: " + " ".join(format_vector(vector) for vector in space_directions)
        )

    header_lines.extend(
        [
            "kinds: domain domain domain",
            f"endian: {endian}",
            'space units: "mm" "mm" "mm"',
            "encoding: raw",
        ]
    )

    if "Offset" in meta:
        origin = tuple(parse_numbers(meta["Offset"], float))
        header_lines.append(f"space origin: {format_vector(origin)}")

    header_lines.append("")
    header_lines.append("")
    return "\n".join(header_lines).encode("ascii")


def expected_bytes(meta: dict[str, str]) -> int:
    dims = parse_numbers(meta["DimSize"], int)
    element_type = meta["ElementType"]
    if element_type not in TYPE_INFO:
        raise ValueError(f"Unsupported ElementType: {element_type}")
    return math.prod(dims) * TYPE_INFO[element_type]["bytes"]


def validate_segmentation_values(raw_path: Path, element_type: str) -> None:
    if element_type != "MET_CHAR":
        return
    values = np.fromfile(raw_path, dtype=np.int8)
    if values.size == 0:
        raise ValueError(f"{raw_path.name}: segmentation is empty")
    unique_values = set(np.unique(values).tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"{raw_path.name}: expected binary labels, found values {sorted(unique_values)}"
        )


def convert_case(mhd_path: Path, output_path: Path, segmentation: bool) -> None:
    meta = parse_mhd(mhd_path)
    raw_path = mhd_path.with_name(meta["ElementDataFile"])
    if not raw_path.exists():
        raise FileNotFoundError(f"{mhd_path.name}: raw file not found: {raw_path.name}")

    actual_size = raw_path.stat().st_size
    required_size = expected_bytes(meta)
    if actual_size != required_size:
        raise ValueError(
            f"{raw_path.name}: expected {required_size} bytes from metadata, found {actual_size}"
        )

    element_type = meta["ElementType"]
    if segmentation:
        validate_segmentation_values(raw_path, element_type)
        nrrd_type = "uchar" if element_type == "MET_CHAR" else TYPE_INFO[element_type]["nrrd_type"]
    else:
        nrrd_type = TYPE_INFO[element_type]["nrrd_type"]

    header = build_header(meta, nrrd_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as target:
        target.write(header)
        with raw_path.open("rb") as source:
            shutil.copyfileobj(source, target, length=1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MetaImage (.mhd + .raw) cases into attached .nrrd files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("training_data"),
        help="Folder that contains the MetaImage cases.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("nrrd_images"),
        help="Output folder for MRI volumes.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("nrrd_labels"),
        help="Output folder for segmentation labelmaps.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    images_dir = args.images_dir.resolve()
    labels_dir = args.labels_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    cases = sorted(input_dir.glob("*.mhd"))
    image_cases = [path for path in cases if "_segmentation" not in path.stem]
    label_cases = [path for path in cases if "_segmentation" in path.stem]

    for case in image_cases:
        convert_case(case, images_dir / f"{case.stem}.nrrd", segmentation=False)

    for case in label_cases:
        convert_case(case, labels_dir / f"{case.stem}.nrrd", segmentation=True)

    print(f"Converted {len(image_cases)} image volumes to {images_dir}")
    print(f"Converted {len(label_cases)} segmentation volumes to {labels_dir}")


if __name__ == "__main__":
    main()
