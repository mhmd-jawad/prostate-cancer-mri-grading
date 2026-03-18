import vtk
import slicer


def describe(path):
    node = slicer.util.loadVolume(path)
    if node is None:
        raise RuntimeError(f"Failed to load {path}")

    mat = vtk.vtkMatrix4x4()
    node.GetIJKToRASMatrix(mat)
    image = node.GetImageData()
    values = image.GetScalarRange() if image else None

    lines = [
        f"path={path}",
        f"name={node.GetName()}",
        f"dims={image.GetDimensions()}",
        f"spacing={node.GetSpacing()}",
        f"origin={node.GetOrigin()}",
        f"scalar_range={values}",
        "ijk_to_ras=",
    ]
    for r in range(4):
        lines.append(str([mat.GetElement(r, c) for c in range(4)]))

    slicer.mrmlScene.RemoveNode(node)
    return "\n".join(lines)


paths = [
    r"c:/Users/Administrator/Desktop/New folder (2)/training_data/Case00_segmentation.mhd",
    r"c:/Users/Administrator/Desktop/New folder (2)/nrrd_labels/Case00_segmentation.nrrd",
]

report = "\n---\n".join(describe(path) for path in paths)
with open(r"c:/Users/Administrator/Desktop/New folder (2)/slicer_geometry_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

slicer.util.exit()
