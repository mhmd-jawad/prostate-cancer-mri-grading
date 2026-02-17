## Models to be used in this project

| Model Name | Model Type | Version | Datatype | Model Links (Download, README) | Data Annotation Approach | Purpose |
|---|---|---|---|---|---|---|
| YOLOv26-seg | Segmentation | YOLOv26 | 2D MRI images (tumor localization/segmentation workflow) | [Download the model](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-seg.pt), [README](https://docs.ultralytics.com/tasks/segment/#models) | Roboflow (polygon masks + manual correction/review) | Detect and segment tumor regions from MRI scans. |
| SAM (Segment Anything Model) | Segmentation | SAM 3 | 2D MRI slices (promptable masks from points/boxes) | [Download](https://huggingface.co/facebook/sam3), [README](https://github.com/facebookresearch/sam3/blob/main/README.md) | SAM-assisted pre-annotation + manual correction in Roboflow (Smart Select / Smart Polygon) | Speed up tumor mask creation and improve annotation quality for training/evaluation. |
