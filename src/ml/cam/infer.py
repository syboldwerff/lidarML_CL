"""Inference utility for camera object detection.

This module provides a function to run a trained YOLO model on a
single image file.  It returns a list of detections with bounding
boxes and class confidences.  The Ultralytics YOLO API is used for
convenience.
"""

from typing import List, Tuple

from ultralytics import YOLO


def infer_cam(model_path: str, image_path: str) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
    """Run YOLO inference on a single image.

    Parameters
    ----------
    model_path : str
        Path to a trained YOLO model file (e.g. best.pt).
    image_path : str
        Path to an image file.

    Returns
    -------
    list of (class_id, confidence, (x1, y1, x2, y2))
        A list of detections.  Each entry contains the class index,
        confidence score and bounding box coordinates in pixel units.
    """
    model = YOLO(model_path)
    results = model(image_path, verbose=False)
    detections: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
    for result in results:
        # result.boxes contains a Boxes object; convert to xyxy format
        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)
            detections.append((int(cls), float(conf), (x1, y1, x2, y2)))
    return detections