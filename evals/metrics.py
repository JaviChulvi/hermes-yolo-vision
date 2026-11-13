"""Independent rubric metrics; callers explicitly normalize each artifact's documented format."""

import cv2
import lap
import numpy as np


def box_iou(first, second):
    """Axis-aligned boxes in original-pixel [left, top, right, bottom] coordinates."""
    boxes = np.asarray([first, second], dtype=float)
    if boxes.shape != (2, 4) or not np.isfinite(boxes).all():
        raise ValueError("Expected two finite xyxy boxes")
    sizes = boxes[:, 2:] - boxes[:, :2]
    if (sizes <= 0).any():
        raise ValueError("Boxes must have positive area")
    overlap = np.maximum(
        0, np.minimum(boxes[0, 2:], boxes[1, 2:]) - np.maximum(boxes[0, :2], boxes[1, :2])
    )
    intersection = float(np.prod(overlap))
    return intersection / (float(np.prod(sizes[0]) + np.prod(sizes[1])) - intersection)


def polygon_iou(first, second):
    """Unclipped rotated-box corners; reject malformed or self-intersecting quadrilaterals."""
    polygons = [np.asarray(points, dtype=np.float32) for points in (first, second)]
    for polygon in polygons:
        if (
            polygon.shape != (4, 2)
            or not np.isfinite(polygon).all()
            or not cv2.isContourConvex(polygon)
        ):
            raise ValueError("Expected four finite convex corners in perimeter order")
    areas = [cv2.contourArea(polygon) for polygon in polygons]
    if min(areas) <= 0:
        raise ValueError("Polygon area must be positive")
    intersection, _ = cv2.intersectConvexConvex(*polygons)
    return float(intersection / (sum(areas) - intersection))


def mask_iou(prediction, segmentation, *, width, height):
    """Compare a binary source-sized mask with the frozen COCO segmentation."""
    from pycocotools import mask as coco_mask

    prediction = np.asarray(prediction)
    if prediction.shape != (height, width) or not np.isfinite(prediction).all():
        raise ValueError("Mask must be finite and mapped to the source dimensions")
    encoded = coco_mask.frPyObjects(segmentation, height, width)
    if isinstance(encoded, list):
        encoded = coco_mask.merge(encoded)
    truth = coco_mask.decode(encoded).astype(bool)
    prediction = prediction.astype(bool)
    union = np.count_nonzero(prediction | truth)
    return float(np.count_nonzero(prediction & truth) / union) if union else 1.0


def keypoint_score(prediction, annotation):
    """PCK on visible COCO joints, with the rubric's 0.15 box-diagonal tolerance."""
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (17, 2) or not np.isfinite(prediction).all():
        raise ValueError("Expected 17 finite original-pixel xy keypoints")
    truth = np.asarray(annotation["keypoints"], dtype=float).reshape(17, 3)
    visible = truth[:, 2] == 2
    if not visible.any():
        raise ValueError("No visible reference joints")
    tolerance = 0.15 * np.linalg.norm(annotation["bbox"][2:])
    errors = np.linalg.norm(prediction[visible] - truth[visible, :2], axis=1)
    return {
        "visible_joints": int(visible.sum()),
        "tolerance_pixels": float(tolerance),
        "fraction_within_tolerance": float((errors <= tolerance).mean()),
    }


def match_overlaps(overlaps, threshold=0.5):
    """Match one class at a time; maximize accepted matches before total overlap."""
    overlaps = np.asarray(overlaps, dtype=float)
    if (
        overlaps.ndim != 2
        or not np.isfinite(overlaps).all()
        or ((overlaps < 0) | (overlaps > 1)).any()
    ):
        raise ValueError("Expected a ground-truth by prediction IoU matrix in [0, 1]")
    weight = (overlaps >= threshold) * (min(overlaps.shape) + 1) + overlaps
    if not all(overlaps.shape):
        return []
    _, columns, _ = lap.lapjv(-weight, extend_cost=True)
    return [
        {"truth_index": int(i), "prediction_index": int(j), "iou": float(overlaps[i, j])}
        for i, j in enumerate(columns)
        if j >= 0 and overlaps[i, j] >= threshold
    ]
