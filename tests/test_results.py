import numpy as np

from hermes_yolo_vision.worker import observations


def test_depth_summary_is_finite_and_not_an_empty_detection_list():
    from ultralytics.engine.results import Results

    depth = np.array([[1, 2], [np.nan, 0]], dtype=np.float32)
    result = Results(np.zeros((2, 2, 3), dtype=np.uint8), path="depth.png", names={}, depth=depth)
    result.depth.data = __import__("torch").from_numpy(depth)
    summary = observations(result)["depth"]
    assert summary["median"] == 1.5 and summary["valid_pixel_ratio"] == 0.5


def test_empty_detection_is_successful_empty_observations():
    import torch
    from ultralytics.engine.results import Results

    result = Results(
        np.zeros((4, 4, 3), dtype=np.uint8),
        path="blank.png",
        names={0: "person"},
        boxes=torch.empty((0, 6)),
    )
    assert observations(result) == {"observations": []}
