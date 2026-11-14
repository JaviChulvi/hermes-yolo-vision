---
name: yolo-python
description: Analyze uploaded images and videos with installed Ultralytics Python/CLI and produce verifiable artifacts.
---

Use the installed Python environment and Ultralytics package. Resolve the actual attachment path; do not invent filenames or install a different model package. Prefer short Python scripts over shell quoting for multi-step work.

Choose small pretrained weights: detect yolo26n.pt; segment yolo26n-seg.pt; semantic yolo26n-sem-ade20k.pt; depth yolo26n-depth.pt; classify yolo26n-cls.pt; pose yolo26n-pose.pt; obb yolo26n-obb.pt. Use a supplied trusted custom .pt when requested. Inspect model.task and model.names before claiming category support. Ordinary classes filter known class IDs; unseen labels require YOLOE, not renaming fixed classes.

```python
from ultralytics import YOLO, YOLOE
model = YOLO('yolo26n.pt')
for result in model.predict(source, stream=True, device='cpu', verbose=False):
    rows = result.summary(normalize=False)
```

Results.summary() serializes boxes/confidence/class IDs, segmentation polygons, pose keypoints, rotated corners, classification rankings, and semantic class areas. Depth must be serialized separately from result.depth.data.cpu().numpy(); preserve finite values and describe estimated units. Dense masks are result.masks.data; semantic labels are result.semantic_mask.data. Save NumPy arrays with np.savez_compressed when requested; JSONL is useful for per-frame observations. Use result.plot() for annotations (BGR pixels) and preserve the original coordinates/orientation. PIL inputs are RGB; NumPy inputs are BGR.

YOLOE text prompting:
```python
model = YOLOE('yoloe-26s-seg.pt')
model.set_classes(['helmet', 'bag'])
```

For video, iterate frames with PyAV for actual presentation timestamps and display rotation. Use stream=True for bounded inference memory; do not load the entire clip's predictions into a list. For persistent IDs, call model.track(frame, persist=True, stream=True, tracker='bytetrack.yaml'). Check IDs exist. Write H.264 MP4 using PyAV/libx264 and preserve timestamps; if audio is omitted, disclose it. Sample only when useful and state coverage. Avoid conflating per-frame observations, concurrent occupancy, distinct track IDs, and line crossings.

Use upstream Solutions where they fit: AIGym for repetition counting with visible relevant joints; QueueManager for occupancy within a specified polygon; ObjectCounter for crossing a specified line. A region/line and stable tracks are required. Do not infer workouts from a single frame or invent a camera calibration. For cutouts, use instance masks to create an alpha channel; preserve dimensions and inspect saved output.

Ground conclusions in saved model outputs. Empty results mean nothing was detected with these settings, not proof of absence. Confidence scores are model scores. Monocular depth is an estimate, not a calibrated distance measurement. Occluded joints and fragmented tracks limit downstream claims.

Return requested artifacts with standalone MEDIA:/absolute/path lines outside JSON/code fences. Use the allowed generated-media directory under the active HERMES_HOME/cache/generated. Save full observations for follow-ups and reuse them instead of repeating inference unnecessarily. Verify files exist and decode before announcing success. Report errors and unfinished work honestly.
