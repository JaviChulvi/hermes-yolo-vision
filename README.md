# YOLO Vision for Hermes

**Give your Hermes agent eyes with Ultralytics YOLO.**

`hermes-yolo-vision` is a planned plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent) that brings [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) computer vision into agent workflows. The goal is to let agents process images and videos through simple tools, then use structured results to answer questions and complete tasks.

> **Status: planning.** This repository currently contains the project overview. The plugin is not implemented or available to install yet. The capabilities below describe the intended scope.

## What it should feel like

Ask your agent to:

- “Find the people and vehicles in this image.”
- “Segment each object and save the masks.”
- “Track the cars through this video and show their trajectories.”
- “Extract the human poses from this clip.”
- “Estimate the scene's depth and save the depth map.”
- “Look for helmets in these images using text prompts.”

The agent chooses the appropriate vision task, runs a compatible model, and returns a concise explanation alongside the underlying results and visual artifacts.

## Planned capabilities

The intended coverage includes all seven [Ultralytics YOLO vision tasks](https://docs.ultralytics.com/tasks/):

| Task | Output |
| --- | --- |
| Object detection | Classes, confidence scores, and bounding boxes |
| Instance segmentation | Individual object masks and boundaries |
| Semantic segmentation | A class label for each pixel |
| Depth estimation | Estimated per-pixel depth |
| Image classification | Image-level classes and scores |
| Pose estimation | Keypoints and confidence scores |
| Oriented object detection | Rotated bounding boxes |

Additional planned support:

- **Video tracking:** object IDs and trajectories across frames for compatible models.
- **Open-vocabulary vision:** text and visual prompts through [YOLOE](https://docs.ultralytics.com/models/yoloe/).
- **Custom models:** use compatible weights trained for a specific domain.
- **Reusable results:** structured data, annotated images and videos, masks, and depth maps.

Available classes and operations depend on the selected model. Tracking IDs describe estimated tracks; depth values are model estimates.

## Designed for agents

- **Simple tools:** expose clear inputs and outputs so the agent can choose and run a vision task without writing inference code.
- **A bundled skill:** guide task selection, model choice, and interpretation of results.
- **Compact responses:** return useful summaries while keeping complete outputs in files.
- **Video-aware execution:** process frames incrementally and make long jobs observable and cancelable.
- **Traceable results:** record the model, parameters, and processed frames alongside predictions.

The initial direction is a native Hermes plugin backed by the Ultralytics Python package, with local inference. The tool interface, dependency setup, and implementation milestones will be defined during planning.

## Development

The next step is to define the first working release and its validation criteria. Installation instructions and runnable examples will be added once an implementation is available.

Useful references:

- [Build a Hermes plugin](https://hermes-agent.nousresearch.com/docs/developer-guide/plugins/)
- [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics tracking](https://docs.ultralytics.com/modes/track/)

This is an independent integration project for Hermes Agent and Ultralytics YOLO.
