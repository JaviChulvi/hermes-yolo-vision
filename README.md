# YOLO Vision for Hermes

Give your Hermes agent eyes with Ultralytics YOLO.

`hermes-yolo-vision` adds four native tools and a skill to [Hermes Agent](https://github.com/NousResearch/hermes-agent). Ask about an uploaded image or video; Hermes can run a vision model, inspect structured observations, and return annotated media.

**Status: release candidate under evaluation.** Real inference and artifact checks pass for all seven tasks, YOLOE text prompts, and tracking. The conversation benchmark and live Telegram acceptance tests are still in progress. This is not yet a published or universally validated release.

## What it can do

| Task | Result | Default model |
| --- | --- | --- |
| Object detection | Classes, confidence, pixel boxes | `yolo26n.pt` |
| Instance segmentation | Objects and mask polygons | `yolo26n-seg.pt` |
| Semantic segmentation | Class areas and optional pixel mask | `yolo26n-sem-ade20k.pt` |
| Depth estimation | Estimated depth summary and optional depth array | `yolo26n-depth.pt` |
| Classification | Ranked image labels and scores | `yolo26n-cls.pt` |
| Pose estimation | Person boxes and joint coordinates | `yolo26n-pose.pt` |
| Oriented boxes | Rotated object corners | `yolo26n-obb.pt` |

All seven tasks accept local images and videos. Tracking adds ByteTrack IDs to detection, segmentation, pose, or oriented boxes. YOLOE supports text-prompted detection/segmentation with `yoloe-26s-seg.pt`. Trusted local `.pt` weights can replace defaults.

Accepted extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, and `.tiff` for images; `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, and `.m4v` for videos.

Video display rotations in multiples of 90 degrees are applied before analysis. Mirrored display metadata is rejected; export an upright video first.

Examples of requests:

- “Mark the people and vehicles in this photo and give me the counts.”
- “Extract the people’s silhouettes and send the masks.”
- “Track the person in this workout clip and export their joint coordinates over time.”
- “Find the helmet in this image.”
- “Give me a depth map and explain its limitations.”
- “Using the previous results, list when each tracked person appeared.”

Pose and tracking provide evidence for workout analysis and counting. Repetition counting, queue rules, product cutouts, and other applications require further analysis by the agent; they are not separate built-in plugin tools. See [the evaluation protocol](evals/README.md) for the use cases being tested.

## Install

Requires Linux or macOS, Hermes **0.23.0 or later**, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/). The inference runtime uses Python 3.12 and is separate from Hermes’s Python environment. Windows is not supported in this release.

Once the implementation is merged:

```bash
hermes plugins install https://github.com/JaviChulvi/hermes-yolo-vision
python3 "${HERMES_HOME:-$HOME/.hermes}/plugins/hermes-yolo-vision/scripts/setup_runtime.py"
hermes plugins enable hermes-yolo-vision
```

Restart the relevant Hermes session/gateway after enabling. Ask “Check whether YOLO Vision is ready.” The setup command installs dependencies; the tool handlers never install packages. CPU is the default. On a CUDA machine, pass `--torch-backend auto` to setup and configure the device as `"0"`.

The first inference downloads the chosen model. YOLOE also downloads a text encoder. Allow disk space for PyTorch, dependencies, and model caches. Re-run setup after upgrading the plugin.

Optional settings in the active profile’s `config.yaml`:

```yaml
plugins:
  enabled:
    - hermes-yolo-vision
  entries:
    hermes-yolo-vision:
      allow_gateway_injection: true
      settings:
        device: cpu
        max_job_seconds: 900
        # python: /absolute/path/to/another/venv/bin/python
```

Preserve other enabled plugins when editing this list. `allow_gateway_injection` enables background completion notifications in the originating chat. Without it, Hermes retrieves results by polling. Fast results return directly without an extra notification.

## Tools and outputs

- **`yolo_info`** checks dependencies without loading a model, or inspects a specific model’s task and classes. `class_query` filters large label lists.
- **`yolo_predict`** processes one image/video. Choose `task`, `model`, fixed `classes` or YOLOE `text_prompts`, confidence, device, and video sampling.
- **`yolo_track`** processes a video while preserving track state across frames.
- **`yolo_job`** checks, waits for, retrieves, or cancels an existing job.

A prediction waits up to 20 seconds, then returns a job ID if still running. Poll that ID with `wait_seconds: 30`; do not submit the same job again. Only one inference job runs per profile. Another request receives `busy` instead of silently queuing. Jobs time out at the configured limit, terminate on plugin unload, and recover as interrupted after a restart.

Results contain a compact preview and paths to:

- Annotated JPG or H.264 MP4, enabled by default.
- Full observations as JSONL, with original frame indexes and video timestamps.
- Metadata with input/model hashes, parameters, versions, and sampling coverage.
- A ZIP containing observations and metadata, suitable for chat delivery.
- Optional compressed NumPy masks/depth arrays when `save_raw: true`.

Files live under `$HERMES_HOME/cache/generated/hermes-yolo-vision/<job-id>/`. Job records and downloaded models live under `$HERMES_HOME/plugin-data/hermes-yolo-vision/`. Artifacts are retained until the operator removes them. Hermes can return the supplied `MEDIA:` references as standalone lines in chat; acceptance of a notification does not prove Telegram delivered a file.

## Boundaries

- Files and downloaded attachments only. No URLs, cameras, streams, folders, training, export, or cloud deployment.
- Pretrained models recognize their training classes. A missing detection does not prove absence; YOLOE prompts also have limits.
- Video totals count observations across frames, not unique people or products. Track IDs can fragment or switch. Sampling may miss events.
- Depth is a model estimate, not a calibrated distance measurement. Pose coordinates alone do not establish safe exercise form or a medical assessment.
- Annotated videos preserve presentation timing and display orientation, but omit audio. Unsampled frames are retained without annotations.
- Model checkpoints execute through PyTorch: use only trusted local weights.
- The file policy uses Hermes’s existing `agent.file_safety.get_read_block_error` guard, also used by bundled image plugins. Hermes currently flags this import as a private compatibility seam; version-specific integration checks are required.

## Development and verification

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python --torch-backend cpu -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/ruff check hermes_yolo_vision scripts tests
.venv/bin/python scripts/smoke.py --text --tracking
.venv/bin/python scripts/smoke.py --video --text --tracking \
  --report .agent/video-smoke-report.json
```

Smoke tests download official sample media and weights and run real models. Their repeated-image video fixtures verify processing, serialization, and encoding, **not tracking accuracy or user usefulness**. Use the [evaluation protocol](evals/README.md) for conversation-level comparisons.

For standalone diagnosis, use the same worker without an LLM:

```bash
.venv/bin/hermes-yolo-vision info
.venv/bin/hermes-yolo-vision predict \
  --args '{"source":"/absolute/path/photo.jpg","task":"detect"}'
```

The standalone runner waits synchronously and stops its worker when it exits. Background jobs belong to a running Hermes process.

## License

[AGPL-3.0-only](LICENSE). Ultralytics and model assets retain their own licenses. Review [Ultralytics licensing](https://www.ultralytics.com/license) for your intended distribution and commercial use.
