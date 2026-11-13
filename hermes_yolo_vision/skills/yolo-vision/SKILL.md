---
name: yolo-vision
description: Analyze images and videos with YOLO; deliver grounded results.
---

# YOLO Vision

Use the `yolo_vision` tools for measured visual observations and reusable artifacts.

1. Resolve the user's attached image/video to its local Hermes attachment path. Ask for missing input; never invent a file path. URLs and live cameras are not supported in v1.
2. Choose `yolo_predict` with detect (boxes), segment (individual masks), semantic (scene class map), depth (estimated depth), classify (whole image), pose (keypoints), or obb (rotated boxes). Use `yolo_track` for identities/trajectories in a video. Defaults choose small pretrained models; no need to ask the user for a checkpoint.
3. Fixed models only recognize their trained classes. Use `yolo_info(model=..., class_query=...)` to inspect them. `classes` filters known names; `text_prompts` invokes YOLOE for categories like helmet. Text prompts support detect/segment only. Standard pose models describe human joints, and standard OBB models describe aerial categories.
4. Request `save_raw=true` when the user wants masks or depth arrays. Outputs include a visual preview, data ZIP, and optionally a raw-array ZIP. For a follow-up, reuse the original input and known model/settings; retrieve an existing job if its artifacts already answer the request.
5. If a tool returns running, retain its job_id and call `yolo_job` with `wait_seconds=30`. Do not restart the same job. If it returns busy, wait for your existing job or explain the occupied worker. Never cancel another request merely to free capacity. On a completion notification, first check whether this job's request was already fulfilled in the conversation; retrieve and deliver only outstanding results. Never resend artifacts already provided merely because a notification arrived. A notification does not prove inference or delivery succeeded.
6. Inspect the actual result before answering. Explain failures or partial coverage. Empty detections mean the chosen model/threshold found none, not proof of absence. Confidence scores are model scores. Video detection counts sum frame observations, not unique people or objects; track IDs can fragment. Depth is an estimate and must not be described as a calibrated distance measurement.
7. Deliver requested artifacts using the exact `media` entries as standalone lines in the final response, outside JSON, inline code, or fenced code blocks. For a visual request send the annotated image/video; include the data/raw ZIP when useful or requested. Do not claim delivery merely because a file exists or a notification was accepted.
8. Annotated videos are silent. Sampled analyses leave unsampled frames unannotated; report vid_stride and coverage. A user's cancellation request calls `yolo_job(action="cancel")` and checks the returned status.

## Everyday use

- Gym: extract poses or track a visible movement over a short clip. Ground any rep count in a sequence of keypoints and actual timestamps; ask for a better view when joints are occluded. Pose coordinates alone do not establish injury risk or personalized medical advice.
- Daily life: locate named belongings with YOLOE, produce cutouts, organize images, or summarize scene coverage. Use Hermes's existing vision for broader scene interpretation when YOLO outputs cannot answer it.
- Business: produce object locations, footage observations, segmentation assets, and track tables. Counts of observations or track IDs are not a guaranteed inventory, occupancy, or unique-customer total.

Read JSONL observations with Hermes's file/terminal tools when the compact first-frame preview cannot support the requested conclusion. The full timeline is in the data ZIP and the local observations artifact. Never infer an entire clip from its first-frame preview.
