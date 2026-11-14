---
name: yolo-vision
description: Analyze images and videos with YOLO; deliver grounded results.
---

# YOLO Vision

Use the `yolo_vision` tools for measured visual observations and reusable artifacts.

1. Resolve the user's attached image/video to its local Hermes attachment path. Ask for missing input; never invent a file path. URLs and live cameras are not supported in v1.
2. Choose `yolo_predict` with detect (boxes), segment (individual masks), semantic (scene class map), depth (estimated depth), classify (whole image), pose (keypoints), or obb (rotated boxes). Use `yolo_track` for identities/trajectories in a video. Defaults choose small pretrained models; no need to ask the user for a checkpoint.
3. Fixed models only recognize their trained classes. Use `yolo_info(model=..., class_query=...)` to inspect them. `classes` filters known names; `text_prompts` invokes YOLOE for categories like helmet. Text prompts support detect/segment only. Standard pose models describe human joints, and standard OBB models describe aerial categories.
4. Request `save_raw=true` when the user wants pixel masks, class-ID maps, or depth arrays. A silhouette export includes its pixel mask and outline; a scene segmentation map includes integer class IDs and their labels alongside the colored preview. Use the full observations and raw files for these exports. Outputs include a visual preview, data ZIP, and optionally a raw-array ZIP. For a follow-up, reuse the original input and known model/settings; retrieve an existing job if its artifacts already answer the request.
5. If a tool returns running, retain its job_id and call `yolo_job` with `wait_seconds=30`. Do not restart the same job. If it returns busy, wait for your existing job or explain the occupied worker. Never cancel another request merely to free capacity. On a completion notification, first check whether this job's request was already fulfilled in the conversation; retrieve and deliver only outstanding results. Never resend artifacts already provided merely because a notification arrived. A notification does not prove inference or delivery succeeded.
6. Inspect the actual result before answering. Explain failures or partial coverage. Empty detections mean the chosen model/threshold found none, not proof of absence. When checking a disputed identification, do not limit verification to suggested alternatives; inspect enough context and leave it unresolved if the evidence is insufficient. Nearby or aligned features alone do not establish a physical connection or mechanism. Confidence scores are model scores. Video detection counts sum frame observations, not unique people or objects; track IDs can fragment. Depth is an estimate and must not be described as a calibrated distance measurement.
7. Deliver requested artifacts using the exact `media` entries as standalone lines in the final response, outside JSON, inline code, or fenced code blocks. Prefer the returned annotated image/video; when adding joint labels, preserve the connecting skeleton lines. Include the data/raw ZIP when useful or requested, and deliver every requested data file directly. Do not claim delivery merely because a file exists or a notification was accepted.
8. Annotated videos are silent. Sampled analyses leave unsampled frames unannotated; report vid_stride and coverage. A user's cancellation request calls `yolo_job(action="cancel")` and checks the returned status.

## Everyday use

- Gym: extract poses or track a visible movement; when counting repetitions, use the workflow below. Pose coordinates alone do not establish injury risk or personalized medical advice.
- Daily life: locate named belongings with YOLOE, produce cutouts, organize images, or summarize scene coverage. Use Hermes's existing vision for broader scene interpretation when YOLO outputs cannot answer it.
- Business: produce object locations, footage observations, segmentation assets, and track tables. Counts of observations or track IDs are not a guaranteed inventory, occupancy, or unique-customer total.

Read the full JSONL timeline using Hermes's file/terminal tools whenever the compact preview cannot support the requested conclusion. The timeline is in the data ZIP and local observations artifact. Never infer an entire clip from its first-frame preview.

## Counts over time

For video, choose a sampling interval suited to the speed of the event and available resources; report the coverage and do not claim events that sampling cannot resolve. Start from the saved timeline, combine analysis and event-table creation in one pass, and save intermediate evidence before targeted source checks to verify candidate events and resolve ambiguities. Leave time to deliver a qualified answer and requested artifacts.

Follow the same subject using track IDs or documented geometric association, checking continuity across occlusions and ID changes. Changing per-frame confidence or box-size rankings do not establish identity. Treat missing, low-confidence or implausibly jumping landmarks as gaps, not motion. Describe the method actually used: framewise pose detection is not identity tracking.

Define the visible event before counting. Save a compact event table with subject, event-specific timestamps (for cycles: start, reversal and completion), status (complete, partial or uncertain), and supporting observations. Count distinct supported events from that table; deduplicate overlapping inspection windows. Preparation, repositioning, holds and recovery movements are not extra events unless they meet the definition. Derive timing summaries from the same table.

Verify candidate events against the original footage. Inspect uncertain transitions with a crop and sampling interval that resolve the relevant movement; preserve original timestamps when extracting clips. Do not count across an unresolved gap. Distinguish a visible completed cycle from technique criteria the view cannot establish; an obscured endpoint is not proof of a partial movement. When giving or revising a count, state any relevant unverified endpoint or completion standard alongside it. Describe equipment only as specifically as visible evidence supports.

For corrections and follow-ups, reuse the saved observations and event table. Reconcile conflicting evidence before giving another exact count; report unresolved events explicitly and carry forward relevant uncertainties, or request a better view. Do not simply adopt a suggested count or announce a definitive correction while verification is still pending.
