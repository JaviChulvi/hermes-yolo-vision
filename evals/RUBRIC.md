# Frozen grading rubric

Apply this checklist without arm identifiers. Review files and final answers; never score from an agent's assertion of success. Record per-check pass/fail and evidence paths. Unparseable output may be inspected manually using the same thresholds; document any manual normalization. A case passes only when every applicable required check passes. Timeouts, budget exhaustion, missing requested artifacts, unsupported claims, and a missing requested follow-up fail. No partial credit enters completion rate.

These are integration/user-task criteria for these fixtures, not a general model-accuracy benchmark. In particular, semantic/depth cases measure delivery and validity where pixel ground truth is unavailable. Several cases reuse images; report descriptive paired outcomes without an independence/significance claim.

## Common requirements

- All requested files exist, are nonempty, decode, and are referenced in the final answer. Artifacts must belong to this run. The annotated image/video must differ from the source and show the requested overlay; a source copy alone fails.
- Image coordinates match the displayed input dimensions after orientation. Videos retain their displayed aspect ratio and duration within max(0.15 seconds, 2 source frames), excluding explicitly disclosed padding. Silent output is acceptable only if not falsely described as retaining audio.
- Numerical outputs include the requested task fields and finite values. Confidence scores are in [0,1]. Negative/empty results must be represented truthfully. Arrays must have a documented spatial mapping to the original image; simply reshaping mismatched dimensions fails.
- A video timeline must have strictly increasing timestamps and cover at least 90% of source duration, beginning within one analyzed-frame interval. The analysis sampling interval must be disclosed. Workout repetition cases require at least 5 analyzed frames/second; missing a short event is still a failure, even when sampling is disclosed.
- Do not claim exact object absence from an unsupported class, perfect identity tracking, calibrated monocular depth, a medical diagnosis, or safe exercise form from these outputs.
- A valid alternative implementation receives the same score. Tool selection is constrained only when the user explicitly requests a classifier or text-prompted model.

## Task-specific checks

**Detection and follow-up filter.** Match same-class predicted boxes one-to-one to the frozen COCO boxes in annotations.json at IoU >=0.5. Required class counts must be exact; every required instance must match. Extra classes the user did not ask to filter out are acceptable if their reported counts match saved predictions. Final counts must agree with exported detections. The follow-up must give the original umbrella box/score without a second model inference.

**Instance segmentation.** Each requested person/dog must have a raw pixel mask and outline plus an overlay. Decode COCO polygons from annotations.json with pycocotools; match same-class masks one-to-one at IoU >=0.5, after restoring any documented mask resize to source dimensions with nearest-neighbor interpolation. Require exactly one requested instance for each selected fixture and at least 99% finite array entries. A box-filled rectangle is not accepted as a silhouette.

**Semantic segmentation.** Require an integer class-ID mask, label names, a colored preview, and per-class area shares. Map shape must correspond to the source geometry. At least two distinct classes must have nonzero area; at least 99% of pixels must map to declared classes. Shares sum to 1 within 0.01 and agree with pixel frequencies within 0.01 per class. For semantic-field, a grass/field/lawn class must occupy at least 0.05 of the image; for semantic-outdoors, vegetation/tree/grass classes together occupy at least 0.10. These broad presence labels were fixed from visual inspection; no pixel-accuracy claim is made.

**Depth.** Require a numerical depth array, a spatial mapping, preview, and summary. At least 99% of entries must be finite and positive; max must exceed min. Reported min/median/max must agree with the supplied array within relative tolerance 0.01. Clearly qualify distances as estimates. No accuracy threshold is claimed without calibrated ground truth.

**Classification.** Require at least three ranked labels with finite probabilities/scores, nonincreasing scores, and a labeled preview. The expected label family in cases.py must appear in the top five. The final answer must agree with the ranking; the dog breed remains a model guess.

**Pose.** Require 17 named or COCO-ordered keypoints per detected person, confidence/visibility and a skeleton overlay. On pose-photo, compare the visible ground-truth joints in annotations.json: at least 60% must lie within 0.15 of the person's ground-truth box diagonal. Hidden joints must not be asserted precise. On pose-workout, at least 80% of analyzed frames must contain a person with six or more confidence>=0.25 joints. This video check establishes usable timeline coverage, not validated biomechanics.

**Oriented boxes.** Require four finite corners per object, class and confidence. Match same-class quadrilaterals to the DOTA polygons in annotations.json at polygon IoU >=0.5. Each selected image contains one labeled plane/baseball diamond: that object must match. Corners must form a non-self-intersecting, positive-area quadrilateral. Rotated boxes may extend beyond the frame for partially visible objects; compare the uncut polygons consistently.

**Tracking.** At least 90% of analyzed frames must contain the visible person. One consistent ID must cover at least 80% of analyzed frames; an ID cannot represent multiple people in one frame. Track-workout additionally needs four completed squats supported by a joint/angle timeline, with each completion within 0.5s of [2.2,4.0,5.7,7.3]. Track-presence must report one simultaneous person and distinguish this from accumulated detections and estimated distinct tracks.

**Text prompts.** The user explicitly asks for text-prompted inference. Verify the executed model accepted the requested text rather than renaming fixed class IDs. Umbrella/vase localization must match the corresponding COCO box at IoU >=0.5 and supply the requested outline. Color adjectives are part of the query, not proof the model understands every attribute.

**Timeline follow-up.** Require four complete pushups with approximate completions within 0.5s of [2,4,6,8]. Evidence must come from the first turn's ordered pose/angle data, without another inference. Do not accept an answer derived solely from one preview frame or a filename.

## Negative and recovery cases

- **Missing:** ask for the absent attachment; no invented path/count or claim of a successful analysis.
- **Corrupt:** report inability to decode the supplied bytes and ask for a usable file; no fabricated annotation or count. A failed attempted inference is acceptable.
- **Distance:** refuse to certify exact/safe clearance from one uncalibrated image; explain the need for calibration or measurement. An explicitly qualified estimated depth visualization is optional.
- **Recovery:** identify that the fixed detector has no helmet category, avoid claiming confirmed absence, and offer text prompts/custom weights or careful visual review. Running YOLOE is optional because the user asks for a suitable alternative, not mandatory execution.

## Recording the verdict

Persist automated checks and blinded manual reviews separately. Mark unreviewed cases pending, never passed. A model-inference parity test cannot replace these checks. Report all attempted cases, quota stops, failed/timeout pairs and remaining coverage. Fixes made after held-out results require a new version and a new held-out set or an explicitly labeled retest; never rewrite the original result.
