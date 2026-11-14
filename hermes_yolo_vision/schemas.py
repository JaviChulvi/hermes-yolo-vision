"""Small, explicit model-facing interface."""

DEFAULT_MODELS = {
    "detect": "yolo26n.pt",
    "segment": "yolo26n-seg.pt",
    "semantic": "yolo26n-sem-ade20k.pt",
    "depth": "yolo26n-depth.pt",
    "classify": "yolo26n-cls.pt",
    "pose": "yolo26n-pose.pt",
    "obb": "yolo26n-obb.pt",
}
TEXT_MODEL = "yoloe-26s-seg.pt"
TRACK_TASKS = ("detect", "segment", "pose", "obb")


def schema(name, description, properties, required=()):
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
    }


MODEL = {
    "type": "string",
    "description": "Official YOLO checkpoint name or trusted local .pt file.",
}
COMMON = {
    "source": {
        "type": "string",
        "description": "Local image/video file or downloaded Hermes attachment path. No URLs, cameras, directories, or globs.",
    },
    "task": {
        "type": "string",
        "enum": list(DEFAULT_MODELS),
        "description": "Infer from explicit weights; otherwise detect. Text prompts default to segment.",
    },
    "model": MODEL,
    "classes": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 100,
        "description": "Exact class names from this fixed-vocabulary model. Use text_prompts for new categories.",
    },
    "text_prompts": {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "maxItems": 80,
        "description": "YOLOE categories, e.g. person or helmet. Detect/segment only.",
    },
    "conf": {"type": "number", "minimum": 0, "maximum": 1},
    "imgsz": {"type": "integer", "minimum": 32, "maximum": 2048},
    "device": {"type": "string", "description": "cpu (default), mps, or a CUDA device index."},
    "vid_stride": {
        "type": "integer",
        "minimum": 1,
        "maximum": 120,
        "description": "Analyze every Nth frame; default 1. Sampling can miss events.",
    },
    "save_raw": {
        "type": "boolean",
        "description": "Export masks/depth arrays in a ZIP when requested; default false.",
    },
    "save_annotated": {
        "type": "boolean",
        "description": "Save an annotated image or silent video; default true.",
    },
}
SCHEMAS = [
    schema(
        "yolo_info",
        "Check YOLO readiness and defaults. Supply model to load its task/classes in a job; no model means no downloads.",
        {
            "model": MODEL,
            "class_query": {
                "type": "string",
                "description": "Filter model class names by substring.",
            },
        },
    ),
    schema(
        "yolo_predict",
        "Analyze an image or finite video with YOLO. Returns grounded observations and saved artifacts, or a job_id to poll. Use for boxes, masks, scene classes, depth, classification, pose, or rotated boxes.",
        COMMON,
        ["source"],
    ),
    schema(
        "yolo_track",
        "Track objects across a finite video, returning IDs, timestamped observations and annotated video. IDs are estimated tracks, not guaranteed unique real-world objects.",
        {**COMMON, "task": {"type": "string", "enum": list(TRACK_TASKS)}},
        ["source"],
    ),
    schema(
        "yolo_job",
        "Wait for, retrieve, or cancel a previously returned YOLO job. Reuse the job_id; do not start the same analysis again. Artifacts can be sent using standalone MEDIA lines.",
        {
            "job_id": {"type": "string"},
            "action": {"type": "string", "enum": ["status", "result", "cancel"]},
            "wait_seconds": {"type": "number", "minimum": 0, "maximum": 30},
        },
        ["job_id"],
    ),
]
