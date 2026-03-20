"""API endpoint for pose detection on recorded clips."""

import logging

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from peewee import DoesNotExist

from frigate.api.auth import require_camera_access
from frigate.api.defs.tags import Tags
from frigate.config import FrigateConfig
from frigate.data_processing.common.pose.classifier import classify_pose
from frigate.data_processing.common.pose.model import PoseEstimator
from frigate.models import Recordings
from frigate.util.image import get_image_from_recording

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.media])

# Cached estimator (shared across requests)
_estimator: PoseEstimator | None = None
_estimator_model_size: str | None = None


def _get_estimator(config: FrigateConfig) -> PoseEstimator | None:
    """Get or create pose estimator using current config."""
    global _estimator, _estimator_model_size

    model_size = config.pose_detection.model_size
    device = config.pose_detection.device

    if _estimator is not None and _estimator_model_size == model_size:
        return _estimator

    estimator = PoseEstimator(model_size=model_size, device=device)
    if estimator.build():
        _estimator = estimator
        _estimator_model_size = model_size
        return _estimator

    return None


@router.get(
    "/{camera_name}/pose/{frame_time}",
    dependencies=[Depends(require_camera_access)],
    description="Run pose detection on a specific frame from a recording and return keypoints as JSON.",
)
async def pose_at_frame(
    request: Request,
    camera_name: str,
    frame_time: float,
):
    """Return pose keypoints for a specific frame timestamp."""
    config: FrigateConfig = request.app.frigate_config

    if not config.pose_detection.enabled:
        return JSONResponse(
            content={"success": False, "message": "Pose detection is not enabled"},
            status_code=400,
        )

    estimator = _get_estimator(config)
    if estimator is None:
        return JSONResponse(
            content={"success": False, "message": "Pose model not available"},
            status_code=500,
        )

    # Find the recording that contains this frame
    try:
        recording = (
            Recordings.select(Recordings.path, Recordings.start_time)
            .where(
                (frame_time >= Recordings.start_time)
                & (frame_time <= Recordings.end_time)
            )
            .where(Recordings.camera == camera_name)
            .order_by(Recordings.start_time.desc())
            .limit(1)
            .get()
        )
    except DoesNotExist:
        return JSONResponse(
            content={"success": False, "message": "Recording not found"},
            status_code=404,
        )

    # Extract frame from recording
    time_in_segment = frame_time - recording.start_time
    image_data = get_image_from_recording(
        config.ffmpeg, recording.path, time_in_segment, "png"
    )

    if not image_data:
        return JSONResponse(
            content={"success": False, "message": "Could not extract frame"},
            status_code=500,
        )

    # Decode image
    frame = cv2.imdecode(
        np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR
    )

    if frame is None:
        return JSONResponse(
            content={"success": False, "message": "Could not decode frame"},
            status_code=500,
        )

    # Run pose estimation
    keypoints = estimator.estimate(frame)

    if not keypoints:
        return JSONResponse(
            content={"success": True, "poses": []},
            status_code=200,
        )

    pose_cfg = config.pose_detection
    h, w = frame.shape[:2]

    # Normalize keypoints to 0-1 range (same as live overlay)
    norm_keypoints = [
        {"x": round(kx / w, 4), "y": round(ky / h, 4), "confidence": round(kc, 3)}
        for kx, ky, kc in keypoints
    ]

    # Classify
    result = classify_pose(
        keypoints,
        min_conf=pose_cfg.min_keypoint_score,
        enabled_poses=pose_cfg.poses,
    )

    pose_name = "unknown"
    pose_conf = 0.0
    if result:
        pose_name, pose_conf = result

    # Derive bounding box from keypoints (normalized)
    valid_pts = [
        (kp["x"], kp["y"]) for kp in norm_keypoints if kp["confidence"] >= pose_cfg.min_keypoint_score
    ]
    box = [0, 0, 1, 1]
    if len(valid_pts) >= 3:
        xs = [p[0] for p in valid_pts]
        ys = [p[1] for p in valid_pts]
        pad = 0.02
        box = [
            max(0, min(ys) - pad),
            max(0, min(xs) - pad),
            min(1, max(ys) + pad),
            min(1, max(xs) + pad),
        ]

    return JSONResponse(
        content={
            "success": True,
            "poses": [
                {
                    "keypoints": norm_keypoints,
                    "pose": pose_name,
                    "score": round(pose_conf, 3),
                    "box": box,
                }
            ],
        },
        status_code=200,
    )
