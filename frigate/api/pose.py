"""API endpoint for pose detection on recorded clips."""

import logging
import os
import subprocess as sp
import tempfile

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from peewee import DoesNotExist

from frigate.api.auth import require_camera_access
from frigate.api.defs.tags import Tags
from frigate.config import FrigateConfig
from frigate.data_processing.common.pose.classifier import classify_pose
from frigate.data_processing.common.pose.model import PoseEstimator
from frigate.models import Recordings

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.media])

# COCO skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

POSE_COLORS = {
    "hands_up": (0, 0, 255), "t_pose": (255, 0, 255),
    "waving": (0, 255, 255), "left_hand_up": (255, 165, 0),
    "right_hand_up": (255, 165, 0), "sitting": (255, 255, 0),
    "standing": (0, 255, 0), "lying_down": (0, 0, 200),
    "crouching": (200, 100, 0), "pointing": (255, 0, 0),
    "unknown": (128, 128, 128),
}

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


def _draw_pose(frame, keypoints, pose_name, pose_conf, min_conf):
    """Draw skeleton and pose label on frame."""
    color = POSE_COLORS.get(pose_name, (128, 128, 128))
    h, w = frame.shape[:2]

    valid_kps = {}
    for i, (kx, ky, kc) in enumerate(keypoints):
        if kc >= min_conf:
            px, py = int(kx), int(ky)
            valid_kps[i] = (px, py)
            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)
            cv2.circle(frame, (px, py), 4, (0, 0, 0), 1)

    for i, j in SKELETON:
        if i in valid_kps and j in valid_kps:
            cv2.line(frame, valid_kps[i], valid_kps[j], (0, 200, 200), 2)

    # Derive box from keypoints
    if len(valid_kps) >= 3:
        pts = list(valid_kps.values())
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 15
        x1 = max(0, min(xs) - pad)
        y1 = max(0, min(ys) - pad)
        x2 = min(w, max(xs) + pad)
        y2 = min(h, max(ys) + pad)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if pose_name != "unknown":
            label = f"{pose_name} ({pose_conf:.0%})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 8), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


@router.get(
    "/{camera_name}/start/{start_ts}/end/{end_ts}/pose_clip.mp4",
    dependencies=[Depends(require_camera_access)],
    description="Returns a recording clip with pose detection skeleton overlay.",
)
async def pose_clip(
    request: Request,
    camera_name: str,
    start_ts: float,
    end_ts: float,
):
    """Stream a recording clip with pose overlay applied."""
    config: FrigateConfig = request.app.frigate_config

    # Get estimator
    estimator = _get_estimator(config)
    if estimator is None:
        return JSONResponse(
            content={"success": False, "message": "Pose model not available"},
            status_code=500,
        )

    # Find recordings in time range
    recordings = (
        Recordings.select(Recordings.path, Recordings.start_time, Recordings.end_time)
        .where(
            (Recordings.start_time.between(start_ts, end_ts))
            | (Recordings.end_time.between(start_ts, end_ts))
            | ((start_ts > Recordings.start_time) & (end_ts < Recordings.end_time))
        )
        .where(Recordings.camera == camera_name)
        .order_by(Recordings.start_time.asc())
    )

    if recordings.count() == 0:
        return JSONResponse(
            content={"success": False, "message": "No recordings found"},
            status_code=404,
        )

    pose_cfg = config.pose_detection
    min_kp = pose_cfg.min_keypoint_score
    min_pose = pose_cfg.min_pose_score
    enabled_poses = pose_cfg.poses
    min_area = pose_cfg.min_area

    # Smoothing state
    from collections import Counter, deque
    smooth_buffer = deque(maxlen=5)

    def generate_annotated():
        """Read recordings, annotate with pose, encode to mp4."""
        for recording in recordings:
            cap = cv2.VideoCapture(recording.path)
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            rec_start = recording.start_time

            # Skip frames before start_ts
            if rec_start < start_ts:
                skip_secs = start_ts - rec_start
                cap.set(cv2.CAP_PROP_POS_MSEC, skip_secs * 1000)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if past end_ts
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                current_ts = rec_start + pos_ms / 1000
                if current_ts > end_ts:
                    break

                # Run pose estimation
                keypoints = estimator.estimate(frame)
                if keypoints:
                    valid_pts = [
                        (kx, ky) for kx, ky, kc in keypoints if kc >= min_kp
                    ]
                    if len(valid_pts) >= 5:
                        xs = [p[0] for p in valid_pts]
                        ys = [p[1] for p in valid_pts]
                        box_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

                        if box_area >= min_area:
                            result = classify_pose(
                                keypoints,
                                min_conf=min_kp,
                                enabled_poses=enabled_poses,
                            )
                            raw_pose = result[0] if result else "unknown"
                            raw_conf = result[1] if result else 0.0

                            smooth_buffer.append(raw_pose)
                            counts = Counter(smooth_buffer)
                            top_pose, top_count = counts.most_common(1)[0]

                            if top_pose != "unknown" and top_count >= 3:
                                pose_name = top_pose
                                pose_conf = raw_conf
                            else:
                                pose_name = "unknown"
                                pose_conf = 0.0

                            if pose_conf < min_pose:
                                pose_name = "unknown"
                                pose_conf = 0.0

                            _draw_pose(frame, keypoints, pose_name, pose_conf, min_kp)

                yield frame

            cap.release()

    # Use ffmpeg to encode annotated frames to streamable mp4
    frames = generate_annotated()
    first_frame = next(frames, None)
    if first_frame is None:
        return JSONResponse(
            content={"success": False, "message": "Could not read recording frames"},
            status_code=500,
        )

    h, w = first_frame.shape[:2]

    def stream_mp4():
        ffmpeg_cmd = [
            config.ffmpeg.ffmpeg_path,
            "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", "15",
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-movflags", "frag_keyframe+empty_moov",
            "-f", "mp4",
            "pipe:1",
        ]

        proc = sp.Popen(
            ffmpeg_cmd,
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
        )

        try:
            # Write first frame
            proc.stdin.write(first_frame.tobytes())

            # Write remaining frames
            for frame in frames:
                proc.stdin.write(frame.tobytes())

            proc.stdin.close()

            while True:
                data = proc.stdout.read(8192)
                if not data:
                    break
                yield data
        except BrokenPipeError:
            pass
        finally:
            proc.terminate()
            proc.wait()

    return StreamingResponse(
        stream_mp4(),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f"inline; filename=pose_{camera_name}_{int(start_ts)}.mp4",
        },
    )
