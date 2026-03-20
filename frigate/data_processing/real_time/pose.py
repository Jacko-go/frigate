"""Handle processing images for pose detection and classification."""

import datetime
import json
import logging
from collections import Counter, deque
from typing import Any, Optional

import cv2
import numpy as np

from frigate.comms.event_metadata_updater import (
    EventMetadataPublisher,
    EventMetadataTypeEnum,
)
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import FrigateConfig
from frigate.data_processing.common.pose.classifier import classify_pose
from frigate.data_processing.common.pose.model import PoseEstimator
from frigate.types import TrackedObjectUpdateTypesEnum
from frigate.util.builtin import EventsPerSecond, InferenceSpeed
from frigate.util.image import area

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)

MIN_CROP_PIXELS = 40  # skip tiny person crops
SMOOTHING_WINDOW = 5  # number of recent frames for majority-vote smoothing
SMOOTHING_THRESHOLD = 3  # min votes to accept a pose


class PoseRealTimeProcessor(RealTimeProcessorApi):
    def __init__(
        self,
        config: FrigateConfig,
        requestor: InterProcessRequestor,
        sub_label_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
    ):
        super().__init__(config, metrics)
        self.pose_config = config.pose_detection
        self.requestor = requestor
        self.sub_label_publisher = sub_label_publisher
        self.estimator = PoseEstimator(
            model_size=self.pose_config.model_size,
            device=self.pose_config.device,
        )
        self.model_ready = self.estimator.build()

        # Track pose history per object and cooldowns
        self.person_pose_history: dict[str, list[tuple[str, float]]] = {}
        self.person_pose_cooldown: dict[str, dict[str, float]] = {}

        # Smoothing buffer: last N raw classifications per tracked object
        # Each entry is (pose_name, confidence)
        self.pose_smooth_buffer: dict[str, deque] = {}

        self.poses_per_second = EventsPerSecond()
        self.inference_speed = InferenceSpeed(self.metrics.pose_speed)
        self.poses_per_second.start()

    CONFIG_UPDATE_TOPIC = "config/pose_detection"

    def update_config(self, topic: str, payload: Any) -> None:
        """Update pose detection config at runtime."""
        if topic != self.CONFIG_UPDATE_TOPIC:
            return

        self.config.pose_detection = payload
        self.pose_config = payload
        logger.debug("Pose detection config updated dynamically")

    def process_frame(self, obj_data: dict[str, Any], frame: np.ndarray):
        """Look for poses in detected person objects."""
        self.metrics.pose_fps.value = self.poses_per_second.eps()

        # Early exit for non-person objects and model not ready
        if obj_data.get("label", "unknown") != "person":
            return

        if not self.model_ready:
            return

        camera = obj_data["camera"]

        cam_config = (
            self.config.cameras[camera].pose_detection
            if camera in self.config.cameras
            else None
        )

        # Resolve enabled: per-camera explicit > global fallback
        if cam_config and cam_config.enabled is not None:
            cam_pose_enabled = cam_config.enabled
        else:
            cam_pose_enabled = self.config.pose_detection.enabled

        if not cam_pose_enabled:
            return

        # Resolve poses: per-camera list > global list
        cam_poses = (
            cam_config.poses
            if cam_config and cam_config.poses is not None
            else self.pose_config.poses
        )

        start = datetime.datetime.now().timestamp()
        obj_id = obj_data["id"]

        # Don't overwrite sub_label for objects that already have a non-pose sub_label
        if obj_data.get("sub_label") and obj_id not in self.person_pose_history:
            logger.debug(
                f"Skipping pose for {obj_id}: existing sub_label={obj_data.get('sub_label')}"
            )
            return



        # Check person score against threshold
        score = obj_data.get("score", 0.0)
        if score < self.pose_config.min_score:
            logger.debug(
                f"Skipping pose for {obj_id}: score {score:.2f} < min_score {self.pose_config.min_score}"
            )
            return

        # Get person bounding box
        person_box = obj_data.get("box")
        if not person_box:
            logger.debug(f"Skipping pose for {obj_id}: no bounding box")
            return

        # Check minimum area
        min_area = self.config.cameras[camera].pose_detection.min_area
        person_area = area(person_box)
        if person_area < min_area:
            logger.debug(
                f"Skipping pose for {obj_id}: area {person_area} < min_area {min_area}"
            )
            return

        # Extract person crop from frame
        left, top, right, bottom = person_box
        frame_h, frame_w = frame.shape[:2]

        # Convert from YUV to BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        # After YUV conversion the height is 2/3 of the YUV buffer
        rgb_h, rgb_w = rgb.shape[:2]

        # Expand crop by 20% on each side for full body capture
        box_w = right - left
        box_h = bottom - top
        pad_x = int(box_w * 0.2)
        pad_y = int(box_h * 0.2)

        # Clamp expanded box to frame bounds
        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(rgb_w, right + pad_x)
        bottom = min(rgb_h, bottom + pad_y)

        person_crop = rgb[top:bottom, left:right]

        if person_crop.size == 0 or person_crop.shape[0] < MIN_CROP_PIXELS or person_crop.shape[1] < MIN_CROP_PIXELS:
            logger.debug(f"Skipping pose for {obj_id}: crop too small ({person_crop.shape})")
            return

        logger.debug(f"Running pose estimation for {obj_id}: crop={person_crop.shape}, box={person_box}")

        # Run pose estimation
        keypoints = self.estimator.estimate(person_crop)

        if not keypoints:
            logger.debug(f"Pose estimation returned no keypoints for {obj_id}")
            self.__update_metrics(datetime.datetime.now().timestamp() - start)
            return

        # Always normalize and send keypoints for skeleton overlay
        norm_keypoints = []
        for kp in keypoints:
            nx = (left + kp[0]) / rgb_w
            ny = (top + kp[1]) / rgb_h
            norm_keypoints.append({"x": round(nx, 4), "y": round(ny, 4), "confidence": round(kp[2], 3)})

        norm_box = [
            round(top / rgb_h, 4),
            round(left / rgb_w, 4),
            round(bottom / rgb_h, 4),
            round(right / rgb_w, 4),
        ]

        # Classify pose from keypoints (optional for overlay)
        result = classify_pose(
            keypoints,
            min_conf=self.pose_config.min_keypoint_score,
            enabled_poses=cam_poses,
        )

        raw_pose = "unknown"
        raw_conf = 0.0
        if result:
            raw_pose, raw_conf = result

        # --- Temporal smoothing via majority vote ---
        if obj_id not in self.pose_smooth_buffer:
            self.pose_smooth_buffer[obj_id] = deque(maxlen=SMOOTHING_WINDOW)

        self.pose_smooth_buffer[obj_id].append((raw_pose, raw_conf))

        # Count votes (ignoring confidence for the vote)
        pose_votes = Counter(entry[0] for entry in self.pose_smooth_buffer[obj_id])
        top_pose, top_count = pose_votes.most_common(1)[0]

        # Only accept if it has enough votes (and isn't 'unknown')
        if top_pose != "unknown" and top_count >= SMOOTHING_THRESHOLD:
            pose_name = top_pose
            # Use average confidence from frames that matched the winning pose
            matching_confs = [
                conf for name, conf in self.pose_smooth_buffer[obj_id]
                if name == top_pose and conf > 0
            ]
            pose_conf = sum(matching_confs) / len(matching_confs) if matching_confs else 0.0
        else:
            pose_name = "unknown"
            pose_conf = 0.0

        logger.debug(
            f"Pose for {obj_id}: raw={raw_pose}({raw_conf:.2f}), "
            f"smoothed={pose_name}({pose_conf:.2f}), votes={dict(pose_votes)}"
        )

        # Send tracked object update for UI skeleton overlay (always)
        self.requestor.send_data(
            "tracked_object_update",
            json.dumps(
                {
                    "type": TrackedObjectUpdateTypesEnum.pose,
                    "pose": pose_name,
                    "score": round(pose_conf, 3),
                    "id": obj_id,
                    "camera": camera,
                    "timestamp": start,
                    "keypoints": norm_keypoints,
                    "box": norm_box,
                }
            ),
        )

        # Only publish sub_label and MQTT when classification meets min_pose_score
        if pose_name != "unknown" and pose_conf >= self.pose_config.min_pose_score:
            now = datetime.datetime.now().timestamp()
            if obj_id in self.person_pose_cooldown:
                last_time = self.person_pose_cooldown[obj_id].get(pose_name, 0)
                if now - last_time < self.pose_config.cooldown:
                    self.__update_metrics(now - start)
                    return

            if obj_id not in self.person_pose_history:
                self.person_pose_history[obj_id] = []
            self.person_pose_history[obj_id].append((pose_name, pose_conf))

            if obj_id not in self.person_pose_cooldown:
                self.person_pose_cooldown[obj_id] = {}
            self.person_pose_cooldown[obj_id][pose_name] = now

            self.sub_label_publisher.publish(
                (obj_id, pose_name, pose_conf),
                EventMetadataTypeEnum.sub_label.value,
            )

            self.requestor.send_data(
                f"{camera}/pose",
                json.dumps(
                    {
                        "person_id": obj_id,
                        "pose": pose_name,
                        "score": round(pose_conf, 3),
                        "camera": camera,
                        "timestamp": start,
                    }
                ),
            )

        self.__update_metrics(datetime.datetime.now().timestamp() - start)

    def handle_request(self, topic: str, request_data: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Handle pose-related requests."""
        return None

    def expire_object(self, object_id: str, camera: str):
        """Clean up tracking state when a person object expires."""
        self.person_pose_history.pop(object_id, None)
        self.person_pose_cooldown.pop(object_id, None)
        self.pose_smooth_buffer.pop(object_id, None)

    def __update_metrics(self, duration: float) -> None:
        self.poses_per_second.update()
        self.inference_speed.update(duration)
