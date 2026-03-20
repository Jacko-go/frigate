"""Handle processing images for pose detection and classification."""

import datetime
import json
import logging
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

MAX_POSE_ATTEMPTS = 8
MIN_CROP_PIXELS = 40  # skip tiny person crops


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
        self.person_attempt_count: dict[str, int] = {}

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
        camera = obj_data["camera"]

        if not self.model_ready:
            return

        if not self.config.cameras[camera].pose_detection.enabled:
            return

        start = datetime.datetime.now().timestamp()
        obj_id = obj_data["id"]

        # Only process person objects
        if obj_data.get("label") != "person":
            return

        # Don't overwrite sub_label for objects that already have a non-pose sub_label
        if obj_data.get("sub_label") and obj_id not in self.person_pose_history:
            logger.debug(
                f"Not processing pose due to existing sub label: {obj_data.get('sub_label')}."
            )
            return

        # Limit attempts per object
        attempt_count = self.person_attempt_count.get(obj_id, 0)
        if attempt_count >= MAX_POSE_ATTEMPTS:
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

        # Clamp box to frame bounds
        left = max(0, left)
        top = max(0, top)
        right = min(rgb_w, right)
        bottom = min(rgb_h, bottom)

        person_crop = rgb[top:bottom, left:right]

        if person_crop.size == 0 or person_crop.shape[0] < MIN_CROP_PIXELS or person_crop.shape[1] < MIN_CROP_PIXELS:
            logger.debug(f"Skipping pose for {obj_id}: crop too small ({person_crop.shape})")
            return

        # Run pose estimation
        keypoints = self.estimator.estimate(person_crop)

        if not keypoints:
            self.person_attempt_count[obj_id] = attempt_count + 1
            self.__update_metrics(datetime.datetime.now().timestamp() - start)
            return

        # Classify pose from keypoints
        result = classify_pose(
            keypoints,
            min_conf=self.pose_config.min_keypoint_score,
            enabled_poses=self.pose_config.poses,
        )

        self.person_attempt_count[obj_id] = attempt_count + 1

        if not result:
            self.__update_metrics(datetime.datetime.now().timestamp() - start)
            return

        pose_name, pose_conf = result

        # Check cooldown
        now = datetime.datetime.now().timestamp()
        if obj_id in self.person_pose_cooldown:
            last_time = self.person_pose_cooldown[obj_id].get(pose_name, 0)
            if now - last_time < self.pose_config.cooldown:
                self.__update_metrics(now - start)
                return

        # Store pose result
        if obj_id not in self.person_pose_history:
            self.person_pose_history[obj_id] = []

        self.person_pose_history[obj_id].append((pose_name, pose_conf))

        # Update cooldown
        if obj_id not in self.person_pose_cooldown:
            self.person_pose_cooldown[obj_id] = {}
        self.person_pose_cooldown[obj_id][pose_name] = now

        logger.debug(
            f"Detected pose '{pose_name}' (conf={pose_conf:.2f}) for {obj_id} on {camera}"
        )

        # Publish tracked object update for UI
        # Normalize keypoints to full frame (0-1 range)
        norm_keypoints = []
        for kp in keypoints:
            nx = (left + kp[0]) / rgb_w
            ny = (top + kp[1]) / rgb_h
            norm_keypoints.append({"x": round(nx, 4), "y": round(ny, 4), "confidence": round(kp[2], 3)})

        # Normalize bounding box to 0-1 range [y1, x1, y2, x2]
        norm_box = [
            round(top / rgb_h, 4),
            round(left / rgb_w, 4),
            round(bottom / rgb_h, 4),
            round(right / rgb_w, 4),
        ]

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

        # Publish sub_label update (pose as sub_label on person events)
        self.sub_label_publisher.publish(
            (obj_id, pose_name, pose_conf),
            EventMetadataTypeEnum.sub_label.value,
        )

        # Publish dedicated pose MQTT topic
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
        self.person_attempt_count.pop(object_id, None)

    def __update_metrics(self, duration: float) -> None:
        self.poses_per_second.update()
        self.inference_speed.update(duration)
