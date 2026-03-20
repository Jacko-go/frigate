"""Pose estimation model — YOLO-Pose ONNX inference for COCO 17 keypoints."""

import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Expected input size for YOLO-Pose models
POSE_INPUT_SIZE = 640


class PoseEstimator:
    """Run YOLO-Pose ONNX model to extract 17 COCO keypoints from a person crop."""

    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        self.model_size = model_size
        self.device = device
        self.session = None
        self.input_name: str = ""
        self.input_shape: tuple = ()

    def build(self) -> bool:
        """Load the ONNX model. Returns True on success."""
        try:
            import onnxruntime as ort

            model_map = {
                "nano": "yolo11n-pose.onnx",
                "small": "yolo11s-pose.onnx",
                "medium": "yolo11m-pose.onnx",
                "large": "yolo11l-pose.onnx",
                "xlarge": "yolo11x-pose.onnx",
            }
            model_name = model_map.get(self.model_size, "yolo11n-pose.onnx")

            # Check multiple locations for the model
            search_paths = [
                os.path.join("/models", model_name),
                os.path.join(os.path.dirname(__file__), model_name),
                os.path.join("models", model_name),
            ]

            model_cache = os.environ.get(
                "MODEL_CACHE_DIR",
                os.path.join(os.path.expanduser("~"), ".cache", "frigate", "models"),
            )
            search_paths.append(os.path.join(model_cache, "pose", model_name))

            model_path = None
            for p in search_paths:
                if os.path.exists(p):
                    model_path = p
                    break

            if model_path is None:
                # Auto-download and export the model
                model_path = self._download_and_export(model_name, model_cache)
                if model_path is None:
                    return False

            providers = ["CPUExecutionProvider"]
            if self.device:
                # Explicit device override
                if self.device.upper() == "GPU":
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = [self.device, "CPUExecutionProvider"]
            else:
                # Auto-detect best available provider
                available = ort.get_available_providers()
                if "OpenVINOExecutionProvider" in available:
                    providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
                    logger.info("Pose: using OpenVINO acceleration")
                elif "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    logger.info("Pose: using CUDA acceleration")
                else:
                    logger.info("Pose: using CPU (no GPU provider available)")

            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            logger.info(f"Pose estimation model loaded: {model_path}")
            return True

        except ImportError:
            logger.error("onnxruntime is not installed. Pose detection unavailable.")
            return False
        except Exception as e:
            logger.error(f"Failed to load pose model: {e}")
            return False

    def _download_and_export(self, model_name: str, cache_dir: str) -> Optional[str]:
        """Download a pre-exported YOLO11 pose ONNX model from GitHub."""
        try:
            import urllib.request

            pose_cache = os.path.join(cache_dir, "pose")
            os.makedirs(pose_cache, exist_ok=True)
            onnx_path = os.path.join(pose_cache, model_name)

            # Ultralytics publishes ONNX models on GitHub releases
            pt_name = model_name.replace(".onnx", "")
            url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{pt_name}.onnx"

            logger.info(
                f"Pose model {model_name} not found locally. "
                f"Downloading from {url} ..."
            )

            urllib.request.urlretrieve(url, onnx_path)

            if os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 1_000_000:
                logger.info(f"Pose model downloaded and cached: {onnx_path}")
                return onnx_path
            else:
                # File too small — likely a 404 HTML page
                logger.error(
                    f"Downloaded file is too small, likely invalid. "
                    f"Try manually exporting with: "
                    f"python -c \"from ultralytics import YOLO; YOLO('{pt_name}.pt').export(format='onnx')\""
                )
                if os.path.exists(onnx_path):
                    os.remove(onnx_path)
                return None

        except Exception as e:
            logger.error(f"Failed to download pose model {model_name}: {e}")
            return None

    def estimate(
        self, frame: np.ndarray
    ) -> Optional[list[tuple[float, float, float]]]:
        """
        Run pose estimation on a BGR image — returns keypoints for the best detection.

        Args:
            frame: BGR numpy array (cropped person region).

        Returns:
            List of 17 (x, y, confidence) keypoint tuples in pixel coords
            relative to the input frame, or None on failure.
        """
        results = self.estimate_all(frame)
        if results:
            return results[0]
        return None

    def estimate_all(
        self, frame: np.ndarray, min_det_conf: float = 0.25
    ) -> list[list[tuple[float, float, float]]]:
        """
        Run pose estimation on a BGR image — returns keypoints for ALL detected people.

        Args:
            frame: BGR numpy array.
            min_det_conf: Minimum detection confidence to include a person.

        Returns:
            List of person detections, each being a list of 17 (x, y, confidence) keypoint tuples.
            Sorted by detection confidence (highest first). Empty list on failure.
        """
        if self.session is None:
            return []

        try:
            orig_h, orig_w = frame.shape[:2]

            # Preprocess: letterbox resize to model input size
            input_tensor, scale, pad_x, pad_y = self._preprocess(frame)

            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})

            # Parse keypoints from output
            return self._postprocess_all(
                outputs, orig_w, orig_h, scale, pad_x, pad_y, min_det_conf
            )

        except Exception as e:
            logger.debug(f"Pose estimation failed: {e}")
            return []

    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        """
        Letterbox resize and normalize for YOLO-Pose input.

        Returns:
            (input_tensor, scale, pad_x, pad_y)
        """
        h, w = frame.shape[:2]
        size = POSE_INPUT_SIZE

        # Calculate scale and padding for letterbox
        scale = min(size / w, size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pad_x = (size - new_w) / 2.0
        pad_y = (size - new_h) / 2.0

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        padded[top : top + new_h, left : left + new_w] = resized

        # Normalize to 0-1 and convert to NCHW float32
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, 0)  # Add batch dim

        return blob, scale, pad_x, pad_y

    def _extract_keypoints(
        self,
        output: np.ndarray,
        det_idx: int,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> list[tuple[float, float, float]]:
        """Extract 17 keypoints for a single detection."""
        kp_data = output[5:, det_idx]  # shape: (51,)

        keypoints = []
        for i in range(17):
            kp_x = kp_data[i * 3]
            kp_y = kp_data[i * 3 + 1]
            kp_conf = kp_data[i * 3 + 2]

            # Reverse letterbox transform to get original pixel coords
            x = (kp_x - pad_x) / scale
            y = (kp_y - pad_y) / scale

            # Clamp to original frame bounds
            x = max(0, min(x, orig_w - 1))
            y = max(0, min(y, orig_h - 1))

            keypoints.append((float(x), float(y), float(kp_conf)))

        return keypoints

    def _postprocess_all(
        self,
        outputs: list[np.ndarray],
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
        min_det_conf: float = 0.25,
    ) -> list[list[tuple[float, float, float]]]:
        """
        Parse YOLO-Pose output to extract keypoints for ALL valid detections.

        YOLO-Pose output shape: (1, 56, N) where N is number of proposals.
        Layout per proposal: [x_center, y_center, w, h, conf, kp0_x, kp0_y, kp0_conf, ..., kp16_x, kp16_y, kp16_conf]
        Total = 4 (box) + 1 (conf) + 17 * 3 (keypoints) = 56
        """
        output = outputs[0]  # shape: (1, 56, N)

        if output.ndim == 3:
            output = output[0]  # shape: (56, N)

        if output.shape[0] != 56:
            # Try transposing if needed
            if output.shape[1] == 56:
                output = output.T
            else:
                logger.debug(f"Unexpected pose model output shape: {output.shape}")
                return []

        num_proposals = output.shape[1]

        if num_proposals == 0:
            return []

        # Extract confidences (index 4)
        confidences = output[4, :]

        # Get all detections above threshold, sorted by confidence (highest first)
        valid_indices = np.where(confidences >= min_det_conf)[0]
        if len(valid_indices) == 0:
            return []

        # Sort by confidence descending
        sorted_indices = valid_indices[np.argsort(confidences[valid_indices])[::-1]]

        # Simple NMS: skip detections whose center is too close to an already-accepted one
        accepted = []
        accepted_centers = []
        nms_dist = min(orig_w, orig_h) * 0.1  # 10% of frame size

        for idx in sorted_indices:
            cx = float(output[0, idx] - pad_x) / scale
            cy = float(output[1, idx] - pad_y) / scale

            too_close = False
            for ax, ay in accepted_centers:
                if abs(cx - ax) < nms_dist and abs(cy - ay) < nms_dist:
                    too_close = True
                    break

            if not too_close:
                kps = self._extract_keypoints(output, int(idx), orig_w, orig_h, scale, pad_x, pad_y)
                accepted.append(kps)
                accepted_centers.append((cx, cy))

        return accepted
