#!/usr/bin/env python3
"""Pose detection test tool — run the EXACT Frigate pose pipeline on a video file.

Reads your config.yml to use the same model, thresholds, and poses as live Frigate.

Usage:
    # Use settings from Frigate config (default: /config/config.yml)
    python tools/pose_test.py input.mp4 --config /path/to/config.yml

    # Override config path + save output
    python tools/pose_test.py input.mp4 -c config.yml -o output.mp4

    # Headless mode (no GUI, just save output — for running on the NUC over SSH)
    python tools/pose_test.py input.mp4 -c config.yml -o output.mp4 --headless

Controls (live preview):
    q / ESC  = quit
    SPACE    = pause/resume
    s        = save current frame as PNG
"""

import argparse
import os
import sys
from collections import Counter, deque

import cv2
import numpy as np

# Add project root to path so we can import frigate modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from frigate.data_processing.common.pose.classifier import classify_pose
from frigate.data_processing.common.pose.model import PoseEstimator

# COCO skeleton connections for drawing
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # head
    (5, 6),                                  # shoulders
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 11), (6, 12),                        # torso
    (11, 12),                                # hips
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]

POSE_COLORS = {
    "hands_up": (0, 0, 255),       # red
    "t_pose": (255, 0, 255),       # magenta
    "waving": (0, 255, 255),       # yellow
    "left_hand_up": (255, 165, 0), # orange
    "right_hand_up": (255, 165, 0),
    "sitting": (255, 255, 0),      # cyan
    "standing": (0, 255, 0),       # green
    "lying_down": (0, 0, 200),     # dark red
    "crouching": (200, 100, 0),    # teal
    "pointing": (255, 0, 0),       # blue
    "unknown": (128, 128, 128),    # gray
}

# Same smoothing constants as pose.py
SMOOTHING_WINDOW = 5
SMOOTHING_THRESHOLD = 3


def load_pose_config(config_path: str) -> dict:
    """Load pose_detection settings from a Frigate config.yml file."""
    try:
        import yaml
    except ImportError:
        # Try ruamel.yaml (what Frigate uses)
        from ruamel.yaml import YAML
        yaml_loader = YAML()
        with open(config_path) as f:
            raw = yaml_loader.load(f)
    else:
        with open(config_path) as f:
            raw = yaml.safe_load(f)

    pose_cfg = raw.get("pose_detection", {})

    config = {
        "model_size": pose_cfg.get("model_size", "nano"),
        "device": pose_cfg.get("device", None),
        "min_score": pose_cfg.get("min_score", 0.5),
        "min_keypoint_score": pose_cfg.get("min_keypoint_score", 0.3),
        "min_pose_score": pose_cfg.get("min_pose_score", 0.7),
        "min_area": pose_cfg.get("min_area", 2000),
        "poses": pose_cfg.get("poses", [
            "hands_up", "t_pose", "waving", "left_hand_up", "right_hand_up",
            "sitting", "standing", "lying_down", "crouching", "pointing",
        ]),
        "cooldown": pose_cfg.get("cooldown", 5),
    }

    return config


def draw_skeleton(frame, keypoints, box, pose_name, pose_conf, min_conf, raw_pose, raw_conf):
    """Draw skeleton overlay, bounding box, and pose label on frame."""
    h, w = frame.shape[:2]
    color = POSE_COLORS.get(pose_name, (128, 128, 128))

    # Draw bounding box
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw pose label
    if pose_name != "unknown":
        label = f"{pose_name} ({pose_conf:.0%})"
    else:
        label = "detecting..."
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw raw classification (smaller, below box)
    raw_label = f"raw: {raw_pose} ({raw_conf:.0%})"
    cv2.putText(frame, raw_label, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Draw keypoints and skeleton
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


def process_video(args, pose_cfg):
    """Process a video with the exact Frigate pose pipeline."""

    print("\n=== Pose Detection Config (from config.yml) ===")
    for k, v in pose_cfg.items():
        print(f"  {k}: {v}")
    print()

    # Load model (same as Frigate)
    print(f"Loading pose model: {pose_cfg['model_size']}...")
    estimator = PoseEstimator(
        model_size=pose_cfg["model_size"],
        device=pose_cfg["device"],
    )
    if not estimator.build():
        print("ERROR: Failed to load pose model!")
        sys.exit(1)
    print("Model loaded.\n")

    # Open video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Output writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Output: {args.output}")

    # Smoothing (same as pose.py)
    smooth_buffers: dict[str, deque] = {}

    frame_idx = 0
    paused = False
    pose_counts: dict[str, int] = {}

    print("\nProcessing...")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

        display = frame.copy()

        # Run pose estimation (same model as Frigate)
        keypoints = estimator.estimate(frame)

        if keypoints:
            # Derive bounding box from keypoints
            valid_pts = [
                (kx, ky) for kx, ky, kc in keypoints
                if kc >= pose_cfg["min_keypoint_score"]
            ]

            if len(valid_pts) >= 5:
                xs = [p[0] for p in valid_pts]
                ys = [p[1] for p in valid_pts]
                pad = 20
                box = (
                    max(0, int(min(xs)) - pad),
                    max(0, int(min(ys)) - pad),
                    min(width, int(max(xs)) + pad),
                    min(height, int(max(ys)) + pad),
                )

                box_area = (box[2] - box[0]) * (box[3] - box[1])

                if box_area >= pose_cfg["min_area"]:
                    # Classify (same function + params as Frigate)
                    result = classify_pose(
                        keypoints,
                        min_conf=pose_cfg["min_keypoint_score"],
                        enabled_poses=pose_cfg["poses"],
                    )

                    raw_pose = result[0] if result else "unknown"
                    raw_conf = result[1] if result else 0.0

                    # Smoothing (same logic as pose.py)
                    obj_key = "person_0"
                    if obj_key not in smooth_buffers:
                        smooth_buffers[obj_key] = deque(maxlen=SMOOTHING_WINDOW)
                    smooth_buffers[obj_key].append(raw_pose)
                    counts = Counter(smooth_buffers[obj_key])
                    top_pose, top_count = counts.most_common(1)[0]

                    if top_pose != "unknown" and top_count >= SMOOTHING_THRESHOLD:
                        pose_name = top_pose
                        pose_conf = raw_conf
                    else:
                        pose_name = "unknown"
                        pose_conf = 0.0

                    # Apply min_pose_score (same as Frigate)
                    if pose_name != "unknown" and pose_conf < pose_cfg["min_pose_score"]:
                        pose_name = "unknown"
                        pose_conf = 0.0

                    draw_skeleton(display, keypoints, box, pose_name, pose_conf,
                                  pose_cfg["min_keypoint_score"], raw_pose, raw_conf)

                    # Track stats
                    if pose_name != "unknown":
                        pose_counts[pose_name] = pose_counts.get(pose_name, 0) + 1

        # HUD
        progress = f"Frame {frame_idx}/{total_frames}"
        cv2.putText(display, progress, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cfg_info = f"model={pose_cfg['model_size']}  min_area={pose_cfg['min_area']}  min_kp={pose_cfg['min_keypoint_score']}  min_pose={pose_cfg['min_pose_score']}"
        cv2.putText(display, cfg_info, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        if paused:
            cv2.putText(display, "PAUSED", (width // 2 - 50, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if writer:
            writer.write(display)

        if not args.headless:
            cv2.imshow("Pose Detection Test", display)
            key = cv2.waitKey(1 if not paused else 0) & 0xFF

            if key == ord("q") or key == 27:
                break
            elif key == ord(" "):
                paused = not paused
            elif key == ord("s"):
                save_path = f"pose_frame_{frame_idx}.png"
                cv2.imwrite(save_path, display)
                print(f"  Saved: {save_path}")

    cap.release()
    if writer:
        writer.release()
        print(f"\nOutput saved: {args.output}")
    cv2.destroyAllWindows()

    # Print stats
    print(f"\n=== Results ({frame_idx} frames) ===")
    if pose_counts:
        for pose, count in sorted(pose_counts.items(), key=lambda x: -x[1]):
            pct = count / frame_idx * 100
            print(f"  {pose:20s}: {count:5d} frames ({pct:.1f}%)")
    else:
        print("  No poses detected.")


def main():
    parser = argparse.ArgumentParser(
        description="Test pose detection on video files using exact Frigate settings"
    )
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output video file path")
    parser.add_argument("-c", "--config", default="/config/config.yml",
                        help="Path to Frigate config.yml (default: /config/config.yml)")
    parser.add_argument("--headless", action="store_true",
                        help="No preview window (for SSH/servers, requires -o)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    if args.headless and not args.output:
        print("ERROR: --headless requires -o/--output")
        sys.exit(1)

    # Load config
    if os.path.exists(args.config):
        print(f"Loading config from: {args.config}")
        pose_cfg = load_pose_config(args.config)
    else:
        print(f"WARNING: Config not found at {args.config}, using defaults")
        pose_cfg = load_pose_config.__defaults__  # won't work, use fallback
        pose_cfg = {
            "model_size": "nano", "device": None, "min_score": 0.5,
            "min_keypoint_score": 0.3, "min_pose_score": 0.7,
            "min_area": 2000, "cooldown": 5,
            "poses": ["hands_up", "t_pose", "waving", "left_hand_up",
                       "right_hand_up", "sitting", "standing", "lying_down",
                       "crouching", "pointing"],
        }

    process_video(args, pose_cfg)


if __name__ == "__main__":
    main()
