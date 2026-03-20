"""Pose keypoint classifier — classifies COCO 17-keypoint skeletons into named poses."""

import math
from typing import Optional

# COCO 17 keypoint indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def _kp_valid(keypoints: list, idx: int, min_conf: float) -> bool:
    """Check if a keypoint has sufficient confidence."""
    return keypoints[idx][2] >= min_conf


def _angle(p1: tuple, p2: tuple, p3: tuple) -> float:
    """Calculate the angle at p2 formed by p1-p2-p3 in degrees."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 * mag2 == 0:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_val))


def _body_height(keypoints: list, min_conf: float) -> float:
    """Estimate body height from keypoints (shoulder to ankle distance)."""
    points = []
    for idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]:
        if _kp_valid(keypoints, idx, min_conf):
            points.append(keypoints[idx])

    if len(points) < 2:
        return 0.0

    ys = [p[1] for p in points]
    return max(ys) - min(ys)


def classify_pose(
    keypoints: list[tuple[float, float, float]],
    min_conf: float = 0.3,
    enabled_poses: Optional[list[str]] = None,
) -> Optional[tuple[str, float]]:
    """
    Classify a 17-keypoint COCO skeleton into a named pose.

    Args:
        keypoints: List of 17 (x, y, confidence) tuples.
        min_conf: Minimum keypoint confidence threshold.
        enabled_poses: Subset of poses to check. If None, checks all.

    Returns:
        (pose_name, confidence) or None if no pose detected.
    """
    if len(keypoints) < 17:
        return None

    all_poses = enabled_poses or [
        "hands_up", "t_pose", "waving", "left_hand_up", "right_hand_up",
        "lying_down", "sitting", "crouching", "standing", "pointing",
    ]

    # Count valid keypoints for overall confidence
    valid_count = sum(1 for kp in keypoints if kp[2] >= min_conf)
    if valid_count < 5:
        return None

    avg_conf = sum(kp[2] for kp in keypoints if kp[2] >= min_conf) / max(valid_count, 1)

    for pose_name in all_poses:
        result = _check_pose(pose_name, keypoints, min_conf, avg_conf)
        if result:
            return result

    return None


def _check_pose(
    pose_name: str,
    kps: list[tuple[float, float, float]],
    min_conf: float,
    avg_conf: float,
) -> Optional[tuple[str, float]]:
    """Check if keypoints match a specific pose."""
    checkers = {
        "hands_up": _check_hands_up,
        "t_pose": _check_t_pose,
        "waving": _check_waving,
        "left_hand_up": _check_left_hand_up,
        "right_hand_up": _check_right_hand_up,
        "lying_down": _check_lying_down,
        "sitting": _check_sitting,
        "crouching": _check_crouching,
        "standing": _check_standing,
        "pointing": _check_pointing,
    }

    checker = checkers.get(pose_name)
    if checker:
        return checker(kps, min_conf, avg_conf)
    return None


def _check_hands_up(kps, min_conf, avg_conf) -> Optional[tuple[str, float]]:
    """Both wrists above both shoulders."""
    if not all(_kp_valid(kps, i, min_conf) for i in [LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER]):
        return None

    l_wrist_y = kps[LEFT_WRIST][1]
    r_wrist_y = kps[RIGHT_WRIST][1]
    l_shoulder_y = kps[LEFT_SHOULDER][1]
    r_shoulder_y = kps[RIGHT_SHOULDER][1]

    # In image coords, y increases downward, so "above" means lower y value
    if l_wrist_y < l_shoulder_y and r_wrist_y < r_shoulder_y:
        # Both hands are above shoulders
        conf = min(kps[LEFT_WRIST][2], kps[RIGHT_WRIST][2], kps[LEFT_SHOULDER][2], kps[RIGHT_SHOULDER][2])
        return ("hands_up", conf)

    return None


def _check_t_pose(kps, min_conf, avg_conf) -> Optional[tuple[str, float]]:
    """Both arms extended horizontally — T-pose."""
    needed = [LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW, LEFT_SHOULDER, RIGHT_SHOULDER]
    if not all(_kp_valid(kps, i, min_conf) for i in needed):
        return None

    # Both arms must be extended (shoulder-elbow-wrist angle > 150)
    left_arm_angle = _angle(kps[LEFT_SHOULDER], kps[LEFT_ELBOW], kps[LEFT_WRIST])
    right_arm_angle = _angle(kps[RIGHT_SHOULDER], kps[RIGHT_ELBOW], kps[RIGHT_WRIST])

    if left_arm_angle < 150 or right_arm_angle < 150:
        return None

    # Both arms must be roughly horizontal (wrist within shoulder height tolerance)
    for wrist_idx, shoulder_idx in [(LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER)]:
        arm_dy = abs(kps[wrist_idx][1] - kps[shoulder_idx][1])
        arm_dx = abs(kps[wrist_idx][0] - kps[shoulder_idx][0])
        if arm_dx == 0 or arm_dy / arm_dx > 0.5:
            return None

    conf = min(kps[LEFT_WRIST][2], kps[RIGHT_WRIST][2], kps[LEFT_ELBOW][2], kps[RIGHT_ELBOW][2])
    return ("t_pose", conf)


def _check_waving(kps, min_conf, avg_conf) -> Optional[tuple[str, float]]:
    """One hand well above the head (nose level)."""
    if not _kp_valid(kps, NOSE, min_conf):
        return None

    nose_y = kps[NOSE][1]

    for wrist_idx in [LEFT_WRIST, RIGHT_WRIST]:
        if not _kp_valid(kps, wrist_idx, min_conf):
            continue

        wrist_y = kps[wrist_idx][1]

        # Wrist is well above nose (at least some margin)
        if wrist_y < nose_y:
            # Check the corresponding elbow is also raised
            elbow_idx = LEFT_ELBOW if wrist_idx == LEFT_WRIST else RIGHT_ELBOW
            shoulder_idx = LEFT_SHOULDER if wrist_idx == LEFT_WRIST else RIGHT_SHOULDER

            if _kp_valid(kps, elbow_idx, min_conf) and _kp_valid(kps, shoulder_idx, min_conf):
                elbow_y = kps[elbow_idx][1]
                shoulder_y = kps[shoulder_idx][1]

                if elbow_y < shoulder_y:
                    conf = min(kps[wrist_idx][2], kps[elbow_idx][2])
                    return ("waving", conf)

    return None


def _check_left_hand_up(kps, min_conf, avg_conf) -> Optional[tuple[str, float]]:
    """Left wrist above left shoulder (but not both hands up)."""
    if not all(_kp_valid(kps, i, min_conf) for i in [LEFT_WRIST, LEFT_SHOULDER]):
        return None

    if kps[LEFT_WRIST][1] < kps[LEFT_SHOULDER][1]:
        # Make sure RIGHT hand is NOT also up (that would be hands_up)
        if _kp_valid(kps, RIGHT_WRIST, min_conf) and _kp_valid(kps, RIGHT_SHOULDER, min_conf):
            if kps[RIGHT_WRIST][1] < kps[RIGHT_SHOULDER][1]:
                return None  # Both hands up — handled by hands_up

        conf = min(kps[LEFT_WRIST][2], kps[LEFT_SHOULDER][2])
        return ("left_hand_up", conf)

    return None


def _check_right_hand_up(kps, min_conf, avg_conf) -> Optional[tuple[str, float]]:
    """Right wrist above right shoulder (but not both hands up)."""
    if not all(_kp_valid(kps, i, min_conf) for i in [RIGHT_WRIST, RIGHT_SHOULDER]):
        return None

    if kps[RIGHT_WRIST][1] < kps[RIGHT_SHOULDER][1]:
        # Make sure LEFT hand is NOT also up (that would be hands_up)
        if _kp_valid(kps, LEFT_WRIST, min_conf) and _kp_valid(kps, LEFT_SHOULDER, min_conf):
            if kps[LEFT_WRIST][1] < kps[LEFT_SHOULDER][1]:
                return None  # Both hands up — handled by hands_up

        conf = min(kps[RIGHT_WRIST][2], kps[RIGHT_SHOULDER][2])
        return ("right_hand_up", conf)

    return None


def _check_lying_down(kps, min_conf, avg_conf) -> Optional[tuple[str, float]]:
    """Body is mostly horizontal — small vertical spread relative to horizontal spread."""
    valid_pts = [(kps[i][0], kps[i][1]) for i in range(17) if _kp_valid(kps, i, min_conf)]

    if len(valid_pts) < 6:
        return None

    xs = [p[0] for p in valid_pts]
    ys = [p[1] for p in valid_pts]

    x_spread = max(xs) - min(xs)
    y_spread = max(ys) - min(ys)

    if x_spread == 0:
        return None

    # Lying down: horizontal spread is much larger than vertical
    aspect_ratio = y_spread / x_spread

    if aspect_ratio < 0.4:
        return ("lying_down", avg_conf)

    return None


def _check_sitting(kps, min_conf, avg_conf) -> Optional[tuple[str, float]]:
    """Hip-knee angle indicates sitting posture (roughly 90 degrees)."""
    for side in [(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_SHOULDER),
                 (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_SHOULDER)]:
        hip_idx, knee_idx, ankle_idx, shoulder_idx = side

        if not all(_kp_valid(kps, i, min_conf) for i in [hip_idx, knee_idx, shoulder_idx]):
            continue

        # Check that knees are roughly at hip level (sitting)
        hip_y = kps[hip_idx][1]
        knee_y = kps[knee_idx][1]
        shoulder_y = kps[shoulder_idx][1]

        # Torso is vertical (shoulder above hip)
        if shoulder_y >= hip_y:
            continue

        # Knee is roughly at hip level (within some tolerance)
        body_height = hip_y - shoulder_y
        knee_hip_diff = abs(knee_y - hip_y)

        if knee_hip_diff < body_height * 0.5:
            # Knee is close to hip level — check angle
            if _kp_valid(kps, ankle_idx, min_conf):
                angle = _angle(kps[hip_idx], kps[knee_idx], kps[ankle_idx])
                if 60 < angle < 140:
                    conf = min(kps[hip_idx][2], kps[knee_idx][2])
                    return ("sitting", conf)

    return None


def _check_crouching(kps, min_conf, avg_conf) -> Optional[tuple[str, float]]:
    """Body is low — hips are close to ankles and the torso is bent forward."""
    if not all(_kp_valid(kps, i, min_conf) for i in [LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER]):
        return None

    avg_hip_y = (kps[LEFT_HIP][1] + kps[RIGHT_HIP][1]) / 2
    avg_shoulder_y = (kps[LEFT_SHOULDER][1] + kps[RIGHT_SHOULDER][1]) / 2

    torso_height = avg_hip_y - avg_shoulder_y
    if torso_height <= 0:
        return None

    # Check if any ankle is valid and close to hip
    for ankle_idx in [LEFT_ANKLE, RIGHT_ANKLE]:
        if _kp_valid(kps, ankle_idx, min_conf):
            ankle_y = kps[ankle_idx][1]
            hip_ankle_dist = ankle_y - avg_hip_y

            # Crouching: ankle is close to hip (legs bent)
            if hip_ankle_dist < torso_height * 0.8 and hip_ankle_dist >= 0:
                conf = avg_conf
                return ("crouching", conf)

    return None


def _check_standing(kps, min_conf, avg_conf) -> Optional[tuple[str, float]]:
    """Upright posture — shoulders above hips, hips above knees, knees above ankles."""
    needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if not all(_kp_valid(kps, i, min_conf) for i in needed):
        return None

    avg_shoulder_y = (kps[LEFT_SHOULDER][1] + kps[RIGHT_SHOULDER][1]) / 2
    avg_hip_y = (kps[LEFT_HIP][1] + kps[RIGHT_HIP][1]) / 2

    # Shoulders must be above hips
    if avg_shoulder_y >= avg_hip_y:
        return None

    # Check at least one leg is straight (knee between hip and ankle)
    for knee_idx, ankle_idx, hip_idx in [
        (LEFT_KNEE, LEFT_ANKLE, LEFT_HIP),
        (RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HIP),
    ]:
        if _kp_valid(kps, knee_idx, min_conf) and _kp_valid(kps, ankle_idx, min_conf):
            knee_y = kps[knee_idx][1]
            ankle_y = kps[ankle_idx][1]
            hip_y = kps[hip_idx][1]

            # Knee is between hip and ankle, and leg is mostly straight
            if hip_y < knee_y < ankle_y:
                angle = _angle(kps[hip_idx], kps[knee_idx], kps[ankle_idx])
                if angle > 150:  # Leg is nearly straight
                    return ("standing", avg_conf)

    return None


def _check_pointing(kps, min_conf, avg_conf) -> Optional[tuple[str, float]]:
    """One arm extended outward while the other is not."""
    for (wrist_idx, elbow_idx, shoulder_idx) in [
        (LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER),
        (RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER),
    ]:
        if not all(_kp_valid(kps, i, min_conf) for i in [wrist_idx, elbow_idx, shoulder_idx]):
            continue

        # Check arm is extended (shoulder-elbow-wrist angle is close to 180)
        arm_angle = _angle(kps[shoulder_idx], kps[elbow_idx], kps[wrist_idx])

        if arm_angle > 150:
            # Arm is extended — check it's roughly horizontal
            wrist_y = kps[wrist_idx][1]
            shoulder_y = kps[shoulder_idx][1]
            shoulder_x = kps[shoulder_idx][0]
            wrist_x = kps[wrist_idx][0]

            arm_dx = abs(wrist_x - shoulder_x)
            arm_dy = abs(wrist_y - shoulder_y)

            # Arm should be more horizontal than vertical
            if arm_dx > arm_dy * 1.5:
                conf = min(kps[wrist_idx][2], kps[elbow_idx][2])
                return ("pointing", conf)

    return None
