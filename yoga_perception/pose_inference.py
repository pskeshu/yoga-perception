"""Pose inference module using MediaPipe PoseLandmarker for yoga alignment analysis."""

import os
import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
PoseLandmark = mp.tasks.vision.PoseLandmark

# Landmark indices we care about
LANDMARKS = {
    "nose": PoseLandmark.NOSE,
    "left_shoulder": PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": PoseLandmark.LEFT_ELBOW,
    "right_elbow": PoseLandmark.RIGHT_ELBOW,
    "left_wrist": PoseLandmark.LEFT_WRIST,
    "right_wrist": PoseLandmark.RIGHT_WRIST,
    "left_hip": PoseLandmark.LEFT_HIP,
    "right_hip": PoseLandmark.RIGHT_HIP,
    "left_knee": PoseLandmark.LEFT_KNEE,
    "right_knee": PoseLandmark.RIGHT_KNEE,
    "left_ankle": PoseLandmark.LEFT_ANKLE,
    "right_ankle": PoseLandmark.RIGHT_ANKLE,
    "left_foot_index": PoseLandmark.LEFT_FOOT_INDEX,
    "right_foot_index": PoseLandmark.RIGHT_FOOT_INDEX,
}

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "pose_landmarker_heavy.task"
)


def setup_pose_detector(model_path=None):
    """Initialize MediaPipe PoseLandmarker.

    Args:
        model_path: Path to .task model file. Uses bundled model by default.

    Returns:
        PoseLandmarker instance.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(options)


def run_inference(detector, image):
    """Run pose detection on an image.

    Args:
        detector: PoseLandmarker from setup_pose_detector().
        image: BGR image (numpy array) or path to image file.

    Returns:
        dict with keys:
            - landmarks: dict mapping landmark name to {x, y, z, visibility}
            - angles: dict of computed body angles
            - image_shape: (height, width)
            - raw_result: original MediaPipe result
        Returns None if no person detected.
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image}")

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    pose_landmarks = result.pose_landmarks[0]
    landmarks = {}
    for name, idx in LANDMARKS.items():
        lm = pose_landmarks[idx]
        landmarks[name] = {
            "x": lm.x * w,
            "y": lm.y * h,
            "z": lm.z,
            "visibility": lm.visibility,
        }

    angles = compute_body_angles(landmarks)

    return {
        "landmarks": landmarks,
        "angles": angles,
        "image_shape": (h, w),
        "raw_result": result,
    }


def compute_body_angles(landmarks):
    """Compute joint angles and alignment measurements from landmarks.

    Returns:
        dict of angle measurements:
            - torso_vertical: angle of torso from vertical (degrees, 0 = upright)
            - left/right_arm_angle: angle at elbow (degrees)
            - left/right_leg_angle: angle at knee (degrees)
            - left/right_arm_vertical: angle of arm from vertical
            - shoulder_level: y-difference between shoulders (pixels)
            - hip_level: y-difference between hips (pixels)
    """
    angles = {}

    def pt(name):
        lm = landmarks.get(name)
        if lm and lm["visibility"] > 0.5:
            return np.array([lm["x"], lm["y"]])
        return None

    # Torso vertical
    ls, rs = pt("left_shoulder"), pt("right_shoulder")
    lh, rh = pt("left_hip"), pt("right_hip")
    if ls is not None and rs is not None and lh is not None and rh is not None:
        mid_shoulder = (ls + rs) / 2
        mid_hip = (lh + rh) / 2
        torso_vec = mid_shoulder - mid_hip
        vertical = np.array([0, -1])
        cos_a = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-8)
        angles["torso_vertical"] = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    # Shoulder level
    if ls is not None and rs is not None:
        angles["shoulder_level"] = ls[1] - rs[1]

    # Hip level
    if lh is not None and rh is not None:
        angles["hip_level"] = lh[1] - rh[1]

    # Arm angles (at elbow) and arm-from-vertical
    for side in ["left", "right"]:
        shoulder = pt(f"{side}_shoulder")
        elbow = pt(f"{side}_elbow")
        wrist = pt(f"{side}_wrist")
        if shoulder is not None and elbow is not None and wrist is not None:
            angles[f"{side}_arm_angle"] = _angle_at(shoulder, elbow, wrist)
        if shoulder is not None and wrist is not None:
            arm_vec = wrist - shoulder
            vertical = np.array([0, 1])
            cos_a = np.dot(arm_vec, vertical) / (np.linalg.norm(arm_vec) + 1e-8)
            angles[f"{side}_arm_vertical"] = np.degrees(
                np.arccos(np.clip(cos_a, -1, 1))
            )

    # Leg angles (at knee)
    for side in ["left", "right"]:
        hip = pt(f"{side}_hip")
        knee = pt(f"{side}_knee")
        ankle = pt(f"{side}_ankle")
        if hip is not None and knee is not None and ankle is not None:
            angles[f"{side}_leg_angle"] = _angle_at(hip, knee, ankle)

    return angles


def _angle_at(p1, p2, p3):
    """Compute angle at p2 formed by p1-p2-p3 in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def draw_landmarks(image, result):
    """Draw pose landmarks and connections on an image.

    Args:
        image: BGR image (numpy array).
        result: output from run_inference().

    Returns:
        Annotated image copy.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    if not result["raw_result"].pose_landmarks:
        return annotated

    pose_landmarks = result["raw_result"].pose_landmarks[0]

    # Color-coded connections by body region
    connections = [
        # Torso (cyan)
        ("left_shoulder", "right_shoulder", (255, 255, 0)),
        ("left_shoulder", "left_hip", (255, 255, 0)),
        ("right_shoulder", "right_hip", (255, 255, 0)),
        ("left_hip", "right_hip", (255, 255, 0)),
        # Left arm (green)
        ("left_shoulder", "left_elbow", (0, 255, 0)),
        ("left_elbow", "left_wrist", (0, 255, 0)),
        # Right arm (blue)
        ("right_shoulder", "right_elbow", (255, 100, 0)),
        ("right_elbow", "right_wrist", (255, 100, 0)),
        # Left leg (yellow)
        ("left_hip", "left_knee", (0, 255, 255)),
        ("left_knee", "left_ankle", (0, 255, 255)),
        # Right leg (magenta)
        ("right_hip", "right_knee", (255, 0, 255)),
        ("right_knee", "right_ankle", (255, 0, 255)),
    ]

    for name1, name2, color in connections:
        lm1 = result["landmarks"].get(name1)
        lm2 = result["landmarks"].get(name2)
        if lm1 and lm2 and lm1["visibility"] > 0.5 and lm2["visibility"] > 0.5:
            pt1 = (int(lm1["x"]), int(lm1["y"]))
            pt2 = (int(lm2["x"]), int(lm2["y"]))
            cv2.line(annotated, pt1, pt2, color, 2)

    # Draw landmark points
    for name, lm in result["landmarks"].items():
        if lm["visibility"] > 0.5:
            pt = (int(lm["x"]), int(lm["y"]))
            cv2.circle(annotated, pt, 5, (0, 255, 0), -1)
            cv2.circle(annotated, pt, 3, (255, 255, 255), -1)

    return annotated
