"""DensePose inference module for yoga pose analysis."""

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.structures import DensePoseResultQuantized


def setup_densepose(config_path, weights_path, score_threshold=0.7):
    """Initialize DensePose predictor.

    Args:
        config_path: Path to DensePose config YAML.
        weights_path: Path to model weights (.pkl).
        score_threshold: Minimum confidence for detections.

    Returns:
        DefaultPredictor configured for DensePose.
    """
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


def run_inference(predictor, image):
    """Run DensePose on an image.

    Args:
        predictor: DensePose predictor from setup_densepose().
        image: BGR image (numpy array) or path to image file.

    Returns:
        dict with keys:
            - boxes: detected bounding boxes
            - scores: confidence scores
            - densepose: list of DensePose results per person
            - body_parts: dict mapping body part ID to surface coordinates
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image}")

    outputs = predictor(image)
    instances = outputs["instances"]

    results = {
        "boxes": instances.pred_boxes.tensor.cpu().numpy(),
        "scores": instances.scores.cpu().numpy(),
        "densepose": [],
        "body_parts": [],
    }

    if not instances.has("pred_densepose"):
        return results

    densepose_results = instances.pred_densepose
    for i in range(len(instances)):
        dp = extract_body_parts(densepose_results, i)
        results["densepose"].append(densepose_results[i])
        results["body_parts"].append(dp)

    return results


# DensePose body part IDs
BODY_PART_NAMES = {
    0: "background",
    1: "torso_back",
    2: "torso_front",
    3: "right_hand",
    4: "left_hand",
    5: "left_foot",
    6: "right_foot",
    7: "right_upper_leg_back",
    8: "left_upper_leg_back",
    9: "right_upper_leg_front",
    10: "left_upper_leg_front",
    11: "right_lower_leg_back",
    12: "left_lower_leg_back",
    13: "right_lower_leg_front",
    14: "left_lower_leg_front",
    15: "left_upper_arm_back",
    16: "right_upper_arm_back",
    17: "left_upper_arm_front",
    18: "right_upper_arm_front",
    19: "left_lower_arm_back",
    20: "right_lower_arm_back",
    21: "left_lower_arm_front",
    22: "right_lower_arm_front",
    23: "right_face",
    24: "left_face",
}


def extract_body_parts(densepose_results, person_idx):
    """Extract per-body-part surface coordinates for one person.

    Args:
        densepose_results: DensePose output from instances.
        person_idx: Index of the person in the detection results.

    Returns:
        dict mapping body part name to dict with:
            - pixels: list of (x, y) pixel coordinates
            - u: U surface coordinates
            - v: V surface coordinates
            - centroid: (x, y) mean position of the body part
    """
    result = densepose_results[person_idx]

    if isinstance(result, DensePoseResultQuantized):
        labels = result.labels_uv_uint8[0].cpu().numpy()
        u = result.labels_uv_uint8[1].cpu().numpy().astype(float) / 255.0
        v = result.labels_uv_uint8[2].cpu().numpy().astype(float) / 255.0
    else:
        labels = result.labels.cpu().numpy()
        u = result.uv[0].cpu().numpy()
        v = result.uv[1].cpu().numpy()

    body_parts = {}
    for part_id, part_name in BODY_PART_NAMES.items():
        if part_id == 0:
            continue
        mask = labels == part_id
        if not mask.any():
            continue

        ys, xs = np.where(mask)
        body_parts[part_name] = {
            "pixels": list(zip(xs.tolist(), ys.tolist())),
            "u": u[mask].tolist(),
            "v": v[mask].tolist(),
            "centroid": (float(xs.mean()), float(ys.mean())),
        }

    return body_parts


def compute_body_angles(body_parts):
    """Compute angles between body parts from their centroids.

    Args:
        body_parts: Output from extract_body_parts().

    Returns:
        dict of angle measurements:
            - torso_vertical: angle of torso from vertical (degrees)
            - left_arm_angle: angle at left elbow
            - right_arm_angle: angle at right elbow
            - left_leg_angle: angle at left knee
            - right_leg_angle: angle at right knee
            - shoulder_level: difference in y between shoulders
            - hip_level: difference in y between hips
    """
    angles = {}

    # Torso angle from vertical
    if "torso_front" in body_parts:
        pixels = body_parts["torso_front"]["pixels"]
        if len(pixels) > 10:
            xs, ys = zip(*pixels)
            xs, ys = np.array(xs), np.array(ys)
            # Fit line to torso pixels
            if ys.max() - ys.min() > 0:
                slope = np.polyfit(ys, xs, 1)[0]
                angles["torso_vertical"] = np.degrees(np.arctan(slope))

    # Arm angles (using centroids of upper and lower arm segments)
    for side in ["left", "right"]:
        upper_front = f"{side}_upper_arm_front"
        lower_front = f"{side}_lower_arm_front"
        hand = f"{side}_hand"
        if upper_front in body_parts and lower_front in body_parts:
            c_upper = np.array(body_parts[upper_front]["centroid"])
            c_lower = np.array(body_parts[lower_front]["centroid"])
            if hand in body_parts:
                c_hand = np.array(body_parts[hand]["centroid"])
                angle = _angle_between_three_points(c_upper, c_lower, c_hand)
                angles[f"{side}_arm_angle"] = angle
            else:
                # Angle of arm from vertical
                diff = c_lower - c_upper
                angles[f"{side}_arm_vertical"] = np.degrees(
                    np.arctan2(diff[0], diff[1])
                )

    # Leg angles
    for side in ["left", "right"]:
        upper_front = f"{side}_upper_leg_front"
        lower_front = f"{side}_lower_leg_front"
        foot = f"{side}_foot"
        if upper_front in body_parts and lower_front in body_parts:
            c_upper = np.array(body_parts[upper_front]["centroid"])
            c_lower = np.array(body_parts[lower_front]["centroid"])
            if foot in body_parts:
                c_foot = np.array(body_parts[foot]["centroid"])
                angle = _angle_between_three_points(c_upper, c_lower, c_foot)
                angles[f"{side}_leg_angle"] = angle

    # Shoulder level (using upper arm centroids as proxy)
    left_shoulder = body_parts.get("left_upper_arm_front", {}).get("centroid")
    right_shoulder = body_parts.get("right_upper_arm_front", {}).get("centroid")
    if left_shoulder and right_shoulder:
        angles["shoulder_level"] = left_shoulder[1] - right_shoulder[1]

    # Hip level (using upper leg centroids as proxy)
    left_hip = body_parts.get("left_upper_leg_front", {}).get("centroid")
    right_hip = body_parts.get("right_upper_leg_front", {}).get("centroid")
    if left_hip and right_hip:
        angles["hip_level"] = left_hip[1] - right_hip[1]

    return angles


def _angle_between_three_points(p1, p2, p3):
    """Compute angle at p2 formed by p1-p2-p3 in degrees."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))
