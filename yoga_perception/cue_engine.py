"""Cue engine: compares pose landmarks against alignment rules and generates cues."""

import numpy as np
from .pose_definitions import get_pose


def analyze_pose(result, pose_name):
    """Analyze detected pose against a target pose's alignment rules.

    Args:
        result: output from pose_inference.run_inference().
        pose_name: name of the target pose (e.g. "tadasana").

    Returns:
        dict with:
            - pose: the pose definition
            - angles: computed body angles
            - cues: list of triggered cues, sorted by priority
            - aligned: list of rules that passed
            - missing: list of checks that couldn't be evaluated
            - score: alignment score (0-100)
    """
    pose = get_pose(pose_name)
    angles = dict(result["angles"])

    # Add derived checks
    angles.update(_compute_derived_checks(result["landmarks"], angles))

    cues = []
    aligned = []
    missing = []

    for rule in pose["rules"]:
        check_key = rule["check"]
        if check_key not in angles:
            missing.append(check_key)
            continue

        value = angles[check_key]
        if rule["condition"](value):
            aligned.append(rule)
        else:
            if rule["cue"] is not None:
                cues.append({
                    "cue": rule["cue"],
                    "priority": rule["priority"],
                    "check": check_key,
                    "value": value,
                })

    # Sort cues by priority (1 = most important first)
    cues.sort(key=lambda c: c["priority"])

    total_rules = len(aligned) + len(cues)
    score = (len(aligned) / total_rules * 100) if total_rules > 0 else 0

    return {
        "pose": pose,
        "angles": angles,
        "cues": cues,
        "aligned": aligned,
        "missing": missing,
        "score": round(score, 1),
    }


def format_feedback(analysis, max_cues=5):
    """Format analysis into readable feedback.

    Args:
        analysis: output from analyze_pose().
        max_cues: maximum number of cues to show.

    Returns:
        Formatted string with alignment feedback.
    """
    pose = analysis["pose"]
    lines = []
    lines.append(f"Pose: {pose['english']} ({pose['sanskrit']})")
    lines.append(f"Alignment Score: {analysis['score']}%")
    lines.append("")

    if not analysis["cues"]:
        lines.append("Excellent alignment! All checks passed.")
    else:
        lines.append("Alignment cues:")
        for i, cue_info in enumerate(analysis["cues"][:max_cues]):
            priority_label = {1: "***", 2: "**", 3: "*"}[cue_info["priority"]]
            lines.append(f"  {priority_label} {cue_info['cue']}")

        remaining = len(analysis["cues"]) - max_cues
        if remaining > 0:
            lines.append(f"  ... and {remaining} more refinements.")

    if analysis["missing"]:
        lines.append("")
        lines.append(f"Note: Could not evaluate {len(analysis['missing'])} checks "
                      f"(body parts not visible).")

    return "\n".join(lines)


def _compute_derived_checks(landmarks, angles):
    """Compute pose-specific derived measurements."""
    derived = {}

    # Arms aligned vertically (for Trikonasana): left and right wrist x close
    lw = landmarks.get("left_wrist")
    rw = landmarks.get("right_wrist")
    ls = landmarks.get("left_shoulder")
    rs = landmarks.get("right_shoulder")
    if lw and rw and ls and rs:
        wrist_x_diff = abs(lw["x"] - rw["x"])
        wrist_y_diff = abs(lw["y"] - rw["y"])
        derived["arms_aligned"] = wrist_x_diff < 40 and wrist_y_diff > 100

    # Arms horizontal: wrists at roughly same y-level
    if lw and rw:
        derived["arms_horizontal"] = abs(lw["y"] - rw["y"]) < 30

    # Front knee angle: whichever knee is more bent
    left_angle = angles.get("left_leg_angle", 180)
    right_angle = angles.get("right_leg_angle", 180)
    if left_angle < right_angle:
        derived["front_knee_angle"] = left_angle
        derived["back_leg_angle"] = right_angle
        derived["standing_leg_angle"] = right_angle
    else:
        derived["front_knee_angle"] = right_angle
        derived["back_leg_angle"] = left_angle
        derived["standing_leg_angle"] = left_angle

    derived["front_knee_over_ankle"] = derived.get("front_knee_angle", 180) > 165

    return derived
