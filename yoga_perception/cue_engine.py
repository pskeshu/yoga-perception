"""Cue engine: compares DensePose output against pose rules and generates cues."""

from .densepose_inference import compute_body_angles
from .pose_definitions import get_pose


def analyze_pose(body_parts, pose_name):
    """Analyze a detected body against a target pose's alignment rules.

    Args:
        body_parts: dict from extract_body_parts() for one person.
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
    angles = compute_body_angles(body_parts)

    # Add derived checks
    angles.update(_compute_derived_checks(body_parts, angles, pose_name))

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


def _compute_derived_checks(body_parts, angles, pose_name):
    """Compute pose-specific derived measurements."""
    derived = {}

    # Arms aligned vertically (for Trikonasana)
    left_arm = body_parts.get("left_upper_arm_front", {}).get("centroid")
    right_arm = body_parts.get("right_upper_arm_front", {}).get("centroid")
    if left_arm and right_arm:
        x_diff = abs(left_arm[0] - right_arm[0])
        y_diff = abs(left_arm[1] - right_arm[1])
        if y_diff > 0:
            derived["arms_aligned"] = x_diff < 30 and y_diff > 50

    # Arms horizontal (for Warrior II)
    if left_arm and right_arm:
        y_diff = abs(left_arm[1] - right_arm[1])
        derived["arms_horizontal"] = y_diff < 20

    # Front knee angle (use the leg with larger bend)
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

    # Front knee over ankle (simplified: check if front leg is straight)
    derived["front_knee_over_ankle"] = derived.get("front_knee_angle", 180) > 165

    return derived
