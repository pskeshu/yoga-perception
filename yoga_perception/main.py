"""Main entry point for yoga_perception."""

import argparse
import cv2
import numpy as np
from .pose_inference import setup_pose_detector, run_inference, draw_landmarks
from .cue_engine import analyze_pose, format_feedback
from .pose_definitions import list_poses


def main():
    parser = argparse.ArgumentParser(
        description="yoga_perception -- Yoga alignment analysis using pose estimation"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--pose", required=True, choices=list_poses(),
        help="Target pose to evaluate against"
    )
    parser.add_argument("--output", help="Path to save annotated image")
    args = parser.parse_args()

    print("Loading pose detector...")
    detector = setup_pose_detector()

    print(f"Running inference on {args.image}...")
    result = run_inference(detector, args.image)

    if result is None:
        print("No person detected in the image.")
        return

    analysis = analyze_pose(result, args.pose)
    print()
    print(format_feedback(analysis))

    if args.output:
        image = cv2.imread(args.image)
        composite = build_composite(image, result, analysis)
        cv2.imwrite(args.output, composite)
        print(f"\nResult saved to {args.output}")


def build_composite(image, result, analysis):
    """Build a 3-panel side-by-side image: original | pose | annotations.

    Args:
        image: original BGR image.
        result: output from run_inference().
        analysis: output from analyze_pose().

    Returns:
        Composite BGR image with 3 panels.
    """
    h, w = image.shape[:2]
    score = analysis["score"]
    pose = analysis["pose"]

    # Panel 1: Original image with title
    panel_orig = image.copy()
    _draw_title(panel_orig, "Original")

    # Panel 2: Pose skeleton on black background
    panel_pose = np.zeros_like(image)
    panel_pose = draw_landmarks(panel_pose, result)
    _draw_title(panel_pose, "Pose Estimation")

    # Panel 3: Annotation panel — original with skeleton + cues
    panel_cues = draw_landmarks(image.copy(), result)
    _draw_annotation_panel(panel_cues, analysis)

    # Add thin white separator lines
    sep = np.ones((h, 2, 3), dtype=np.uint8) * 255

    composite = np.hstack([panel_orig, sep, panel_pose, sep, panel_cues])
    return composite


def _draw_title(panel, title):
    """Draw a title bar at the top of a panel."""
    h, w = panel.shape[:2]
    # Semi-transparent dark bar
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)
    cv2.putText(
        panel, title, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )


def _draw_annotation_panel(panel, analysis):
    """Draw score and alignment cues on the annotation panel."""
    h, w = panel.shape[:2]
    score = analysis["score"]
    pose = analysis["pose"]

    # Title bar with pose name and score
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (w, 35), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)

    score_color = (0, int(score * 2.55), int((100 - score) * 2.55))
    title = f"{pose['english']} - {score}%"
    cv2.putText(
        panel, title, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2,
    )

    # Cue text at the bottom with dark background
    cues = analysis["cues"][:5]
    if cues:
        line_h = 22
        box_h = len(cues) * line_h + 15
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, h - box_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)

        y = h - box_h + 20
        for cue_info in cues:
            priority_marker = {1: "[!]", 2: "[*]", 3: "[-]"}[cue_info["priority"]]
            text = f"{priority_marker} {cue_info['cue']}"
            # Truncate to fit panel width (~1 char per 7px at scale 0.4)
            max_chars = int(w / 6)
            if len(text) > max_chars:
                text = text[:max_chars - 3] + "..."
            cv2.putText(
                panel, text, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 180, 255), 1,
            )
            y += line_h

    # If no cues, show "aligned" message
    if not cues:
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)
        cv2.putText(
            panel, "Excellent alignment!", (8, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )


if __name__ == "__main__":
    main()
