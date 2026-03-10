"""Main entry point for yoga_perception."""

import argparse
import cv2
from .densepose_inference import setup_densepose, run_inference
from .cue_engine import analyze_pose, format_feedback
from .pose_definitions import list_poses


def main():
    parser = argparse.ArgumentParser(
        description="yoga_perception — Yoga alignment analysis using DensePose"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--pose", required=True, choices=list_poses(),
        help="Target pose to evaluate against"
    )
    parser.add_argument(
        "--config",
        default="models/densepose_rcnn_R_50_FPN_s1x.yaml",
        help="Path to DensePose config file"
    )
    parser.add_argument(
        "--weights",
        default="models/model_final_162be9.pkl",
        help="Path to DensePose model weights"
    )
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--output", help="Path to save annotated image")
    args = parser.parse_args()

    print("Loading DensePose model...")
    predictor = setup_densepose(args.config, args.weights, args.threshold)

    print(f"Running inference on {args.image}...")
    results = run_inference(predictor, args.image)

    if not results["body_parts"]:
        print("No person detected in the image.")
        return

    # Analyze the first detected person
    body_parts = results["body_parts"][0]
    analysis = analyze_pose(body_parts, args.pose)
    print()
    print(format_feedback(analysis))

    if args.output:
        image = cv2.imread(args.image)
        _draw_annotations(image, results, analysis)
        cv2.imwrite(args.output, image)
        print(f"\nAnnotated image saved to {args.output}")


def _draw_annotations(image, results, analysis):
    """Draw bounding box and score on the image."""
    if len(results["boxes"]) > 0:
        box = results["boxes"][0].astype(int)
        score = analysis["score"]
        color = (0, int(score * 2.55), int((100 - score) * 2.55))
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(
            image, f"Alignment: {score}%",
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
        )


if __name__ == "__main__":
    main()
