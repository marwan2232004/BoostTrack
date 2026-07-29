import os
import json
import cv2


def show_results_with_actions(
    output_path,
    merged_detections,
    split_path,
    annotations_path,
    cfg,
    fps=25,
):
    """
    Visualize tracking + action recognition results.

    - Bounding box color = action class
    - Filled label background
    - White text
    """

    with open(annotations_path, "r") as f:
        data = json.load(f)

    # ---------------------------------------------------------
    # Load frame paths
    # ---------------------------------------------------------
    frames_info = {}

    for img in data["images"]:
        if img["file_name"].split("/")[0] != "MOT20-01":
            continue

        frames_info[img["frame_id"]] = os.path.join(
            split_path,
            img["file_name"],
        )

    sorted_frames = sorted(frames_info.keys())

    first_frame = cv2.imread(frames_info[sorted_frames[0]])
    H, W = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    # ---------------------------------------------------------
    # Draw every frame
    # ---------------------------------------------------------
    for frame_id in sorted_frames:

        frame = cv2.imread(frames_info[frame_id])

        for det in merged_detections.get(frame_id, []):

            x1 = int(det["x"])
            y1 = int(det["y"])
            x2 = x1 + int(det["w"])
            y2 = y1 + int(det["h"])

            action = det["action"]

            box_color = cfg.ACTION_COLORS_BGR.get(
                action,
                cfg.ACTION_COLORS_BGR["unknown"],
            )

            # -----------------------------
            # Bounding box
            # -----------------------------
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
                cv2.LINE_AA,
            )

            # -----------------------------
            # Label
            # -----------------------------
            label = f"#{det['tracked_id']}  {action}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2

            (tw, th), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )

            pad = 5

            label_top = max(0, y1 - th - 2 * pad)
            label_bottom = y1

            # Keep label inside image
            if label_top == 0:
                label_bottom = th + 2 * pad

            # Filled label background
            cv2.rectangle(
                frame,
                (x1, label_top),
                (x1 + tw + 2 * pad, label_bottom),
                box_color,
                -1,
            )

            text_y = label_bottom - pad if label_top == 0 else y1 - pad

            cv2.putText(
                frame,
                label,
                (x1 + pad, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        writer.write(frame)

    writer.release()

    print(f"Video saved to {output_path}")
