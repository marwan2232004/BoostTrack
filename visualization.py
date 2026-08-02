import os
import json
import cv2
from collections import defaultdict


def draw_panel(frame, H, worker_times, cfg, max_rows=None):
    panel_x = 20
    panel_y = H - 240
    panel_w = 360
    panel_h = 220

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (255, 255, 255),
        -1,
    )

    alpha = 0.82
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (0, 0, 0),
        2,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        frame,
        "Worker Activity",
        (panel_x + 10, panel_y + 25),
        font,
        0.65,
        (0, 0, 0),
        2,
    )

    y = panel_y + 50
    cv2.putText(frame, "ID", (panel_x + 10, y), font, 0.5, (0, 0, 0), 1)
    cv2.putText(
        frame,
        "Work",
        (panel_x + 70, y),
        font,
        0.5,
        cfg.CATEGORY_COLORS_BGR["working"],
        1,
    )
    cv2.putText(
        frame, "Idle", (panel_x + 170, y), font, 0.5, cfg.CATEGORY_COLORS_BGR["idle"], 1
    )
    cv2.putText(
        frame,
        "Transit",
        (panel_x + 255, y),
        font,
        0.5,
        cfg.CATEGORY_COLORS_BGR["transit"],
        1,
    )
    y += 25

    worker_ids = sorted(worker_times.keys())
    row_limit = max_rows if max_rows is not None else len(worker_ids)
    shown = 0

    for worker_id in worker_ids:
        if y > panel_y + panel_h - 10:
            break
        if shown >= row_limit:
            break

        stats = worker_times[worker_id]

        cv2.putText(frame, f"#{worker_id}", (panel_x + 10, y), font, 0.5, (0, 0, 0), 1)
        cv2.putText(
            frame,
            f"{stats['working']:.1f}s",
            (panel_x + 70, y),
            font,
            0.5,
            cfg.CATEGORY_COLORS_BGR["working"],
            1,
        )
        cv2.putText(
            frame,
            f"{stats['idle']:.1f}s",
            (panel_x + 170, y),
            font,
            0.5,
            cfg.CATEGORY_COLORS_BGR["idle"],
            1,
        )
        cv2.putText(
            frame,
            f"{stats['transit']:.1f}s",
            (panel_x + 255, y),
            font,
            0.5,
            cfg.CATEGORY_COLORS_BGR["transit"],
            1,
        )

        y += 22
        shown += 1

    remaining = len(worker_ids) - shown
    if remaining > 0:
        cv2.putText(
            frame,
            f"+{remaining} more worker(s) not shown",
            (panel_x + 10, y),
            font,
            0.45,
            (80, 80, 80),
            1,
        )

    return frame


def show_results_with_actions(
    output_path,
    merged_detections,
    split_path,
    annotations_path,
    cfg,
    fps=25,
    sequence_name=None,
):
    """
    Visualize tracking + action recognition results.

    - Bounding box color = action class
    - Filled label background
    - White text
    - Running per-worker activity panel (working/idle/transit seconds)

    Parameters
    ----------
    sequence_name : str or None
        If given, only frames whose file path starts with this folder name
        are used (matches the dataset's `images/<sequence_name>/...`
        layout). If None, all frames in the annotation file are used.
    """
    with open(annotations_path, "r") as f:
        data = json.load(f)

    # ---------------------------------------------------------
    # Load frame paths
    # ---------------------------------------------------------
    frames_info = {}
    for img in data["images"]:
        if (
            sequence_name is not None
            and img["file_name"].split("/")[0] != sequence_name
        ):
            continue
        frames_info[img["frame_id"]] = os.path.join(split_path, img["file_name"])

    if not frames_info:
        raise ValueError(
            f"No frames found for sequence_name={sequence_name!r}. "
            "Check the annotations file and sequence_name argument."
        )

    sorted_frames = sorted(frames_info.keys())

    first_frame = cv2.imread(frames_info[sorted_frames[0]])
    if first_frame is None:
        raise FileNotFoundError(
            f"Could not read first frame: {frames_info[sorted_frames[0]]}"
        )
    H, W = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    worker_times = defaultdict(
        lambda: {
            "working": 0.0,
            "idle": 0.0,
            "transit": 0.0,
            "unknown": 0.0,
        }
    )

    frame_time = 1.0 / fps
    n_missing_frames = 0

    # ---------------------------------------------------------
    # Draw every frame
    # ---------------------------------------------------------
    try:
        for frame_id in sorted_frames:
            frame = cv2.imread(frames_info[frame_id])
            if frame is None:
                n_missing_frames += 1
                continue  # skip unreadable frames rather than crashing

            for det in merged_detections.get(frame_id, []):
                x1 = max(0, int(det["x"]))
                y1 = max(0, int(det["y"]))
                x2 = min(W - 1, x1 + int(det["w"]))
                y2 = min(H - 1, y1 + int(det["h"]))

                action = det["action"]
                worker_id = det["tracked_id"]

                category = cfg.ACTION_CATEGORY_MAP.get(action, "unknown")
                worker_times[worker_id][category] += frame_time

                box_color = cfg.ACTION_COLORS_BGR.get(
                    action, cfg.ACTION_COLORS_BGR["unknown"]
                )

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

                # Label
                label = f"#{det['tracked_id']}  {action}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2

                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                pad = 5

                label_top = max(0, y1 - th - 2 * pad)
                label_bottom = y1
                if label_top == 0:
                    label_bottom = th + 2 * pad

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

            frame = draw_panel(frame, H, worker_times, cfg)

            writer.write(frame)
    finally:
        writer.release()

    if n_missing_frames:
        print(f"WARNING: skipped {n_missing_frames} unreadable frame(s).")
    print(f"Video saved to {output_path}")
