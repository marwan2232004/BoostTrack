import os
import json
import cv2
from collections import defaultdict


def draw_panel(
    frame,
    worker_times,
    cfg,
    actions,
    elapsed_sec,
    track_ids=None,
    panel_x=0,
    panel_y=None,
    panel_w=460,
    row_h=24,
    max_rows=None,
):
    H, W = frame.shape[:2]
    if panel_y is None:
        panel_y = int(H * 0.55)

    ids = track_ids if track_ids is not None else sorted(worker_times.keys())
    total_rows = len(ids) * len(actions)
    n_rows = min(total_rows, max_rows) if max_rows is not None else total_rows

    header_h = 45
    panel_h = min(header_h + row_h * n_rows + 10, H - panel_y)

    overlay = frame.copy()
    cv2.rectangle(
        overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
        (255, 255, 255), -1,
    )
    
    frame = cv2.addWeighted(overlay, 0.92, frame, 0.08, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (30, 30, 30)

    # Header
    cv2.putText(frame, "Activity", (panel_x + 15, panel_y + 22), font, 0.6, text_color, 2, cv2.LINE_AA)

    # Inline legend: colored square + action name, one per action
    legend_x = panel_x + 110
    legend_y = panel_y + 16
    sq = 10
    slot_w = max(60, (panel_w - 125) // max(len(actions), 1))
    for i, action in enumerate(actions):
        color = cfg.ACTION_COLORS_BGR.get(action.lower(), (150, 150, 150))
        cx = legend_x + i * slot_w
        cv2.rectangle(frame, (cx, legend_y - sq), (cx + sq, legend_y), color, -1)
        cv2.putText(frame, action.capitalize(), (cx + sq + 4, legend_y), font, 0.38, text_color, 1, cv2.LINE_AA)

    # Rows: one per (track_id, action)
    label_x = panel_x + 12
    bar_x = panel_x + 95
    value_x = panel_x + panel_w - 55
    bar_max_w = value_x - bar_x - 10

    y = panel_y + header_h
    shown = 0
    for tid in ids:
        for action in actions:
            if max_rows is not None and shown >= max_rows:
                break
            value = worker_times.get(tid, {}).get(action.lower(), 0.0)
            frac = 0.0 if elapsed_sec <= 0 else min(1.0, value / elapsed_sec)
            bar_w = int(frac * bar_max_w)
            color = cfg.ACTION_COLORS_BGR.get(action.lower(), (150, 150, 150))

            cv2.putText(frame, f"{tid}-{action.capitalize()}", (label_x, y + 4),
                        font, 0.4, text_color, 1, cv2.LINE_AA)
            if bar_w > 0:
                cv2.rectangle(frame, (bar_x, y - 8), (bar_x + bar_w, y + 3), color, -1)
            cv2.putText(frame, f"{value:.0f}sec", (value_x, y + 4),
                        font, 0.4, text_color, 1, cv2.LINE_AA)

            y += row_h
            shown += 1
        if max_rows is not None and shown >= max_rows:
            break

    return frame


def show_results_with_actions(
    output_path,
    merged_detections,
    split_path,
    annotations_path,
    cfg,
    fps=25,
    sequence_name=None,
    panel_actions=None,
    panel_track_ids=None,
):
    """
    Parameters
    ----------
    sequence_name : str or None
        If given, only frames whose file path starts with this folder name
        are used (matches the dataset's `images/<sequence_name>/...`
        layout). If None, all frames in the annotation file are used.
    panel_actions : list[str] or None
        Ordered actions shown in the panel (e.g.
        ["hooking", "lifting", "positioning", "returning"]). Defaults to
        cfg.PANEL_ACTIONS if present, else the sorted set of actions seen
        in cfg.ACTION_COLORS_BGR.
    panel_track_ids : list or None
        Fixed set/order of track IDs to always show in the panel (keeps
        rows stable even before a worker first appears). Defaults to
        cfg.PANEL_TRACK_IDS if present, else whichever tracks have shown
        up so far.
    """
    with open(annotations_path, "r") as f:
        data = json.load(f)

    # ---------------------------------------------------------
    # Load frame paths
    # ---------------------------------------------------------
    frames_info = {}
    for img in data["images"]:
        if sequence_name is not None and img["file_name"].split("/")[0] != sequence_name:
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
        raise FileNotFoundError(f"Could not read first frame: {frames_info[sorted_frames[0]]}")
    H, W = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    actions = panel_actions or getattr(cfg, "PANEL_ACTIONS", None) or sorted(cfg.ACTION_COLORS_BGR.keys())
    track_ids = panel_track_ids if panel_track_ids is not None else getattr(cfg, "PANEL_TRACK_IDS", None)

    worker_times = defaultdict(lambda: defaultdict(float))

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

                worker_times[worker_id][action.lower()] += frame_time

                box_color = cfg.ACTION_COLORS_BGR.get(action, cfg.ACTION_COLORS_BGR.get("unknown", (150, 150, 150)))

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

                # Label -- thinner, smaller text than before (thickness=1,
                # smaller font_scale) so it doesn't look heavy/blocky on
                # screen.
                label = f"#{det['tracked_id']}  {action}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                thickness = 1

                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                pad = 4

                label_top = max(0, y1 - th - 2 * pad)
                label_bottom = y1
                if label_top == 0:
                    label_bottom = th + 2 * pad

                cv2.rectangle(
                    frame, (x1, label_top), (x1 + tw + 2 * pad, label_bottom), box_color, -1,
                )

                text_y = label_bottom - pad if label_top == 0 else y1 - pad
                cv2.putText(
                    frame, label, (x1 + pad, text_y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA,
                )

            # draw_panel returns a NEW frame (addWeighted doesn't blend in
            # place) -- must reassign, or the panel silently disappears.
            elapsed_sec = frame_id * frame_time
            frame = draw_panel(frame, worker_times, cfg, actions, elapsed_sec, track_ids=track_ids)

            writer.write(frame)
    finally:
        writer.release()

    if n_missing_frames:
        print(f"WARNING: skipped {n_missing_frames} unreadable frame(s).")
    print(f"Video saved to {output_path}")