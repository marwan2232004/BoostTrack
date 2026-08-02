import os
import json
import cv2
import numpy as np
from collections import defaultdict

# ============================================================
# Style constants
# ============================================================
PANEL_BG_BGR = (255, 255, 255)
PANEL_BORDER_BGR = (50, 50, 55)
PANEL_TITLE_BG_BGR = (32, 36, 44)
PANEL_TITLE_FG_BGR = (255, 255, 255)
PANEL_SUB_FG_BGR = (220, 220, 220)
PANEL_BODY_FG_BGR = (40, 40, 40)
PANEL_MUTED_FG_BGR = (0, 0, 0)
BAR_TRACK_BGR = (228, 228, 230)
DIVIDER_BGR = (225, 225, 228)
BADGE_BG_BGR = (32, 36, 44)
BADGE_BORDER_BGR = (210, 210, 210)
BADGE_FG_BGR = (255, 255, 255)

WORKING_COLOR_BGR = (89, 199, 52)     
NOT_WORKING_COLOR_BGR = (48, 59, 255)

FONT_TITLE = cv2.FONT_HERSHEY_DUPLEX
FONT_BODY = cv2.FONT_HERSHEY_SIMPLEX
FONT_BADGE = cv2.FONT_HERSHEY_DUPLEX
FONT_VALUE = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================
# Helpers
# ============================================================
def _try_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=8):
    """Draw a rounded rectangle if OpenCV supports it (>=4.5), else plain."""
    try:
        cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_AA, radius)
    except (TypeError, cv2.error):
        cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def _category_of(action, cfg):
    """Map a raw action ('work-sit', 'walk', ...) to its category."""
    return cfg.ACTION_CATEGORY_MAP.get(action.lower(), "unknown")


def _is_working(action, cfg):
    return _category_of(action, cfg) == "working"


def _is_not_working(action, cfg):
    """idle + transit both count as 'not working'."""
    return _category_of(action, cfg) in ("idle", "transit")


def _format_time(seconds):
    """Render seconds as 'M:SS' when over a minute, else 'Ns'."""
    if seconds >= 60:
        m = int(seconds) // 60
        s = int(seconds) % 60
        return f"{m}:{s:02d}"
    return f"{seconds:.0f}s"


# ============================================================
# Panel drawing
# ============================================================
def draw_panel(
    frame,
    worker_times,
    cfg,
    elapsed_sec,
    track_ids=None,
    panel_x=20,
    panel_y=None,
    panel_w=440,
    row_h=64,
    max_rows=None,
    title="Worker Activity",
):
    """
    Draw an overlay panel showing per-worker Working / Not-Working time bars.
    Transit (walk) is counted as Not Working.

    Parameters
    ----------
    worker_times : dict[dict[float]]
        {track_id: {"working": sec, "not_working": sec}}.
    elapsed_sec : float
        Total elapsed video time. Used to scale bar widths.
    track_ids : list | None
        Optional fixed ordering of track IDs (keeps rows stable).
    """
    H, W = frame.shape[:2]
    if panel_y is None:
        panel_y = 20

    ids = list(track_ids) if track_ids is not None else sorted(worker_times.keys())
    if max_rows is not None:
        ids = ids[:max_rows]

    title_h = 34
    legend_h = 28
    header_h = title_h + legend_h
    body_h = row_h * len(ids) + 12
    panel_h = min(header_h + body_h, H - 2 * panel_y)

    # ---------------- Drop shadow ----------------
    shadow_off = 6
    shadow_overlay = frame.copy()
    cv2.rectangle(
        shadow_overlay,
        (panel_x + shadow_off, panel_y + shadow_off),
        (panel_x + panel_w + shadow_off, panel_y + panel_h + shadow_off),
        (0, 0, 0),
        -1,
    )
    frame = cv2.addWeighted(shadow_overlay, 0.30, frame, 0.70, 0)

    # ---------------- Panel background ----------------
    overlay = frame.copy()
    _try_rounded_rect(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        PANEL_BG_BGR,
        -1,
        radius=12,
    )
    frame = cv2.addWeighted(overlay, 0.94, frame, 0.06, 0)

    # Outer border
    _try_rounded_rect(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        PANEL_BORDER_BGR,
        1,
        radius=12,
    )

    # ---------------- Title bar ----------------
    # Dark strip across the top of the panel (clipped inside the rounded panel).
    title_rect_overlay = frame.copy()
    cv2.rectangle(
        title_rect_overlay,
        (panel_x + 1, panel_y + 1),
        (panel_x + panel_w - 1, panel_y + title_h),
        PANEL_TITLE_BG_BGR,
        -1,
        cv2.LINE_AA,
    )
    frame = cv2.addWeighted(title_rect_overlay, 0.95, frame, 0.05, 0)
    # Subtle bottom shadow under the title strip
    cv2.line(
        frame,
        (panel_x + 1, panel_y + title_h),
        (panel_x + panel_w - 1, panel_y + title_h),
        (22, 24, 30),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        title,
        (panel_x + 14, panel_y + 23),
        FONT_TITLE,
        0.6,
        PANEL_TITLE_FG_BGR,
        1,
        cv2.LINE_AA,
    )

    # Elapsed time on the right of the title
    mm = int(elapsed_sec) // 60
    ss = int(elapsed_sec) % 60
    elapsed_txt = f"{mm:02d}:{ss:02d}"
    (tw_et, _), _ = cv2.getTextSize(elapsed_txt, FONT_TITLE, 0.55, 1)
    cv2.putText(
        frame,
        elapsed_txt,
        (panel_x + panel_w - tw_et - 14, panel_y + 23),
        FONT_TITLE,
        0.55,
        PANEL_SUB_FG_BGR,
        1,
        cv2.LINE_AA,
    )

    # ---------------- Legend ----------------
    legend_y = panel_y + title_h + 18
    swatches = [
        ("Working", WORKING_COLOR_BGR),
        ("Not Working", NOT_WORKING_COLOR_BGR),
    ]
    sx = panel_x + 14
    for name, color in swatches:
        cv2.rectangle(
            frame,
            (sx, legend_y - 11),
            (sx + 12, legend_y + 1),
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            name,
            (sx + 18, legend_y),
            FONT_BODY,
            0.42,
            PANEL_BODY_FG_BGR,
            1,
            cv2.LINE_AA,
        )
        (tw, _), _ = cv2.getTextSize(name, FONT_BODY, 0.42, 1)
        sx += 18 + tw + 22

    # ---------------- Worker rows ----------------
    label_x = panel_x + 62
    bar_x = panel_x + 150
    value_x = panel_x + panel_w - 14
    bar_max_w = value_x - bar_x - 60
    bar_h = 8

    y = panel_y + header_h + 6
    for tid in ids:
        wt = worker_times.get(tid, {})
        working_sec = float(wt.get("working", 0.0))
        not_working_sec = float(wt.get("not_working", 0.0))

        # --- Worker ID badge (left column) ---
        badge_cx = panel_x + 26
        badge_cy = y + row_h // 2
        badge_r = 15
        cv2.circle(frame, (badge_cx, badge_cy), badge_r, BADGE_BG_BGR, -1, cv2.LINE_AA)
        cv2.circle(
            frame, (badge_cx, badge_cy), badge_r, BADGE_BORDER_BGR, 1, cv2.LINE_AA
        )
        id_txt = str(tid)
        (idw, idh), _ = cv2.getTextSize(id_txt, FONT_BADGE, 0.55, 1)
        cv2.putText(
            frame,
            id_txt,
            (badge_cx - idw // 2, badge_cy + idh // 2),
            FONT_BADGE,
            0.55,
            BADGE_FG_BGR,
            1,
            cv2.LINE_AA,
        )

        # --- Working row ---
        r1_y = y + 14
        cv2.putText(
            frame,
            "Working",
            (label_x, r1_y + 7),
            FONT_BODY,
            0.40,
            PANEL_MUTED_FG_BGR,
            1,
            cv2.LINE_AA,
        )
        _draw_bar(
            frame,
            bar_x,
            r1_y,
            bar_max_w,
            bar_h,
            working_sec,
            elapsed_sec,
            WORKING_COLOR_BGR,
        )
        _draw_value(frame, value_x, r1_y + 7, working_sec)

        # --- Not-working row ---
        r2_y = y + 34
        cv2.putText(
            frame,
            "Not Work",
            (label_x, r2_y + 7),
            FONT_BODY,
            0.40,
            PANEL_MUTED_FG_BGR,
            1,
            cv2.LINE_AA,
        )
        _draw_bar(
            frame,
            bar_x,
            r2_y,
            bar_max_w,
            bar_h,
            not_working_sec,
            elapsed_sec,
            NOT_WORKING_COLOR_BGR,
        )
        _draw_value(frame, value_x, r2_y + 7, not_working_sec)

        # Subtle divider between workers
        cv2.line(
            frame,
            (panel_x + 8, y + row_h - 6),
            (panel_x + panel_w - 8, y + row_h - 6),
            DIVIDER_BGR,
            1,
            cv2.LINE_AA,
        )

        y += row_h

    return frame


def _draw_bar(frame, x, y, w, h, value, total, color):
    """Horizontal track + filled portion (filled = value / total)."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), BAR_TRACK_BGR, -1, cv2.LINE_AA)
    frac = 0.0 if total <= 0 else min(1.0, value / total)
    fill_w = int(frac * w)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1, cv2.LINE_AA)


def _draw_value(frame, right_x, baseline_y, value):
    """Right-aligned time label."""
    txt = _format_time(value)
    (tw, _), _ = cv2.getTextSize(txt, FONT_VALUE, 0.42, 1)
    cv2.putText(
        frame,
        txt,
        (right_x - tw, baseline_y),
        FONT_VALUE,
        0.42,
        PANEL_BODY_FG_BGR,
        1,
        cv2.LINE_AA,
    )


# ============================================================
# Video rendering
# ============================================================
def show_results_with_actions(
    output_path,
    merged_detections,
    split_path,
    annotations_path,
    cfg,
    fps=25,
    sequence_name=None,
    panel_track_ids=None,
    panel_max_rows=None,
    output_panel_title="Worker Activity",
):
    """
    Render an annotated MP4 with bounding boxes + the per-worker
    Working / Not-Working panel overlay.

    Transit (walk) is counted as Not Working.
    """
    with open(annotations_path, "r") as f:
        data = json.load(f)

    # ---------------------------------------------------------
    # Filter frames by sequence
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

    track_ids = (
        panel_track_ids
        if panel_track_ids is not None
        else getattr(cfg, "PANEL_TRACK_IDS", None)
    )

    # {tid: {"working": sec, "not_working": sec}}
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
                continue

            for det in merged_detections.get(frame_id, []):
                x1 = max(0, int(det["x"]))
                y1 = max(0, int(det["y"]))
                x2 = min(W - 1, x1 + int(det["w"]))
                y2 = min(H - 1, y1 + int(det["h"]))

                action = det["action"]
                worker_id = det["tracked_id"]

                # Accumulate per-category time (transit -> not_working)
                category = cfg.ACTION_CATEGORY_MAP.get(action.lower(), "unknown")
                if category == "working":
                    worker_times[worker_id]["working"] += frame_time
                elif category in ("idle", "transit"):
                    worker_times[worker_id]["not_working"] += frame_time

                # Per-action color (keeps boxes visually distinct)
                box_color = cfg.ACTION_COLORS_BGR.get(
                    action, cfg.ACTION_COLORS_BGR.get("unknown", (170, 170, 170))
                )

                _draw_detection_box(
                    frame, x1, y1, x2, y2, action, det["tracked_id"], box_color
                )

            elapsed_sec = frame_id * frame_time
            frame = draw_panel(
                frame,
                worker_times,
                cfg,
                elapsed_sec,
                track_ids=track_ids,
                max_rows=panel_max_rows,
                title=output_panel_title,
            )
            writer.write(frame)
    finally:
        writer.release()

    if n_missing_frames:
        print(f"WARNING: skipped {n_missing_frames} unreadable frame(s).")
    print(f"Video saved to {output_path}")


def _draw_detection_box(frame, x1, y1, x2, y2, action, tracked_id, color):
    """Bounding box + label with a subtle accent stripe on the left."""
    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Label
    label = f"#{tracked_id}  {action}"
    font = FONT_BODY
    font_scale = 0.45
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 5
    label_top = max(0, y1 - th - 2 * pad)
    label_bottom = y1
    if label_top == 0:
        label_bottom = th + 2 * pad

    # Label background
    cv2.rectangle(
        frame,
        (x1, label_top),
        (x1 + tw + 2 * pad, label_bottom),
        color,
        -1,
        cv2.LINE_AA,
    )

    # Left accent stripe (darker shade of the box color)
    dark = tuple(max(0, c - 60) for c in color)
    cv2.rectangle(
        frame,
        (x1, label_top),
        (x1 + 3, label_bottom),
        dark,
        -1,
        cv2.LINE_AA,
    )

    text_y = label_bottom - pad if label_top == 0 else y1 - pad
    cv2.putText(
        frame,
        label,
        (x1 + pad + 3, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
