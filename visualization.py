import os
import json
import cv2

def show_results_with_actions(output_path, merged_detections, split_path, annotations_path, cfg, fps=25):
    """Colors each box by action category and labels with 'ID <n> | <action>'."""
    with open(annotations_path, "r") as f:
        data = json.load(f)

    frames_info = {}
    for img in data["images"]:
        if img["file_name"].split("/")[0] != "MOT20-01":
            continue
        frames_info[img["frame_id"]] = os.path.join(split_path, img["file_name"])

    sorted_frames = sorted(frames_info.keys())
    first_frame = cv2.imread(frames_info[sorted_frames[0]])
    H, W, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for frame_id in sorted_frames:
        frame = cv2.imread(frames_info[frame_id])
        for det in merged_detections.get(frame_id, []):
            x1, y1 = int(det["x"]), int(det["y"])
            x2, y2 = x1 + int(det["w"]), y1 + int(det["h"])
            category = cfg.ACTION_CATEGORY_MAP.get(det["action"], "unknown")
            color = cfg.CATEGORY_COLORS_BGR.get(category, cfg.CATEGORY_COLORS_BGR["unknown"])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {det['tracked_id']} | {det['action']}"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(frame)

    writer.release()
    print(f"Video saved to {output_path}")