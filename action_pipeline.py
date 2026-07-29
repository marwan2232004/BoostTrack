import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

def frame_path(frames_dir, frame_id):
    return os.path.join(frames_dir, f"{frame_id:06d}.jpg")

def get_clip_frame_indices(center_frame_id, num_frames, min_frame, max_frame):
    half = num_frames // 2
    idxs = np.arange(center_frame_id - half, center_frame_id - half + num_frames)
    return np.clip(idxs, min_frame, max_frame)

def crop_frame(frames_dir, frame_id, box_xyxy, input_size):
    frame = cv2.imread(frame_path(frames_dir, frame_id))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H0, W0 = frame.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
    cx2, cy2 = min(W0, int(x2)), min(H0, int(y2))
    if cx2 <= cx1:
        cx2 = min(W0, cx1 + 1); cx1 = max(0, cx2 - 1)
    if cy2 <= cy1:
        cy2 = min(H0, cy1 + 1); cy1 = max(0, cy2 - 1)
    crop = frame[cy1:cy2, cx1:cx2]
    return cv2.resize(crop, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

def build_clip(frames_dir, center_frame_id, box_xywh, meta, min_frame, max_frame):
    x, y, w, h = box_xywh
    x1, y1, x2, y2 = x, y, x + w, y + h
    padding_ratio = meta["padding_ratio"]
    if padding_ratio > 0:
        pad_x, pad_y = (x2 - x1) * padding_ratio, (y2 - y1) * padding_ratio
        x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y

    idxs = get_clip_frame_indices(center_frame_id, meta["num_frames"], min_frame, max_frame)
    crops = np.stack([crop_frame(frames_dir, int(i), (x1, y1, x2, y2), meta["input_size"]) for i in idxs], axis=0)

    clip = crops.astype(np.float32) / 255.0
    clip = (clip - meta["mean"]) / meta["std"]
    return torch.from_numpy(clip).permute(3, 0, 1, 2).float()

@torch.no_grad()
def classify_samples(model, samples, batch_size, frames_dir, meta, min_frame, max_frame, device):
    results = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        clips = torch.stack([
            build_clip(frames_dir, s["frame_id"], s["box"], meta, min_frame, max_frame)
            for s in batch
        ]).to(device)
        probs = F.softmax(model(clips), dim=1).cpu().numpy()
        for p in probs:
            idx = int(p.argmax())
            results.append((meta["classes"][idx], float(p[idx])))
    return results

def propagate_actions(frame_ids_sorted, samples):
    if not samples:
        return {fid: ("unknown", 0.0) for fid in frame_ids_sorted}
    result, si = {}, 0
    current = samples[0][1:]
    for fid in frame_ids_sorted:
        while si < len(samples) and samples[si][0] <= fid:
            current = samples[si][1:]
            si += 1
        result[fid] = current
    return result

def run_action_pipeline(detections, action_model, meta, cfg, device):
    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    tracks = defaultdict(dict)
    for frame_id, dets in detections.items():
        for d in dets:
            tracks[d["tracked_id"]][frame_id] = (d["x"], d["y"], d["w"], d["h"])

    total_frames = len(os.listdir(cfg.FRAMES_DIR))
    min_frame, max_frame = 1, total_frames
    sample_stride = max(1, round(fps * cfg.ACTION_SAMPLE_INTERVAL_SEC))

    samples_to_run = []
    track_frame_lists = {}
    for track_id, frame_box_map in tracks.items():
        frame_ids_sorted = sorted(frame_box_map.keys())
        track_frame_lists[track_id] = frame_ids_sorted
        last_sampled = -10**9
        for fid in frame_ids_sorted:
            if fid - last_sampled >= sample_stride:
                samples_to_run.append({"track_id": track_id, "frame_id": fid, "box": frame_box_map[fid]})
                last_sampled = fid

    print(f"Running action model on {len(samples_to_run)} sampled clips...")
    predictions = classify_samples(
        action_model, samples_to_run, cfg.ACTION_BATCH_SIZE,
        cfg.FRAMES_DIR, meta, min_frame, max_frame, device
    )

    track_samples = defaultdict(list)
    for s, (label, conf) in zip(samples_to_run, predictions):
        track_samples[s["track_id"]].append((s["frame_id"], label, conf))

    final_actions = {
        track_id: propagate_actions(frame_ids_sorted, track_samples[track_id])
        for track_id, frame_ids_sorted in track_frame_lists.items()
    }

    merged_detections = defaultdict(list)
    for frame_id, dets in detections.items():
        for d in dets:
            label, conf = final_actions[d["tracked_id"]].get(frame_id, ("unknown", 0.0))
            merged = dict(d)
            merged["action"] = label
            merged["action_conf"] = conf
            merged_detections[frame_id].append(merged)

    return merged_detections, fps