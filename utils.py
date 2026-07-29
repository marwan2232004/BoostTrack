import glob
import os
import cv2
import json
import random
import subprocess
import sys
import numpy as np
import shutil


def write_results_no_score(filename, results):
    """Writes results in MOT style to filename."""
    save_format = "{frame},{id},{x1},{y1},{w},{h},{c},-1,-1,-1\n"
    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids, conf in results:
            for tlwh, track_id, c in zip(tlwhs, track_ids, conf):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                    c=round(c, 2)
                )
                f.write(line)


def filter_targets(online_targets, aspect_ratio_thresh, min_box_area):
    """Removes targets not meeting threshold criteria.

    Returns (list of tlwh, list of ids).
    """
    online_tlwhs = []
    online_ids = []
    online_conf = []
    for t in online_targets:
        tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
        tid = t[4]
        tc = t[5]
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_conf.append(tc)
    return online_tlwhs, online_ids, online_conf


def dti(txt_path, save_path, n_min=25, n_dti=20):
    def dti_write_results(filename, results):
        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n"
        with open(filename, "w") as f:
            for i in range(results.shape[0]):
                frame_data = results[i]
                frame_id = int(frame_data[0])
                track_id = int(frame_data[1])
                x1, y1, w, h = frame_data[2:6]
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=-1)
                f.write(line)

    seq_txts = sorted(glob.glob(os.path.join(txt_path, "*.txt")))
    # breakpoint()
    for seq_txt in seq_txts:
        seq_name = seq_txt.replace("\\", "/").split("/")[-1]  ## To better play along with windows paths
        print(seq_name)
        seq_data = np.loadtxt(seq_txt, dtype=np.float64, delimiter=",")
        min_id = int(np.min(seq_data[:, 1]))
        max_id = int(np.max(seq_data[:, 1]))
        seq_results = np.zeros((1, 10), dtype=np.float64)
        tracklets_to_remove = []
        for track_id in range(min_id, max_id + 1):
            index = seq_data[:, 1] == track_id
            tracklet = seq_data[index]
            tracklet_dti = tracklet
            if tracklet.shape[0] == 0:
                continue
            n_frame = tracklet.shape[0]
            # for idx in range(len(tracklet)):
            #     print(tracklet[idx])
                # print(tracklet[:, 0])
            n_conf = np.sum(tracklet[:, 6] > 0.5)
            if n_frame > n_min:
                frames = tracklet[:, 0]
                frames_dti = {}
                for i in range(0, n_frame):
                    right_frame = frames[i]
                    if i > 0:
                        left_frame = frames[i - 1]
                    else:
                        left_frame = frames[i]
                    # disconnected track interpolation
                    if 1 < right_frame - left_frame < n_dti:
                        num_bi = int(right_frame - left_frame - 1)
                        right_bbox = tracklet[i, 2:6]
                        left_bbox = tracklet[i - 1, 2:6]
                        for j in range(1, num_bi + 1):
                            curr_frame = j + left_frame
                            curr_bbox = (curr_frame - left_frame) * (right_bbox - left_bbox) / (
                                right_frame - left_frame
                            ) + left_bbox
                            frames_dti[curr_frame] = curr_bbox
                num_dti = len(frames_dti.keys())
                if num_dti > 0:
                    data_dti = np.zeros((num_dti, 10), dtype=np.float64)
                    for n in range(num_dti):
                        data_dti[n, 0] = list(frames_dti.keys())[n]
                        data_dti[n, 1] = track_id
                        data_dti[n, 2:6] = frames_dti[list(frames_dti.keys())[n]]
                        data_dti[n, 6:] = [1, -1, -1, -1]
                    tracklet_dti = np.vstack((tracklet, data_dti))
            seq_results = np.vstack((seq_results, tracklet_dti))
        save_seq_txt = os.path.join(save_path, seq_name)
        seq_results = seq_results[1:]
        seq_results = seq_results[seq_results[:, 0].argsort()]
        dti_write_results(save_seq_txt, seq_results)


def read_detections(file_path):
    detections = {}
    with open(file_path, "r") as f:
        for line in f:
            data = line.strip().split(",")
            frame_id = int(data[0])
            object_id = int(data[1])
            x, y, w, h = map(float, data[2:6])
            score = float(data[6])

            if frame_id not in detections:
                detections[frame_id] = []
            item = {
                "tracked_id": object_id,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "confidence": score,
            }
            detections[frame_id].append(item)
    return detections


def extract_frames(*, video_path, output_folder):
    """
    Extracts frames from a video and saves them as images in a specified folder.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the output folder where frames will be saved.

    Returns:
        str: Path to the folder containing the extracted frames.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"{frame_count:06d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'")
    return output_folder


# Generating the folder structure used for prediction
def generate_mot20_structure(root: str) -> str:
    base = os.path.join(root, "BoostTrack/data/MOT20")
    test = os.path.join(base, "test")
    test_1 = os.path.join(test, "MOT20-01")
    train = os.path.join(base, "train")
    frames = os.path.join(test_1, "img1")
    folders = [base, test, train, test_1, frames]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    return frames


def run_mot(video_path: str, frames_path: str, model_path: str, root: str):
    if video_path is None and frames_path is None:
        raise ValueError("You must provide either video_path or frames_path.")

    dst_frames = generate_mot20_structure(root)

    if frames_path is None:
        extract_frames(
            video_path=video_path,
            output_folder=dst_frames,
        )
    else:
        shutil.copytree(frames_path, dst_frames, dirs_exist_ok=True)

    original_dir = os.getcwd()
    boosttrack_dir = os.path.join(original_dir, "BoostTrack")

    try:
        os.chdir(boosttrack_dir)

        # Convert MOT20 dataset to COCO format
        print("Converting MOT20 to COCO...")
        subprocess.run(
            ["python", "data/tools/convert_mot20_to_coco.py"],
            check=True,
        )

        # Run BoostTrack
        print("Running BoostTrack pipeline...")
        
        cmd = [
            "python", "main.py",
            "--dataset", "mot20",
            "--exp_name", "BTPP",
            "--detection_model_path", model_path,
            "--model_type", "yolov26",
            "--test_dataset"
        ]

        result = subprocess.run(
            cmd,
            cwd=boosttrack_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("\n BoostTrack Execution Failed! Traceback below:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"BoostTrack main.py failed with exit code {result.returncode}")

        print("BoostTrack tracking finished successfully!")

    finally:
        os.chdir(original_dir)


def show_results(output_path:str, results_path:str, split_path:str, annotations_path:str, fps:int = 25):
    """
    Visualizes tracking results by drawing bounding boxes and tracked IDs on video frames,
    then compiles them into a video.
    
    Args:
        output_path (str): Path to save the output video with visualized tracking results.
        results_path (str): Path to the detection results file (used by `read_detections`).
        split_path (str): Path to the dataset split (e.g., 'train', 'val', 'test') folder containing image frames.
        annotations_path (str): Path to the COCO-format JSON file containing image/frame metadata.
    Notes:
        - Only frames from 'MOT20-01' are processed.
    """
    color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(30)]
    
    # Load JSON data
    with open(annotations_path, "r") as f:
        data = json.load(f)
    
    # Extract frames information
    frames_info = {}
    for img in data["images"]:
        frame_id = img["frame_id"]
        if img["file_name"].split('/')[0] != 'MOT20-01': continue  
        img_path = os.path.join(split_path, img["file_name"])
        frames_info[frame_id] = img_path  # Store frame path by frame_id
    
    # Sort frames by frame_id
    sorted_frames = sorted(frames_info.keys())
    
    # Load detections
    detections = read_detections(results_path)
    
    
    # Load first frame to get video dimensions
    first_frame = cv2.imread(frames_info[sorted_frames[0]])
    
    H, W, _ = first_frame.shape
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    # Process each frame
    for frame_id in sorted_frames:
        frame_path = frames_info[frame_id]
        frame = cv2.imread(frame_path)
    
        if frame_id in detections:
            for detection in detections[frame_id]:
                x1 = int(detection['x'])
                y1 = int(detection['y'])
                x2 = x1 + int(detection['w'])
                y2 = y1 + int(detection['h'])
                obj_id = detection['tracked_id']
                score = detection['confidence']
    
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),  color[obj_id % len(color)], 2)
    
                # Put ID on top of the box
                cv2.putText(frame, f"ID: {obj_id}, {score}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,  color[obj_id % len(color)], 2)
    
        # Write frame to video
        video_writer.write(frame)
    
    # Release resources
    video_writer.release()
    print(f"Video saved as {output_path}")