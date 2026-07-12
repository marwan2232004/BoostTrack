import cv2
import os
import shutil
import json
import random
import subprocess



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
        subprocess.run(
            ["python", "data/tools/convert_mot20_to_coco.py"],
            check=True,
        )

        # Run BoostTrack
        subprocess.run(
            [
                "python",
                "main.py",
                "--dataset",
                "mot20",
                "--exp_name",
                "BTPP",
                "--detection_model_path",
                model_path,
                "--model_name",
                "yolov26",
                "--test_dataset",
            ],
            check=True,
        )

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