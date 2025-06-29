from video import *
import numpy as np
import ultralytics
import cv2 as cv
from utils import *
from tqdm import tqdm 
from ultralytics import RTDETR
from multiprocessing import Pool
from tracking import *
import glob
import os
import argparse
from pathlib import Path

fundamental_matrices = {
    "A" : {
        "B": np.load("fundamental_matrices/AB.npy"),
        "C": np.load("fundamental_matrices/AC.npy"),
    },
    "B": {
        "A": np.load("fundamental_matrices/BA.npy"),
        "C": np.load("fundamental_matrices/BC.npy"),
    },
    "C": {
        "B": np.load("fundamental_matrices/CB.npy"),
        "A": np.load("fundamental_matrices/CA.npy"),
    }
}

homography_matrices = {
    "A" : {
        "B": np.load("homography_matrices/AB.npy"),
        "C": np.load("homography_matrices/AC.npy"),
    },
    "B": {
        "A": np.load("homography_matrices/BA.npy"),
        "C": np.load("homography_matrices/BC.npy"),
    },
    "C": {
        "B": np.load("homography_matrices/CB.npy"),
        "A": np.load("homography_matrices/CA.npy"),
    }
}

def read_box(path):
    with open(path, "r") as f:
        lines = f.readlines()
    _, box = lines[:2]

    box = box.strip()
    _index, x1, y1, x2, y2 = list(map(int , box.split(" ")))
    return (x1, y1, x2, y2)

def output_trajectory_video(video, trajectory, output_path):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, 20.0, (FRAME_WIDTH,  FRAME_HEIGHT))

    bboxes = trajectory.bounding_boxes()

    for i in range(video.num_frames()):
        copy = video.get_frame(i).raw().copy()
        cv.rectangle(copy, bboxes[i][:2], bboxes[i][2:], (0, 255, 0), 2)
        out.write(copy)
    
    out.release()

def point_inside_window(point):
    x, y = point
    return 0 <= x < FRAME_WIDTH and 0 <= y < FRAME_HEIGHT

# Check if any of the four corners of the bounding box are inside the window.
def bounding_box_inside_window(bounding_box):
    x1, y1, x2, y2 = bounding_box

    point1 = (x1, y1)
    point2 = (x1, y2)
    point3 = (x2, y1)
    point4 = (x2, y2) 
    return point_inside_window(point1) or point_inside_window(point2) or point_inside_window(point3) or point_inside_window(point4)

def is_degenerate(bounding_box):
    x1, y1, x2, y2 = bounding_box
    return x1 >= x2 or y1 >= y2

def output_trajectory_text(video, trajectory, output_path):
    boxes = trajectory.bounding_boxes()

    with open(output_path, "w") as file:
        file.write(f"{video.num_frames()} -1 -1 -1 -1\n")

        for i in range(len(boxes)):
            bounding_box = trajectory.bounding_boxes()[i]

            if not bounding_box_inside_window(bounding_box):
                continue
            if is_degenerate(bounding_box):
                continue

            x1, y1, x2, y2 = bounding_box

            x1 = max(0, x1)
            x1 = min(FRAME_WIDTH - 1, x1)

            x2 = max(0, x2)
            x2 = min(FRAME_WIDTH - 1, x2)

            y1 = max(0, y1)
            y1 = min(FRAME_HEIGHT - 1, y1)

            y2 = max(0, y2)
            y2 = min(FRAME_HEIGHT - 1, y2)

            file.write(f"{i} {x1} {y1} {x2} {y2}\n")

def solve_pair(vid1_path, vid2_path, output_dir):
    global fundamental_matrices, homography_matrices

    print("Processing pair:", vid1_path, vid2_path)

    print("Loading videos...")
    a = load_video(vid1_path)
    b = load_video(vid2_path)

    box_path = vid1_path.replace(".mp4", ".txt")
    initial_bbox = read_box(box_path)

    print("Establishing cameras...")
    camera_a = a.get_camera()
    camera_b = b.get_camera()

    H = homography_matrices[camera_a][camera_b]
    H_inv = homography_matrices[camera_b][camera_a]

    print("Doing object detection...")
    a.do_tracking(visualize=False)
    b.do_tracking(visualize=False)

    # First, compute the trajectory of the marked vehicle.
    trajectory_a = compute_trajectory(a, initial_bbox, visualize=False)

    # Then, find the best matching trajectory in video b.
    objects_b = []
    # Take the objects in the first 10 frames, instead of just the objects in the first one.
    # This helps reduce the chance of not finding the target in the first frame.
    for i in range(min(10, b.num_frames())):
        objects_b += b.get_frame(i).objects()

    best_trajectory = None
    min_cost = np.inf

    for object in objects_b:
        trajectory_b = compute_trajectory(b, object["bbox"], visualize=False)

        cost = compare_trajectory(trajectory_a.trajectory(), trajectory_b.trajectory(), H, H_inv, threshold=150)
        if cost < min_cost:
            min_cost = cost
            best_trajectory = trajectory_b

    print("Writing results...")

    vid1_base = os.path.basename(vid1_path)
    vid2_base = os.path.basename(vid2_path)

    output_trajectory_video(a, trajectory_a, os.path.join(output_dir, "task2", vid1_base)) 
    output_trajectory_video(b, best_trajectory, os.path.join(output_dir, "task2", vid2_base)) 

    textfile1_base = f"{vid1_base[:2]}_1_predicted.txt"
    textfile2_base = f"{vid2_base[:2]}_2_predicted.txt"

    output_trajectory_text(a, trajectory_a, os.path.join(output_dir, "task2", textfile1_base)) 
    output_trajectory_text(b, best_trajectory, os.path.join(output_dir, "task2", textfile2_base)) 

    return trajectory_a, best_trajectory

parser = argparse.ArgumentParser("task2")
parser.add_argument("evaluation_dir", help="Directory where all the video pairs are stored.")
parser.add_argument("output_dir", help="Directory where the solution files will be created.")
args = parser.parse_args()

evaluation_dir = args.evaluation_dir
output_dir = args.output_dir

vids1 = glob.glob(os.path.join(evaluation_dir, "task2/*_1.mp4"))
vids1 = sorted(vids1)

vids2 = glob.glob(os.path.join(evaluation_dir, "task2/*_2.mp4"))
vids2 = sorted(vids2)

# Make sure the output directory exists.
Path(os.path.join(output_dir, "task2")).mkdir(parents=True, exist_ok=True)

for vid1, vid2 in zip(vids1, vids2):
    solve_pair(vid1, vid2, output_dir)