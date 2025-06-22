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

def output_trajectory(video, trajectory, output_path):
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, 20.0, (FRAME_WIDTH,  FRAME_HEIGHT))

    bboxes = trajectory.bounding_boxes()

    for i in range(video.num_frames()):
        copy = video.get_frame(i).raw().copy()
        cv.rectangle(copy, bboxes[i][:2], bboxes[i][2:], (0, 255, 0), 2)
        out.write(copy)
    
    out.release()

vids1 = glob.glob("train/task2/*_1.mp4")
vids1 = sorted(vids1)

vids2 = glob.glob("train/task2/*_2.mp4")
vids2 = sorted(vids2)

def solve_pair(vid1_path, vid2_path):
    global fundamental_matrices, homography_matrices

    print("Processing pair:", vid1_path, vid2_path)

    vid1_base = os.path.basename(vid1_path)
    vid2_base = os.path.basename(vid2_path)

    print(vid1_base, vid2_base)

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
    output_trajectory(a, trajectory_a, os.path.join("results", vid1_base)) 
    output_trajectory(b, best_trajectory, os.path.join("results", vid2_base)) 


for vid1, vid2 in zip(vids1, vids2):
    print(vid1, vid2)
    solve_pair(vid1, vid2)