from video import *
import numpy as np
import ultralytics
import cv2 as cv
from utils import *
from tqdm import tqdm 
from ultralytics import RTDETR
from multiprocessing import Pool
import os
import glob

JUMP_SIZE = 10 

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

# Removes those points that would be outside the frame when transformed to the other camera's frame.
def filter_points_homography(points, H):
    transformed_points = cv.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), H).squeeze()

    in_bounds = np.logical_and(
        np.logical_and(transformed_points[:, 0] >= 0, transformed_points[:, 0] < FRAME_WIDTH),
        np.logical_and(transformed_points[:, 1] >= 0, transformed_points[:, 1] < FRAME_HEIGHT)
    )

    return points[in_bounds]

def similarity_epipolar(query, reference, F, H, H_inv, visualize=False):
    moving_pixels_query = query.moving_pixels()
    moving_pixels_reference = reference.moving_pixels()

    # Get the actual coordinates of the moving pixels
    rows, cols = np.where(moving_pixels_query)
    points_query = np.stack((cols, rows), axis=1) 

    rows, cols = np.where(moving_pixels_reference)
    points_reference = np.stack((cols, rows), axis=1) 

    if len(points_query) == 0 or len(points_reference) == 0:
        return None

    # Filter points that would be outside the frame when transformed to the other camera's frame.
    points_query = filter_points_homography(points_query, H)
    points_reference = filter_points_homography(points_reference, H_inv)

    if len(points_query) == 0 or len(points_reference) == 0:
        return None

    # Downsample points to avoid too many points
    if len(points_query) > 6000:
        indices = np.random.choice(len(points_query), 6000, replace=False)
        points_query = points_query[indices]

    if len(points_reference) > 6000:
        indices = np.random.choice(len(points_reference), 6000, replace=False)
        points_reference = points_reference[indices]

    # Compute the epipolar lines for the query points in the reference frame.
    lines = cv.computeCorrespondEpilines(points_query.reshape(-1, 1, 2), 1, F).squeeze()
    score = 0

    # Just prepare the plots if visualization is enabled.
    if visualize:
        query_plot = query.raw().copy()
        reference_plot = reference.raw().copy()

    chosen = np.zeros(len(points_reference), dtype=bool)
    for i, line in enumerate(lines):
        distances = distance_point_to_line(points_reference, line)
        distances[chosen] = np.inf  # Ignore already chosen points

        minimum_index = np.argmin(distances)
        minimum_distance = distances[minimum_index]

        if minimum_distance < 5:
            score+=1

            # Mark this point as chosen so we don't use it again.
            chosen[minimum_index] = True

            if visualize:
                color = np.random.randint(0, 255, size=3).tolist()
                cv.circle(query_plot, (points_query[i][0], points_query[i][1]), 10, color, -1)
                close_point = points_reference[minimum_index]
                cv.circle(reference_plot, (close_point[0], close_point[1]), 10, color, -1)
                draw_epiline(line, reference_plot, color=color) 
    
    if visualize:
        fig, axs = plt.subplots(1, 2, figsize=(20, 30))
        axs[0].imshow(moving_pixels_query)
        axs[1].imshow(moving_pixels_reference)
        plt.show()
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 30))
        axs[0].imshow(query_plot)
        axs[0].set_title("Query Frame")
        axs[0].axis('off')
        axs[1].imshow(reference_plot)
        axs[1].set_title("Reference Frame")
        axs[1].axis('off')
        plt.show()

    return score

query = None
reference = None

def overlap_score(reference_start_index, F, H, H_inv):
    global query, reference
    num_query_frames = query.num_frames()
    score = 0
    for t in range(0, num_query_frames, JUMP_SIZE):
        query_frame = query.get_frame(t)
        reference_frame = reference.get_frame(reference_start_index + t)

        cost = similarity_epipolar(query_frame, reference_frame, F, H, H_inv)
        if cost is not None:
            score += cost

    return (reference_start_index, score)

def overlap_score_star(args):
    return overlap_score(*args)

def solve_pair(query_path, reference_path):
    global fundamental_matrices, homography_matrices
    global query, reference

    print("Processing pair:", query_path, reference_path)

    print("Loading query and reference videos...")
    query = load_video(query_path)
    reference = load_video(reference_path)

    print("Doing tracking on query and reference videos...")
    query.compute_moving_pixels()
    reference.compute_moving_pixels()

    print("Establishing cameras...")
    camera_query = query.get_camera()
    camera_reference = reference.get_camera()

    print("Query camera:", camera_query)
    print("Reference camera:", camera_reference)

    F = fundamental_matrices[camera_query][camera_reference] 

    H = homography_matrices[camera_query][camera_reference]
    H_inv = homography_matrices[camera_reference][camera_query]

    num_query_frames = query.num_frames()
    num_reference_frames = reference.num_frames()

    with Pool() as pool:
        args = [(i, F, H, H_inv) for i in range(0, num_reference_frames - num_query_frames + 1, JUMP_SIZE)]
        results = list(tqdm(pool.imap(overlap_score_star, args), total=len(args)))

    argmax = np.argmax([result[1] for result in results])
    print(results[argmax])

queries = glob.glob("train/task1/*_query.mp4")
queries = sorted(queries)

references = glob.glob("train/task1/*_reference.mp4")
references = sorted(references)


for query_path, reference_path in zip(queries, references):
    solve_pair(query_path, reference_path)

