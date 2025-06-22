import numpy as np    
import cv2 as cv
from video import FRAME_WIDTH, FRAME_HEIGHT
from scipy.optimize import linear_sum_assignment
import torch
import matplotlib.pyplot as plt

def bounding_box_mask(bbox, image_shape):
    x1, y1, x2, y2 = bbox
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask

def draw_epiline(epiline, image, color):
    x0, y0 = map(int, [0, -epiline[2] / epiline[1]])
    x1, y1 = map(int, [image.shape[1], -(epiline[2] + epiline[0] * image.shape[1]) / epiline[1]])
    cv.line(image, (x0, y0), (x1, y1), color, 2)

def distance_point_to_line(point, line):
    a, b, c = line
    return abs(a * point[:, 0]+ b * point[:, 1] + c) / np.sqrt(a**2 + b**2)

def bounding_box_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return np.array([center_x, center_y])

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def bb_intersection(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    return interArea

def compare_trajectory(trajectory1, trajectory2, H, H_inv, threshold=150):
    MAX_DISTANCE = FRAME_WIDTH + FRAME_HEIGHT

    trajectory1 = filter_points_homography(trajectory1, H)
    trajectory2 = filter_points_homography(trajectory2, H_inv)
    
    if len(trajectory2) == 0:
        return len(trajectory1) * MAX_DISTANCE 
    
    transformed = cv.perspectiveTransform(trajectory1.reshape(-1, 1, 2).astype(np.float32), H).squeeze()

    if np.linalg.norm(transformed[0] - trajectory2[0]) > 2*threshold:
        # If the first point is too far, we assume the trajectories are not matching
        return len(trajectory1) * MAX_DISTANCE

    # Compute the minimum matching
    costs = np.ones((len(trajectory1), len(trajectory2))) * MAX_DISTANCE 
    for i in range(len(transformed)):
        distances = np.linalg.norm(transformed[i] - trajectory2, axis=1)
        distances[distances > threshold] = MAX_DISTANCE

        costs[i, :] = distances 

    row_ind, col_ind = linear_sum_assignment(costs)
    total_cost = costs[row_ind, col_ind].sum()

    unassigned = [i for i in range(len(trajectory1)) if i not in row_ind]
    total_cost += len(unassigned) * MAX_DISTANCE

    return total_cost 

# Removes those points that would be outside the frame when transformed to the other camera's frame.
def filter_points_homography(points, H):
    transformed_points = cv.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), H).squeeze()

    in_bounds = np.logical_and(
        np.logical_and(transformed_points[:, 0] >= 0, transformed_points[:, 0] < FRAME_WIDTH),
        np.logical_and(transformed_points[:, 1] >= 0, transformed_points[:, 1] < FRAME_HEIGHT)
    )

    return points[in_bounds]