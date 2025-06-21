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
