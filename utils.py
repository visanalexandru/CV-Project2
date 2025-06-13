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
    x0, y0 = point
    return abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)

def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def get_assignment(objects1, objects2, frame1, frame2, fundamentalMatrix):
    costs = np.ones((len(objects1), len(objects2))) * 3000

    for i, object1 in enumerate(objects1): 
        bounding_box1 = object1["bbox"]
        center1 = get_bbox_center(bounding_box1)

        for j, object2 in enumerate(objects2):
            bounding_box2 = object2["bbox"]
            center2 = get_bbox_center(bounding_box2)

            if object1["class"] != object2["class"]:
                costs[i][j] = 3000
                continue
            
            epiline = cv.computeCorrespondEpilines(np.array([center1]), 1, fundamentalMatrix)[0][0]
            distance = distance_point_to_line(center2, epiline)
            if distance > 50:
                costs[i, j] = 3000
            else:
                costs[i, j] = distance

    assignment = linear_sum_assignment(costs) 

    indices_1, indices_2 = assignment
    cost = costs[indices_1, indices_2].sum()

    unmatched = [i for i in range(len(objects1)) if i not in indices_1]
    cost += len(unmatched) * 3000  # Penalize unmatched objects

    return assignment, cost, unmatched