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


# Iterates through all the object bounding boxes and extracts a point
# in the bounding box to be used as a keypoint. It's usually a point on
# the lowest part of the bounding box. This way, it's approximately on the
# street, so we can apply the homography matrix to map it to another camera
# view.
def get_keypoints(frame, homography):
    objects = frame.get_objects()
    keypoints = []

    for object in objects:
        x1, y1, x2, y2 = object["bbox"]

        keypoint = np.array(((x1+x2)//2, y2)).reshape(1, 1, 2).astype(np.float32)
        transformed_keypoint = cv.perspectiveTransform(keypoint, homography).squeeze()
        transformed_keypoint = transformed_keypoint.astype(np.int32)

        # Don't include those keypoints that are not visible in the second camera.
        if transformed_keypoint[0] < 0 or transformed_keypoint[0] >= FRAME_WIDTH:
            continue
        if transformed_keypoint[1] < 0 or transformed_keypoint[1] >= FRAME_HEIGHT:
            continue
            
        keypoints.append({
            "position": ((x1+x2)//2, y2),
            "class": object["class"]
        })

    return keypoints

def get_assignment(keypoints1, keypoints2, homography):
    max_distance = np.linalg.norm([FRAME_WIDTH, FRAME_HEIGHT])
    costs = np.ones((len(keypoints1), len(keypoints2))) * max_distance 

    for i, a in enumerate(keypoints1): 
        for j, b in enumerate(keypoints2):
            pos1 = a["position"]
            pos2 = b["position"]

            if a["class"] != b["class"]:
                costs[i, j] = max_distance 
                continue
           
            pos1_tmp = np.array([[pos1]]).astype(np.float32)
            transformed = cv.perspectiveTransform(pos1_tmp, homography).squeeze()

            distance = np.linalg.norm(transformed - np.array(pos2))
            costs[i, j] = distance

    assignment = linear_sum_assignment(costs) 

    indices_1, indices_2 = assignment
    cost = costs[indices_1, indices_2].sum()

    unmatched = [i for i in range(len(keypoints1)) if i not in indices_1]
    cost += len(unmatched) * max_distance # Penalize unmatched objects

    unmatched = [i for i in range(len(keypoints2)) if i not in indices_2]
    cost += len(unmatched) * max_distance # Penalize unmatched objects

    return assignment, cost, unmatched