from video import *
from utils import *
import numpy as np
import cv2 as cv

def find_best_intersection(objects, bounding_box, threshold=0.5):
    max_iou = 0
    best = None

    for object in objects:
        bb = object["bbox"] 

        iou = bb_intersection_over_union(bb, bounding_box)
        if iou > threshold and iou > max_iou:
            best = object
            max_iou = iou

    return best 

def find_closest_object(objects, position, threshold=100):
    min_distance = np.inf
    best = None

    for object in objects:
        bb = object["bbox"]
        center= bounding_box_center(bb) 
        distance = np.linalg.norm(np.array(center) - np.array(position))

        if distance < threshold and distance < min_distance:
            best = object
            min_distance = distance

    return best

class Trajectory:
    def __init__(self, boxes):
        self.boxes_ = boxes
    
    def bounding_boxes(self):
        return self.boxes_
    
    def trajectory(self):
        result = []
        for box in self.boxes_:
            point = (box[0] + box[2]) // 2, box[3] 
            result.append(point)
        return np.array(result)

def compute_trajectory(video, initial_bbox, visualize=False):
    bounding_boxes = []
    kalman = cv.KalmanFilter(8, 4)
    kalman.measurementMatrix = np.array(
                [[1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                ], np.float32)

    kalman.transitionMatrix = np.array(
                [[1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
                ], np.float32)

    kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01

    kalman.measurementNoiseCov = np.array(
                [[200, 0, 0, 0],
                [0, 200, 0, 0],
                [0, 0, 1000, 0],
                [0, 0, 0, 1000]], np.float32)

    kalman.statePre = np.array([[initial_bbox[0]],
                                [initial_bbox[1]],
                                [initial_bbox[2] - initial_bbox[0]],
                                [initial_bbox[3] - initial_bbox[1]],
                                [0],
                                [0],
                                [0],
                                [0]], np.float32)

    kalman.statePost = np.array([[initial_bbox[0]],
                                [initial_bbox[1]],
                                [initial_bbox[2] - initial_bbox[0]],
                                [initial_bbox[3] - initial_bbox[1]],
                                [0],
                                [0],
                                [0],
                                [0]], np.float32)

    for frame_index in range(0, video.num_frames()):
        frame = video.get_frame(frame_index)
        objects = frame.objects()

        if visualize:
            plot = frame.raw().copy()
            for object in objects:
                cv.rectangle(plot, object["bbox"][:2], object["bbox"][2:], (0, 255, 0), 2)

        prediction = kalman.predict()[:4].flatten().astype(np.int32)
        prediction = (prediction[0], prediction[1], prediction[0] + prediction[2], prediction[1] + prediction[3])

        bounding_boxes.append(prediction)

        # Proritize finding the best intersection with existing objects
        best_intersection = find_best_intersection(objects, prediction, threshold=0.2)

        found = None 
        if best_intersection is not None:
            found = best_intersection["bbox"]
        else:
            # If that fails, find the closest pbject.
            center = bounding_box_center(prediction)
            closest_object = find_closest_object(objects, center, threshold=50)
            if closest_object is not None:
                found = closest_object["bbox"]

        if found:
            kalman.correct(np.array([[found[0]], [found[1]], [found[2] - found[0]], [found[3] - found[1]]], np.float32))

        if visualize:
            cv.rectangle(plot, prediction[:2], prediction[2:], (255, 0, 0), 2)
            cv.imshow("tracking", plot)
            cv.waitKey(100)

    if visualize:
        cv.destroyAllWindows()
    
    return Trajectory(bounding_boxes) 