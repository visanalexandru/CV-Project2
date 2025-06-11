import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# The original shape was 1920x1000
# Change this if you want to use a different resolution.
FRAME_WIDTH = 1920 
FRAME_HEIGHT = 1000 

CAMERA_A_TEMPLATE_DAY = cv.imread("camera_templates/camera_a_day.png")
CAMERA_B_TEMPLATE_DAY = cv.imread("camera_templates/camera_b_day.png")
CAMERA_C_TEMPLATE_DAY = cv.imread("camera_templates/camera_c_day.png")

CAMERA_A_TEMPLATE_NIGHT = cv.imread("camera_templates/camera_a_night.png")
CAMERA_B_TEMPLATE_NIGHT = cv.imread("camera_templates/camera_b_night.png")
CAMERA_C_TEMPLATE_NIGHT = cv.imread("camera_templates/camera_c_night.png")


class Frame:
    def __init__(self, frame):
        self.frame_ = frame 
    
    def raw(self):
        return self.frame_

class Video:
    def __init__(self, frames):
        self.frames_ = frames

    def get_frame(self, index):
        return self.frames_[index]
    
    def compute_camera(self):
        first_frame = self.frames_[0]
        self.camera_ = get_camera(first_frame.raw())
    
    def get_camera(self):
        return self.camera_


# Loads a video from the given path and returns a Video object.
# If num_frames is specified, it will only load that many frames.
def load_video(path, num_frames=None):
    frames = []

    if num_frames == 0:
        return Video(frames)

    cap = cv.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), cv.INTER_CUBIC)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        frames.append(Frame(frame))

        if num_frames is not None and len(frames) >= num_frames:
            break
    return Video(frames)

# Returns a cropped region of the frame that represents the content above the 
# horizon line.
def crop_horizon(frame):
    return frame[:int(0.55*FRAME_HEIGHT), :].copy()

# Returns the number of inliners when trying to
# allign the query to the template using sift + homography matrix.
# This is mainly used to see to which camera view a frame
# corresponds.
def similarity(query, template):
    sift = cv.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(query, None)
    kp2, des2 = sift.detectAndCompute(template, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) < 4:
        return 0

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    _, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    if mask is None:
        return 0
    return int(np.sum(mask))


# Returns "A" if the frame was taken by camera A, "B" if
# the frame was taken by camera B, etc.
def get_camera(frame):
    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    cropped_frame = crop_horizon(frame)
    to_match = [
        crop_horizon(CAMERA_A_TEMPLATE_DAY),
        crop_horizon(CAMERA_B_TEMPLATE_DAY),
        crop_horizon(CAMERA_C_TEMPLATE_DAY),
        crop_horizon(CAMERA_A_TEMPLATE_NIGHT),
        crop_horizon(CAMERA_B_TEMPLATE_NIGHT),
        crop_horizon(CAMERA_C_TEMPLATE_NIGHT)
    ]

    similarities = [similarity(cropped_frame, t) for t in to_match]
    similarities = np.array(similarities)

    return ["A", "B", "C", "A", "B", "C"][np.argmax(similarities)]