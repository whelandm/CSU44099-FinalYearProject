# wash tracker
import math
import numpy as np
import cv2

def convertKeypoint(keypoint, image):
    x = keypoint.x*image.shape[1]
    y = keypoint.y*image.shape[0]
    return x, y

def getDepthAtPixel(depth_frame, pixel_x, pixel_y):
	return depth_frame.as_depth_frame().get_distance(round(pixel_x), round(pixel_y))

def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

def getLineIntersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(x_diff, y_diff)
    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return x, y

def getDistance(x1, y1, x2, y2):
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def getCoverageRight(static_elbow, static_wrist, motion_elbow, motion_wrist, arm_distance):
    # get distance to elbow
    rise = motion_elbow.y - motion_wrist.y
    run = motion_elbow.x - motion_wrist.x
    wrist_line = (motion_wrist.x - run, motion_wrist.y - rise), (motion_elbow.x, motion_elbow.y)
    arm_line = ((static_elbow.x, static_elbow.y), (static_wrist.x, static_wrist.y))
    intersection_x, intersection_y = getLineIntersection(wrist_line, arm_line)
    intersection_distance = getDistance(static_elbow.x, static_elbow.y, intersection_x, intersection_y)
    coverage = (arm_distance - intersection_distance) / arm_distance
    if (coverage >= 0.7):
        return True
    return False

def getCoverageLeft(static_elbow, static_wrist, motion_elbow, motion_wrist, arm_distance):
    # get distance to elbow
    rise = motion_elbow.y - motion_wrist.y
    run = motion_elbow.x - motion_wrist.x
    wrist_line = (motion_wrist.x - run, motion_wrist.y - rise), (motion_elbow.x, motion_elbow.y)
    arm_line = ((static_elbow.x, static_elbow.y), (static_wrist.x, static_wrist.y))
    intersection_x, intersection_y = getLineIntersection(wrist_line, arm_line)
    intersection_distance = getDistance(static_elbow.x, static_elbow.y, intersection_x, intersection_y)
    coverage = (arm_distance - intersection_distance) / arm_distance
    if (coverage >= 0.7):
        return True
    return False

# Coverage Calculation using Depth: NOT USED
def findDepthThreshold(motion_wrist, static_wrist, static_elbow, depth_frame, image):
    x, y = convertKeypoint(motion_wrist, image)
    motion_depth = getDepthAtPixel(depth_frame, x, y)
    x, y = convertKeypoint(static_wrist, image)
    static_w = getDepthAtPixel(depth_frame, x, y)
    x, y = convertKeypoint(static_elbow, image)
    static_e = getDepthAtPixel(depth_frame, x, y)
    if (static_w > static_e):
        static_depth = static_w
    else:
        static_depth = static_e
    return motion_depth, static_depth

def getClipImages(motion_wrist, static_wrist, static_elbow, depth_frame, depth_scale, image, depth_image):
    thresh_mot, thresh_sta = findDepthThreshold(motion_wrist, static_wrist, static_elbow, depth_frame, image)
    clipping_distance_mot = thresh_mot / depth_scale
    print(clipping_distance_mot)
    clipping_distance_sta = thresh_sta / depth_scale
    print(clipping_distance_sta)
    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 0
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
    bg_removed_mot = np.where((depth_image_3d > clipping_distance_mot) | (depth_image_3d <= 0), 0, image)
    bg_removed_sta = np.where((depth_image_3d > clipping_distance_sta) | (depth_image_3d <= 0), 0, image)
    # bg_removed_mot = np.where((depth_image_3d < clipping_distance_mot) | (depth_image_3d <= 0), 255, bg_removed_mot)
    # bg_removed_sta = np.where((depth_image_3d < clipping_distance_sta) | (depth_image_3d <= 0), 255, bg_removed_sta)
    images = np.hstack((bg_removed_mot, bg_removed_sta))
    cv2.imshow('Clipped Images', images)
    return bg_removed_mot, bg_removed_sta

def calculateCoverage(motion_wrist, static_wrist, static_elbow, depth_frame, depth_scale, image, depth_image):
    bg_mot, bg_sta = getClipImages(motion_wrist, static_wrist, static_elbow, depth_frame, depth_scale, image, depth_image)
    diff = bg_mot - bg_sta
    # cv2.imshow("Diff", diff)
    # cv2.imshow("Thresh", thresh)
