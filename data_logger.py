import os
import argparse
import logging
import pyrealsense2 as rs
import cv2
import numpy as np
import math
import time
import csv

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import classifier

image = []
prev_image = []

prev_rw = None
prev_lw = None

def get_keypoints(h):
    r_elbow = h.body_parts[3]
    r_wrist = h.body_parts[4]
    l_elbow = h.body_parts[6]
    l_wrist = h.body_parts[7]
    return r_elbow, r_wrist, l_elbow, l_wrist

def distance_keypoint(point1, point2):
    x1 = point1.x
    y1 = point1.y
    x2 = point2.x
    y2 = point2.y
    # calculate distance
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance

def convert_keypoint(keypoint, image):
    x = keypoint.x*image.shape[1]
    y = keypoint.y*image.shape[0]
    return x, y

def optical_flow(frame, prev_frame, keypoint):
    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 5, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    x, y = convert_keypoint(keypoint, frame)
    magnitude = mag[int(y), int(x)]
    angle = ang[int(y), int(x)]
    return magnitude, angle


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Read recorded bag file to measure surgical scrub.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='320x240',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')

    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    # Check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=False)
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(640, 480), trt_bool=False)

    try:

        # Create pipeline
        pipeline = rs.pipeline()
        # Create a config object
        config = rs.config()
        # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, args.input)
        # Configure the pipeline to stream the depth stream
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
        # Start streaming from file
        pipeline.start(config)

        # Create opencv window to render image in
        cv2.namedWindow('Measuring Surgical Scrub', cv2.WINDOW_AUTOSIZE)

        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            image_frame = frames.get_color_frame()
            # Convert image to numpy arrays
            prev_image = image
            image = np.asanyarray(image_frame.get_data())
            dim = (480, 360)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            depth_image = np.asanyarray(depth_frame.get_data())
            # Pose Estimation from Inference Model
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

            try:
                for human in humans:

                    print("Keypoints: Getting")
                    re, rw, le, lw = get_keypoints(human)
                    print("Keypoints: Got")

                    # calculate distances between points
                    print("Distance: Getting")
                    dist_rwlw = distance_keypoint(rw, lw)
                    dist_rwle = distance_keypoint(rw, le)
                    dist_lwre = distance_keypoint(lw, re)
                    print("Distance: Got")

                    # calculate optical flow at points
                    print("Optical: Getting")
                    lw_mag, lw_ang = optical_flow(image, prev_image, lw)
                    rw_mag, rw_ang = optical_flow(image, prev_image, rw)
                    le_mag, le_ang = optical_flow(image, prev_image, le)
                    re_mag, re_ang = optical_flow(image, prev_image, re)
                    print("Optical: Got")

                    # if (prev_lw != None):
                    #     rd_score = classifier.dipRight(rw, lw, re, le, prev_rw, prev_lw)
                    #     rw_score = classifier.washRight(rw, lw, re, le, prev_rw, prev_lw)
                    #     ld_score = classifier.dipLeft(rw, lw, re, le, prev_rw, prev_lw)
                    #     lw_score = classifier.washLeft(rw, lw, re, le, prev_rw, prev_lw)
                    prev_rw = rw
                    prev_lw = lw
                    print("Keypoints: Writing")
                    try:
                        with open('dipRight.csv', 'a', newline='') as csvfile:
                            fieldnames = ['lw_x', 'lw_y', 'rw_x', 'rw_y', 'le_x', 'le_y', 're_x', 're_y']
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow({'lw_x': lw.x, 'lw_y': lw.y, 'rw_x': rw.x, 'rw_y': rw.y, 'le_x': le.x, 'le_y': le.y, 're_x': re.x, 're_y': re.y})
                    except e:
                        print("Error; ", e)
                    print("Distance: Writing")
                    try:
                        with open('dipRightDistance.csv', 'a', newline='') as csvfile:
                            fieldnames = ['rw2lw', 'rw2le', 'lw2re']
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow({'rw2lw': dist_rwlw, 'rw2le': dist_rwle, 'lw2re': dist_lwre})
                    except e:
                        print("Error; ", e)
                    print("Optical: Writing")
                    try:
                        with open('dipRightOptical.csv', 'a', newline='') as csvfile:
                            fieldnames = ['lw_mag', 'lw_ang', 'rw_mag', 'rw_ang', 'le_mag', 'le_ang', 're_mag', 're_ang']
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow({'lw_mag': lw_mag, 'lw_ang': lw_ang, 'rw_mag': rw_mag, 'rw_ang': rw_ang, 'le_mag': le_mag, 'le_ang': le_ang, 're_mag': re_mag, 're_ang': re_ang})
                    except e:
                        print("Error; ", e)

            except:
                pass

            # WINDOW DISPLAYING FRAMES
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            cv2.imshow('Measuring Surgical Scrub', image)
            key = cv2.waitKey(1)

            # if pressed escape exit program
            if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pass
        # pipeline.stop()
