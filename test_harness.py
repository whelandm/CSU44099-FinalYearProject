import os
import argparse
import logging
import pyrealsense2 as rs
import cv2
import numpy as np
import math
import time
import csv
import sys

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import classifier
import state_machine
import wash_tracker

DIP_R = 'dip right'
DIP_L = 'dip left'
WASH_R = 'wash right'
WASH_L = 'wash left'
UNKNOWN = 'unknown'

image = []
prev_image = []

rw = None
lw = None
prev_rw = None
prev_lw = None

frame_count = 0
frame_detect_count = 0

classification = ''
prev_classification = ''
transition = 0

hand_alert = ''
transition_alert = ''

arm_dist_l = 0
wrist_l = None
arm_dist_r = 0
wrist_r = None

error_log = []

timer = 0
curr_time = 0
prev_time = 0

time_dr = 0
time_wr = 0
time_dl = 0
time_wl = 0

timingTest_1 = False
timingTest_2 = False
timingTest_3 = False
timingTest_4 = False

coverage_l = False
coverage_r = False

misclass = 0
state = ''
prev_state = ''
state_window = []
state_time = 0

clipping_distance = 1

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def get_keypoints(h):
    r_elbow = h.body_parts[3]
    r_wrist = h.body_parts[4]
    l_elbow = h.body_parts[6]
    l_wrist = h.body_parts[7]
    return r_elbow, r_wrist, l_elbow, l_wrist

def hands_raised(r_elbow, r_wrist, l_elbow, l_wrist):
    # calculate distance
    r_distance = r_elbow.y - r_wrist.y
    #print(r_distance)
    l_distance = l_elbow.y - l_wrist.y
    #print(l_distance)
    if (r_distance < 0 and l_distance < 0):
        print("Raise Both Hands Above Elbows")
        return False
    if r_distance < 0:
        print("Raise Right Hand Above Elbow")
        return False
    if l_distance < 0:
        print("Raise Left Hand Above Elbow")
        return False
    return True

def convert_keypoint(keypoint, image):
    x = keypoint.x*image.shape[1]
    y = keypoint.y*image.shape[0]
    return x, y

def timingTest(dip_right, dip_left, wash_right, wash_left):
    score_dr = (dip_right/5)
    if (score_dr >= 0.8):
        timing_dr = True
    else:
        timing_dr = False
    score_dl = (dip_left/5)
    if (score_dl >= 0.8):
        timing_dl = True
    else:
        timing_dl = False
    score_wr = (wash_right/10)
    if (score_wr >= 0.8):
        timing_wr = True
    else:
        timing_wr = False
    score_wl = (wash_left/10)
    if (score_wl >= 0.8):
        timing_wl = True
    else:
        timing_wl = False
    return timing_dr, timing_dl, timing_wr, timing_wl

def getDepthAtPixel(depth_frame, pixel_x, pixel_y):
	return depth_frame.as_depth_frame().get_distance(round(pixel_x), round(pixel_y))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Read recorded bag file to measure surgical scrub.\
                                    Remember to change the stream resolution, fps and format to match the recorded.")
    # add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    parser.add_argument('--camera', type=str, default=0)
    parser.add_argument('--resize', type=str, default='320x240',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')

    # parse the command line arguments to an object
    args = parser.parse_args()
    # safety if no parameter have been given
    if not args.input:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    # check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=False)
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=False)

    try:

        # create pipeline
        pipeline = rs.pipeline()
        # create a config object
        config = rs.config()
        # tell config that we will use a recorded device from filem to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, args.input, False)
        # Configure the pipeline to stream the depth stream
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
        # start streaming from file
        profile = pipeline.start(config)
        # getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , depth_scale)
        # create align object
        align_to = rs.stream.color
        align = rs.align(align_to)
        # create opencv window to render image
        cv2.namedWindow('Measuring Surgical Scrub', cv2.WINDOW_AUTOSIZE)

        while True:

            # Wait for a pair of frames
            frames = pipeline.wait_for_frames()
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            image_frame = aligned_frames.get_color_frame()
            # Convert image to numpy arrays
            prev_image = image
            image = np.asanyarray(image_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            dim = (480, 360)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            # Pose Estimation from Inference Model
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            # Get time
            prev_time = curr_time
            curr_time = time.perf_counter()
            if (prev_time == 0):
                prev_time = curr_time
            timer += curr_time - prev_time

            try:
                for human in humans:

                    prev_rw = rw
                    prev_lw = lw

                    re, rw, le, lw = get_keypoints(human)

                    # first frame get arm length
                    if (arm_dist_l == 0):
                        arm_dist_l = classifier.findDistance(le, lw)
                        wrist_l = lw
                        arm_dist_r = classifier.findDistance(re, rw)
                        wrist_r = rw

                    # classify frame
                    rd_score, rw_score, ld_score, lw_score = classifier.classify(rw, lw, re, le, prev_rw, prev_lw, image, prev_image)
                    prev_classification = classification

                    # assign times and classification
                    if (rd_score > rw_score and rd_score > ld_score and rd_score > lw_score):
                        classification = DIP_R
                    if (rw_score > rd_score and rw_score > ld_score and rw_score > lw_score):
                        classification = WASH_R
                    if (ld_score > rd_score and ld_score > rw_score and ld_score > lw_score):
                        classification = DIP_L
                    if (lw_score > rd_score and lw_score > rw_score and lw_score > ld_score):
                        classification = WASH_L

                    state_window.append(classification)

                    if (len(state_window) >= 5):
                        # update state
                        prev_state = state
                        state_dr, state_dl, state_wr, state_wl = 0, 0, 0, 0
                        for s in state_window:
                            if s == DIP_R:
                                state_dr += 1
                            if s == DIP_L:
                                state_dl += 1
                            if s == WASH_R:
                                state_wr += 1
                            if s == WASH_L:
                                state_wl += 1
                        if (state_dr > state_dl and state_dr > state_wr and state_dr > state_wl):
                            state = DIP_R
                        elif (state_dl > state_dr and state_dl > state_wr and state_dl > state_wl):
                            state = DIP_L
                        elif (state_wr > state_dr and state_wr > state_dl and state_wr > state_wl):
                            state = WASH_R
                        elif (state_wl > state_dr and state_wl > state_dl and state_wl > state_wr):
                            state = WASH_L
                        state_time = 0
                        state_window.clear()

                    # check state machine for accepted transition
                    if state != prev_state:
                        if state_machine.acceptedState(state, prev_state):
                            transition_alert = ''
                        else:
                            transition_alert = 'Invalid Transition'
                            state = prev_state
                    else:
                        transition_alert = ''

                    # add time to state
                    if (state == DIP_R and transition_alert == ''):
                        time_dr += curr_time - prev_time
                    if (state == WASH_R):
                        time_wr += curr_time - prev_time
                    if (state == DIP_L):
                        time_dl += curr_time - prev_time
                    if (state == WASH_L):
                        time_wl += curr_time - prev_time

                    frame_detect_count +=1

                    # error check - hands above elbows
                    if hands_raised(re, rw, le, lw):
                        hand_alert = ''
                    else:
                        hand_alert = 'Raise hands above elbows'

                    # error check - wash coverage
                    if (classification == WASH_L):
                        if wash_tracker.getCoverageLeft(le, wrist_l, re, rw, arm_dist_l):
                            coverage_l = True
                    if (classification == WASH_R):
                        if wash_tracker.getCoverageRight(re, wrist_r, le, lw, arm_dist_r):
                            coverage_r = True

                    # error check - adequate times
                    timingTest_1, timingTest_2, timingTest_3, timingTest_4 = timingTest(time_dr, time_dl, time_wr, time_wl)

            except:
                pass

            # draw Human Model on Image
            display = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            # Write Classifications to image
            cv2.putText(display, str(frame_count), (0,12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0 , 255), 2, cv2.LINE_AA)
            cv2.putText(display, str(frame_detect_count),(0,24), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display, str(timer), (0,358), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(display, 'State: ' + state, (200,335), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, 'Frame Class: ' + classification, (200,355), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # error alerts
            cv2.putText(display, hand_alert, (200,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display, transition_alert, (200,190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # coverage alerts
            cv2.putText(display, 'Coverage', (300,84), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if coverage_r:
                cv2.putText(display, 'Wash Right: Passed', (300,96), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(display, 'Wash right arm', (300,96), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if coverage_l:
                cv2.putText(display, 'Wash Left: Passed', (300,108), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(display, 'Wash left arm', (300,108), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # timing alerts
            cv2.putText(display, 'Timing', (300,12), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if (timingTest_1):
                cv2.putText(display, 'Dip Right: Passed', (300,24), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(display, 'Dip right fingers', (300,24), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if (timingTest_3):
                cv2.putText(display, 'Wash Right: Passed', (300,36), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(display, 'Wash right arm', (300,36), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if (timingTest_2):
                cv2.putText(display, 'Dip Left: Passed', (300,48), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(display, 'Dip left fingers', (300,48), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if (timingTest_4):
                cv2.putText(display, 'Wash Left: Passed', (300,60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(display, 'Wash left arm', (300,60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Measuring Surgical Scrub', display)

            frame_count += 1 # increase frame count
            error_log.clear() # clear errors from this frame
            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        sys.exit()
