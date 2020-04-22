# Final Year Project - Measuring Surgical Scrub

This project examines the application of upper body tracking for uses relating to the measuring of the Surgical Scrub Technique, as defined by the World Health Organization. 
This washing technique occurs between surgeries but is prone to human error. 
The aim of this project is to develop a Computer Vision based solution to identify and prevent the common errors that occur during this short but frequent process. 

## Getting Started

### Prerequisites

This project requires the installation of [TF Pose Estimation](https://github.com/ildoonet/tf-pose-estimation) from [Ildoo Kim](https://github.com/ildoonet). 
A video documenting this process can be found [here](https://www.youtube.com/watch?v=4FZrE3cmTPA). 
It also makes use of the [Intel Real Sense SDK](https://github.com/IntelRealSense/librealsense), where the Viewer allows the .bag files to be viewed in isolation.

### Installing

1. Download this repository.
2. Extract its contents to the TF Pose Estimation directory.
3. (Optional) Install [Anaconda](https://docs.anaconda.com/anaconda/install/). 
4. Download the training and testing video files from here and here. 
5. Extract the training and testing folders to the TF Pose Estimation directory.
6. Run the test harness to see if all dependencies are installed.

```
python test_harness.py -i ./testing/fullRun1.bag
```

## File Descriptions

Below is a breakdown of the python scripts included and their purpose.

### test_harness.py
The test harness is the main compenent which takes in the video file (must be .bag format) and reads in the video stream.
From this the classification is given and test ran, and displayed to user.
```
python test_harness.py -i .BAG FILE PATH
```

### classifier.py
Contains all the classification tests for each of the actions within the action set.

### state_machine.py
Mapping of valid transitions between one action to the next

### wash_tracker.py
For coverage measuring. Included is unused method which makes use of depth stream.

### data_analyser.py
Reads and plots datasets from .csv files, mapping data relating to keypoint locations, keypoint distances, and keypoint optical flow. 
Note that only location and distance data is used.

### data_logger.py
Can be used to append additional frames of data to a designated file, must be in .bag format.
```
python data_logger.py -i .BAG FILE PATH
```

### data_visualiser.py
Can be used to visualise the current dataset, including their lines of best fit.
```
python data_visualiser.py
```

## Dataset Descriptions

Below is a breakdown of the datasets included, found in the .csv files. All data is taken from the training videos, which contain 10 unique videos for each of the individual actions (dip left, dip right, wash left, wash right).

### Keypoint Location
```
Files: dipLeft.csv, dipRight.csv, washLeft.csv, wasRight.csv
```
The data points represent the location of the arm keypoints in the current frame, these are specific to each of the actions within the protocol. 

### Keypoint Distances
```
Files: dipLeftDistance.csv, dipRightDistance.csv, washLeftDistance.csv, wasRightDistance.csv
```
Each row represents the distance between opposing keypoints for that frame; from wrist to wrist, left wrist to right elbow, and and right wrist to left elbow.

### Keypoint Optical Flow
```
Files: dipLeftOptical.csv, dipRightOptical.csv, washLeftOptical.csv, wasRightOptical.csv
```
Each data point contains the magnitude and angle of the keypoints for a given frame.
