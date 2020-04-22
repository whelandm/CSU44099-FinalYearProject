# Final Year Project - Measureing Surgical Scrub

This project examines the application of upper body tracking for uses relating to the measuring of the Surgical Scrub Technique, as defined by the World Health Organization. 
This washing technique occurs between surgeries but is prone to human error. 
The aim of this project is to develop a Computer Vision based solution to identify and prevent the common errors that occur during this short but frequent process. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

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

### classifier.py

Contains all the classification tests for each of the actions within the action set. 
