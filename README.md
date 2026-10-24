
# PenguinPi Autonomous Navigation & Fruit Search

**ECE4078 – Intelligent Robotics | Monash University**

## Project Overview

This project focuses on developing an autonomous navigation system for a PenguinPi mobile robot using Python.

The system integrates computer vision, robot localisation, path planning, and motion control to enable the robot to navigate through an environment and search for designated fruit targets while avoiding obstacles.

The project explores the integration of multiple robotics algorithms into a single autonomous system.

## Technologies Used

- **Programming Language:** Python
- **Robot Platform:** PenguinPi (Raspberry Pi-based mobile robot)
- **Path Planning:** Rapidly-exploring Random Tree (RRT)
- **Localisation:** Extended Kalman Filter (EKF)
- **Landmark Detection:** ArUco Markers
- **Object Detection:** YOLO
- **Computer Vision:** OpenCV
- **Interface:** Pygame
- **Numerical Computation:** NumPy

## Key Features

### 1. Autonomous Path Planning
- Uses RRT-based path planning to generate navigation routes.
- Performs waypoint-based navigation towards target locations.
- Integrates robot movement commands with the planned trajectory.

### 2. Robot Localisation and SLAM
- Uses an Extended Kalman Filter to estimate robot position and orientation.
- Incorporates ArUco landmark observations to improve localisation.
- Uses differential-drive motion models for position estimation.

### 3. Computer Vision
- Uses YOLO-based object detection to identify fruit targets.
- Processes camera images using OpenCV.
- Estimates fruit distance using detected bounding boxes and camera parameters.

### 4. Autonomous Navigation
- Controls robot rotation and forward movement.
- Updates localisation estimates during navigation.
- Includes relocalisation logic based on landmark visibility and pose uncertainty.

### 5. Robot Interface
- Uses Pygame to display robot information.
- Visualises camera images and detected objects.
- Provides feedback on localisation and navigation status.

## Repository Structure

```text
penguinpi-autonomous-navigation/
|
|-- # M4 - Autonomous fruit searching with U.py
|
|-- LiveDemo/
|   |-- operate.py
|   |-- util/
|   |-- pics/
|
|-- SLAM_unit_test/
|   |-- slam/
|       |-- ekf.py
|       |-- robot.py
|       |-- mapping_utils.py
|
|-- Screenshots/
|
|-- README.md
|-- .gitignore
```

## Project Demonstration

### PenguinPi Robot
Add a photograph of the physical robot and experimental environment.

### Autonomous Navigation
Add a screenshot showing the planned navigation route.

### SLAM and Localisation
Add a screenshot showing the robot's estimated position and detected landmarks.

### Fruit Detection
Add a screenshot demonstrating YOLO fruit detection.

## Running the Project

The project was developed for the ECE4078 PenguinPi robotics environment.

A complete setup requires:

- Python and the necessary Python libraries
- Compatible PenguinPi hardware and communication interface
- Camera and wheel calibration parameters
- RRT planning and YOLO detection modules
- Map and fruit search-list files

**Note:** This repository currently contains a project code snapshot. Some supporting modules, calibration files, and model files are not included, so additional setup is required before the complete autonomous system can run.

## Project Skills Demonstrated

- Autonomous mobile robotics
- Robotic path planning
- Robot localisation and SLAM
- Computer vision and object detection
- Sensor-data integration
- Python programming
- Robot motion control
- Algorithm integration and testing

## Acknowledgements

Developed as part of ECE4078 – Intelligent Robotics at Monash University.

The project uses university-provided PenguinPi utilities and robotics framework components alongside project implementation code.

