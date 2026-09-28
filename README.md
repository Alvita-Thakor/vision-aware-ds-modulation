# Vision-Aware DS-Modulation Obstacle Avoidance

A research project investigating how realistic computer-vision perception errors affect Dynamical System (DS) modulation obstacle avoidance, and whether Kalman filtering can recover part of the resulting performance degradation.

## Research Question

How does DS-modulation-based obstacle avoidance degrade when obstacle position estimates come from noisy, realistic computer-vision perception — including pixel noise, blur, latency, low frame rate, and occlusion — instead of clean ground-truth data, and can a Kalman filter recover performance toward the clean baseline?

## Overview

The project connects computer vision, state estimation, and reactive robot control into a single perception-to-control pipeline:

Camera Image
     ↓
Obstacle Detection
     ↓
Pixel → Ground-Coordinate Conversion
     ↓
Perception Degradation
     ↓
Kalman Filtering
     ↓
DS-Modulation Controller
     ↓
Robot Motion
     ↓
Performance Metrics

The main experiments were performed in a 2D simulated environment, followed by an end-to-end ROS 2 and TurtleBot3/Gazebo demonstration.

## Project Components

### Computer Vision

- Synthetic camera-based obstacle perception
- OpenCV-based obstacle detection
- Centroid extraction
- Camera-to-ground coordinate conversion
- Controlled perception degradation

### State Estimation

A constant-velocity Kalman filter estimates the obstacle position from noisy measurements.

### Obstacle Avoidance

The controller uses Dynamical System modulation to modify goal-directed motion around obstacles.

### Experiments

The perception pipeline was evaluated under:

- Pixel noise
- Blur
- Low camera frame rate
- Perception latency
- Occlusion
- Combined pixel noise and occlusion

Performance was evaluated using:

- Localization RMSE
- Minimum clearance
- Convergence time
- Path length
- Collision rate
- Goal-reaching rate

## Key Findings

The experiments showed that:

- Increasing pixel noise increased localization error.
- Kalman filtering consistently reduced localization RMSE under pixel noise.
- Low frame rates could make filtering less effective, particularly at very low update rates.
- Increasing latency reduced obstacle clearance and degraded control behavior.
- Blur had comparatively limited influence over the tested range.
- Occlusion introduced systematic perception errors that were not reliably removed by temporal filtering.
- The benefit of Kalman filtering therefore depended on the type of perception error rather than being uniformly beneficial.

## ROS 2 Demonstration

The complete perception-to-control pipeline was integrated with:

- ROS 2 Jazzy
- TurtleBot3
- Gazebo Sim
- OpenCV
- Kalman filtering
- DS-modulation obstacle avoidance

The ROS 2 implementation provides an end-to-end demonstration of the perception, estimation, and control pipeline.

The ROS 2 package is located in:

ros2/vision_aware_demo/

## Repository Structure

vision-aware-ds-modulation/
├── src/
├── ros2/
│   └── vision_aware_demo/
├── figures/
├── report/
│   └── Vision_Aware_DS_Modulation_Final_Report.pdf
├── main.py
├── requirements.txt
├── README.md
└── LICENSE

## Results

Selected final experiment data and figures are included in the repository.

The complete methodology, experimental results, statistical analysis, discussion, limitations, and conclusions are available in the final report:

report/Vision_Aware_DS_Modulation_Final_Report.pdf

## Limitations

The experiments use a synthetic camera and simulated robot environment with controlled perception degradations. The occlusion experiments use a fixed geometry, and the Kalman filter uses a constant-velocity model that was not evaluated with a moving obstacle.

The ROS 2/Gazebo implementation serves as an end-to-end demonstration rather than a replacement for the controlled experimental evaluation.

## References

The project builds on work in Dynamical System modulation and reactive obstacle avoidance, state estimation, computer vision, and ROS 2. Full references are provided in the final report.

## License

See `LICENSE`.