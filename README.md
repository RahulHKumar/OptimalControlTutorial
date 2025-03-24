# A peak into Robot Planning and Control
This notebook serves as an introductory guide to planning and control algorithms commonly used in robotics. We begin with state estimation—not a planning or control technique per se, but a fundamental prerequisite for building autonomous systems. Next, we explore grid-based planners such as A* and Uniform Cost Search (UCS), followed by the implementation of a PID controller to enable the robot to follow planned waypoints. We then transition into sampling-based planning algorithms, and gradually delve into optimal control methods ranging from Quadratic Programming (QP) to Model Predictive Control (MPC) and MPC with Control Barrier Functions (MPC-CBF), which are widely used in autonomous robotics. This tutorial is designed to be accessible for beginners and also introduces powerful optimization libraries like cvxpy and CasADi.

## Chapter 1: State estiation using Gaussian Estimators

State estimation helps robots infer their true state (like position or velocity) from noisy sensor data. Kalman Filter (KF) is optimal for linear systems with Gaussian noise, combining prediction and correction. EKF and UKF extend KF to nonlinear systems—EKF uses linearization, while UKF uses sigma points for better accuracy.

Algorithms implemented:
- [x] Kalman Filter
- [x] Extended Kalman Filter
- [ ] Unscented Kalman Filter

![alt text](https://github.com/RahulHKumar/Robot-Planning-and-Control/blob/main/results/state_estimation.png)

## Chapter 2: Grid based planning and PID controller

Grid-based path planning algorithms can be used to search over a discretized map to find a path from start to goal. Algorithms like **A\***, **UCS**, **Greedy Search**, and **BFS** differ only in their **priority function** used for node expansion. This function combines factors like **path cost (g)**, **heuristic (h)**, or **depth**, e.g., A\* uses \( g + h \), UCS uses \( g \), Greedy uses \( h \), and BFS uses node depth.
