# A peek into Robot Planning and Control
This repository serves as an introductory guide to planning and control algorithms commonly used in robotics. We begin with state estimation — not a planning or control technique per se, but a fundamental prerequisite for building autonomous systems. Next, we explore grid-based planners such as A* and Uniform Cost Search (UCS), followed by the implementation of a PID controller to enable the robot to follow planned waypoints. We then transition into sampling-based planning algorithms, and gradually delve into optimal control methods ranging from Quadratic Programming (QP) to Model Predictive Control (MPC) and MPC with Control Barrier Functions (MPC-CBF), which are widely used in autonomous robotics. This tutorial is designed to be accessible for beginners and also introduces powerful optimization libraries like cvxpy and CasADi.

## Chapter 1: State estimation using Gaussian Estimators

State estimation helps robots infer their true state (like position or velocity) from noisy sensor data. Kalman Filter (KF) is optimal for linear systems with Gaussian noise, combining prediction and correction. EKF and UKF extend KF to nonlinear systems—EKF uses linearization, while UKF uses sigma points for better accuracy.

Algorithms implemented:
- [x] Kalman Filter
- [x] Extended Kalman Filter
- [ ] Unscented Kalman Filter

![alt text](https://github.com/RahulHKumar/Robot-Planning-and-Control/blob/main/results/state_estimation.png)

## Chapter 2: Grid based planning and PID controller

Grid-based path planning algorithms can be used to search over a discretized map to find a path from start to goal. Algorithms like **A\***, **UCS**, **Greedy Search**, and **BFS** differ only in their **priority function** used for node expansion. This function combines factors like **path cost (g)**, **heuristic (h)**, or **depth**, e.g., A\* uses \( g + h \), UCS uses \( g \), Greedy uses \( h \), and BFS uses node depth.

Algorithms implemented:
- [x] A*
- [x] UCS
- [x] Greedy Search
- [x] BFS
- [x] DFS

![alt text](https://github.com/RahulHKumar/Robot-Planning-and-Control/blob/main/results/astar_pid_res.png)

## Chapter 3: Sampling based methods

Sampling-based planning algorithms like **PRM**, **RRT**, and their optimal variants (**PRM\***, **RRT\***), build paths by randomly sampling the space rather than searching over a fixed grid. They efficiently handle high-dimensional or continuous spaces where grid-based methods become computationally expensive or infeasible. PRM builds a global roadmap and RRT grows a tree from the start.

While **RRT\*** finds optimal paths in configuration space but often ignores system dynamics, making it unsuitable for real robots with motion constraints. **Kinodynamic RRT\*** addresses this by incorporating **dynamics and differential constraints**, generating dynamically feasible trajectories. **LQR-RRT\*** further improves performance by using **Linear Quadratic Regulator (LQR)** for local steering, leading to smoother, lower-cost paths and better control-aware exploration. We consider a unicycle model for kinodynamic RRT* and double integrator for LQR-RRT* in this tutorial. This section also introduces the use of **cvxpy**.

![alt text](https://github.com/RahulHKumar/Robot-Planning-and-Control/blob/main/results/sbmp.png)

This section implements:
- [x] PRM
- [x] PRM*
- [x] RRT
- [x] RRT*
- [x] Kinodynamic RRT*
- [x] LQR-RRT*

## Chapter 4: Optimization

Now, the **most important part of this tutorial!**. What is optimization? Check out the [notebook](Casadi_MPC_tutorial.ipynb) for step by step explanation of what is MPC? This section also introduces the use of **Casadi**.

**Optimization-based techniques** like **Model Predictive Control (MPC)** compute control actions by solving a constrained optimization problem at each timestep. They account for system dynamics, future goals, and input/state constraints, enabling **predictive and constraint-aware** decision making. We also simulate a 2D lidar in matplotlib using Bresenham's algorithm, K-Means Clustering and DBSCAN clustering.

![alt text](https://github.com/RahulHKumar/Robot-Planning-and-Control/blob/main/results/mpc_dc_simulation.gif)

**MPC-CBF** further enhances this by incorporating **Control Barrier Functions (CBFs)** for formal **safety guarantees**, ensuring collision avoidance and constraint satisfaction even in dynamic environments.

![alt text](https://github.com/RahulHKumar/Robot-Planning-and-Control/blob/main/results/mpccbfres.jpg)
