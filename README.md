# Autonomous Quadrotor Navigation in 3D Occupancy Maps

Integrated planning, trajectory generation, and state estimation for end-to-end autonomy.

## Overview
This project integrates discretization, search-based planning, trajectory generation, and state estimation into a single autonomous quadrotor navigation pipeline. The stack takes a 3D occupancy map, computes collision-free routes, generates feasible trajectories, and executes them using visual-inertial state estimation.

## Repo Structure
- `proj3/`: Core project code, including planning, trajectory generation, control, and estimation.
- `util/`: Supporting utilities and helpers.
- `test_*.json`: Example map configurations used for evaluation.
- `test.py`, `sandbox.py`: Local test drivers and experiments.

## Technical Highlights
- **Planning:** A* search on a voxelized 3D occupancy grid using an L2 heuristic, tuned discretization, and obstacle inflation.
- **Trajectory generation:** Minimum-jerk polynomial segments with continuity constraints and an end-of-trajectory hold for stable terminal behavior.
- **Control:** Geometric SE(3) controller that maps desired thrust and moments to rotor speeds under saturation.
- **State estimation:** Error-state Kalman filter with IMU propagation and stereo-vision updates.

## Results Summary
The final integrated stack maintained fast planning and execution across multiple maps, improving total flight time by 63.95 s over the earlier baseline. The estimator converged quickly, enabling stable high-speed motion through cluttered environments.

## Assets
- Thumbnail: `assets/img/projects/autonomous-quadrotor-navigation/3D_Path.png`
- Figures: `assets/img/projects/autonomous-quadrotor-navigation/cov.png`, `assets/img/projects/autonomous-quadrotor-navigation/astar.png`
- Report: `assets/docs/autonomous-quadrotor-navigation.pdf`

## Links
- Report (PDF): `assets/docs/autonomous-quadrotor-navigation.pdf`
- Repo (TBD): https://github.com/ethsen/REPO-TBD
