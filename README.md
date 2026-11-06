# 41014 Integrated Project: TurtleBot3 LiDAR navigation and ArUco docking (Track A)

Group number: **[fill in]**  Members: **[fill in]**

A TurtleBot3 Waffle Pi in MuJoCo drives to a charging dock past tall blocks, using only the
**EKF estimate** and the **LiDAR**; the ArUco camera corrects the EKF and finishes the alignment.
Ground truth from the simulator is used **only** for logging and evaluation (`src/simulation.py`).

## Install and run
```bash
pip install -r requirements.txt
python main.py                 # viewer demo; pick the scenario at the top of main.py
python results_plot.py         # plots of the last run -> results/*.png
python evaluate.py 20          # headless batch: random blocks / noise seeds -> results/evaluation.csv
python state_space_model.py    # model, controllability / observability ranks, LQR gains
python calibrate_odometry.py   # re-derives the odometry calibration in src/config.py
```
Headless machines: `MUJOCO_GL=egl python evaluate.py`.

**Scenario selection** is in `main.py` (comment/uncomment one `BLOCKS = ...` line): no obstacle,
one block in the path, two blocks or three blocks. In the viewer the green dot and disc are the EKF
position estimate and its 2-sigma uncertainty; the TurtleBot itself is the true state.

## Pipeline
```
wheel encoders -> Odometry -> EKF.predict ---------------+
camera -> ArUco -> range, bearing, marker angle -> EKF.update / update_heading
                                                         |  estimate x_hat = (x, y, theta)
LiDAR scan ---------------------------------------------+--> Navigator --> wheel speeds
                                                              (gap-following + LQR)
```
Frames: world (x east, y north) -> robot (x forward, y left, origin at wheel axle) -> camera
(x right, y down, z forward, 9 cm ahead of and 10.7 cm above the axle). The LiDAR is 1 cm behind the
axle. The camera pose converts to the robot frame in `range_bearing.py`; the LiDAR points convert in
`lidar_sensing.get_points`.

## Model (A)
Unicycle: `x' = v cos(theta), y' = v sin(theta), theta' = omega`, state (x, y, theta), inputs (v, omega).
`state_space_model.py` linearises it and checks:
* **Controllability:** rank 3 at v = 0.12 m/s, rank 2 at v = 0 (no sideways motion at standstill).
* **Observability:** range + bearing to one marker is rank 2 at an instant and rank 3 only while the
  robot moves with the marker in view. Odometry alone observes no absolute quantity, so error grows
  between marker sightings.

## Sensing (B)
* **Odometry calibration** (`calibrate_odometry.py`): simulated tyres slip, so the nominal geometry
  (r = 0.033 m, b = 0.288 m) over-reports motion. Effective values **r = 0.0281 m, b = 0.342 m** are
  used by the odometry and wheel commands. Residual noise injected by the simulator: 2% per-step
  noise and 1% wheel-scale error.
* **Camera:** ideal pinhole calibration from the field of view (`camera_matrix`), no distortion.
  `solvePnP` (IPPE square) gives range, bearing and the marker's orientation. Non-finite poses are discarded.
* **LiDAR:** 360 beams, 1 deg apart, beam 0 forward; misses become `inf` and are dropped.

## Estimation (C)
EKF on (x, y, theta). Predict from odometry increments; update from camera range/bearing
(`R = diag(0.05 m, 1 deg)^2`), plus a heading update from the marker orientation used only inside
1.5 m, where its error is small (sigma = 2 deg * range^2, clipped to 0.7..12 deg, fitted from simulation data).
Updates use the Joseph covariance form and a chi-square gate against outlier detections (after 5
consecutive rejections the gate is bypassed, because then the filter is the one that is wrong).
`Q = diag(0.5 mm, 0.5 mm, 1 mrad)^2` per 2 ms step.

## Control (D) and LiDAR navigation
* **LQR tracker** (`src/control/controller.py`): path-error model `e_y' = v e_theta, e_theta' = omega`,
  `Q = diag(10, 1)`, `R = 0.5`, v = 0.12 m/s gives `K = [4.47, 1.75]` and closed-loop poles -0.40 and -1.36.
* **Navigator** (`src/control/navigator.py`), every 20 ms:
  1. follow-the-gap: choose the free LiDAR direction (corridor of half-width 0.23 m, 1 m long) nearest
     the waypoint bearing, with hysteresis against dithering;
  2. that direction is the goal direction -> LQR line tracking (TRACK); otherwise LQR heading control
     toward the gap (AVOID); nothing free -> spin toward the most open side (TURN);
  3. speed falls with distance to goal, heading error and obstacle clearance; hard stop inside 0.25 m;
  4. at the dock waypoint rotate in place to heading 0 using the camera-corrected estimate (ALIGN).
  Obstacles beyond the goal are ignored, so the dock itself is not treated as an obstacle, and
  an intermediate waypoint covered by an obstacle is skipped.

## Results (E)
20 trials from `evaluate.py` (seeds 0-19; 1 or 2 random blocks in the path; every other pair of trials
starts with the EKF 0.28 m and 10 deg away from the true pose). Ground truth is used only for scoring.

| Metric | Result |
|---|---|
| Success (docked, < 10 cm, < 10 deg, no collision) | **20 / 20** (1 block 10/10, 2 blocks 10/10, normal start 10/10, disturbed start 10/10) |
| Final position error vs dock pose | mean 2.0 cm, max 2.8 cm |
| Final heading error | mean 0.65 deg, max 1.4 deg |
| Time to dock | mean 40 s (1 block 39 s, 2 blocks 41 s), max 54 s |
| Collisions | 0 |
| Position RMSE over the run | EKF 3.5 cm vs odometry-only 11.4 cm (normal start); EKF 12.5 cm in disturbed starts, which includes converging from the 28 cm initial error |
| Mean NEES (ideal about 3 for 3 states) | 3.2 normal start, 7.8 disturbed start (transient at the start) |
| Marker visible | 65% of the time; 5 camera updates rejected per run on average |
| Camera error vs truth | range std 2.4 cm, bearing std 0.03 deg (so R = 5 cm, 1 deg is conservative) |
| Marker-orientation heading error within 1.5 m | std 4.1 deg |

Plots from `results_plot.py`: `results/estimate_vs_truth.png` (EKF vs truth with 2-sigma and marker-visibility
shading), `results/path.png` (path around the block), `results/error.png` (error vs the filter's own bound).
Per-trial data is in `results/evaluation.csv`.

## Limitations
* The 2D LiDAR only sees its scan plane, and the reactive planner can be trapped by a dead end or a
  wall of obstacles; blocks are kept clear of the waypoints in the evaluation.
* With one marker, heading and position are only weakly observable while the marker is out of view.
* The camera calibration is ideal (no distortion) and camera noise is not injected.

## Contributions and generative AI declaration
**[Each member: list contribution]**

**[State tools and purpose, e.g. Claude used for code review, debugging and refactoring.]**
