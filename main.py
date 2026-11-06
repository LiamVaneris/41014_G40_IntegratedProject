"""Run the TurtleBot3 docking demo in the MuJoCo viewer.

    python main.py

Pipeline each step:   wheel odometry -> EKF predict ; camera (ArUco) -> EKF update
                      EKF estimate + LiDAR scan -> Navigator (LQR + gap-following) -> wheels
Ground truth is used ONLY for logging / evaluation (see src/simulation.py), never by the controller.
"""
import time

import mujoco
import mujoco.viewer
import numpy as np

from src.config import DOCK_POSE
from src.simulation import Simulation

# =============================== SCENARIO ======================================
# Uncomment ONE line. Obstacles are tall blocks on the way to the dock.

BLOCKS = []                                # no obstacle: straight run to the dock
BLOCKS = [(1.5, -1.0)]                   # one block directly in the robot's path
BLOCKS = [(1.3, -0.8), (1.7, -1.2)]      # two blocks, robot must weave between/around them
BLOCKS = [(1.2, -1.3), (1.6, -0.7), (1.9, -1.3)]   # three blocks (keep x in 1.1..1.8 so no waypoint is covered)

DISTURBED_START = False                    # True: EKF starts at the wrong pose and must correct itself
SEED = 0                                   # odometry noise seed
REAL_TIME = True                           # slow the sim to real time so it is watchable
DRAW_ESTIMATE = True                       # draw the EKF estimate (green dot) and 2-sigma circle in the viewer
# ================================================================================


def draw_estimate(viewer, ekf):
    """Green dot = EKF position estimate, translucent disc = 2-sigma position uncertainty.
    The real robot is the TurtleBot itself, so true vs estimated is visible side by side."""
    scn = viewer.user_scn
    scn.ngeom = 0
    sigma = float(np.sqrt(max(ekf.P[0, 0], ekf.P[1, 1])))
    eye = np.eye(3).flatten()
    mujoco.mjv_initGeom(scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, [0.03, 0, 0],
                        [ekf.x[0], ekf.x[1], 0.35], eye, [0.1, 1.0, 0.2, 1.0])
    mujoco.mjv_initGeom(scn.geoms[1], mujoco.mjtGeom.mjGEOM_CYLINDER, [2 * sigma, 0.002, 0],
                        [ekf.x[0], ekf.x[1], 0.01], eye, [0.1, 1.0, 0.2, 0.25])
    scn.ngeom = 2


def main():
    sim = Simulation(blocks=BLOCKS, seed=SEED, disturbed_start=DISTURBED_START)
    model, data = sim.model, sim.data

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t_wall = time.time()
        while viewer.is_running() and not sim.finished:
            sim.step()
            if sim.step_count % 10 == 0:                    # ~50 Hz display
                if DRAW_ESTIMATE:
                    draw_estimate(viewer, sim.ekf)
                viewer.sync()
                if REAL_TIME:
                    lag = data.time - (time.time() - t_wall)
                    if lag > 0:
                        time.sleep(lag)
            if sim.step_count % 1000 == 0:                  # every 2 s of sim time
                print(f"t={data.time:5.1f}s  mode={sim.nav.mode:5s}  waypoint {sim.nav.i + 1}/3  "
                      f"marker {'seen' if sim.marker_seen else 'not seen'}  "
                      f"EKF sigma_xy={np.sqrt(sim.ekf.P[0, 0]):.3f} m")

    r = sim.result()          # ground truth used here for evaluation only
    print("\n===== RESULT (evaluation vs ground truth) =====")
    print(f"docked:            {r['reached_dock']}   (success: {r['success']})")
    print(f"time:              {r['time_s']:.1f} s")
    print(f"final position err {r['final_pos_err_m'] * 100:.1f} cm   heading err {r['final_head_err_deg']:.1f} deg "
          f"(dock pose x={DOCK_POSE[0]}, y={DOCK_POSE[1]})")
    print(f"collided:          {r['collided']}")
    print(f"EKF RMSE:          {r['ekf_rmse_m'] * 100:.1f} cm   (odometry alone: {r['odom_rmse_m'] * 100:.1f} cm)")
    print(f"mean NEES:         {r['mean_nees']:.2f}   (about 3 = consistent for a 3-state filter)")
    print(f"marker visible:    {r['marker_seen_frac'] * 100:.0f}% of the time")

    np.save("results/run.npy", np.array(sim.log))
    np.save("results/blocks.npy", np.array(BLOCKS).reshape(-1, 2))
    print("saved results/run.npy  ->  python results_plot.py")
    sim.close()


if __name__ == "__main__":
    main()
