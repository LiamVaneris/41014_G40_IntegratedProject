"""One simulated docking run: physics, sensors, EKF and navigator wired together.

Used by both main.py (with the viewer) and evaluate.py (headless batch trials), so the
demo and the reported numbers come from exactly the same code.

GROUND TRUTH RULE: true_pose() and the block/contact queries are used ONLY for logging
and evaluation. The navigator receives the EKF estimate and the LiDAR scan, nothing else.
"""
from pathlib import Path

import mujoco
import numpy as np

from src.config import (CAM_FOVY_DEG, CAM_NAME, CAM_RATE_HZ, CTRL_RATE_HZ, DOCK_POSE, IMG_H, IMG_W,
                        MARKER_NORMAL, MARKER_XY, SIM_TIME, WAYPOINTS)
from src.control.navigator import Navigator
from src.estimation.ekf import EKF, wrap
from src.estimation.odometry import Odometry, read_wheel_angles
from src.perception.lidar_sensing import get_scan
from src.perception.range_bearing import camera_matrix, measure_marker

ROOT = Path(__file__).resolve().parent.parent
PARK = [(8.0, 8.0), (8.0, 9.0), (8.0, 10.0)]     # where unused blocks wait (outside the room)
HEADING_MAX_RANGE = 1.5                           # m, only trust marker orientation inside this
SETTLE_TIME = 1.0                                 # s to keep running after the navigator reports done


def heading_sigma(r):
    """Std of the marker-orientation heading measurement vs range (fitted from simulation data)."""
    return float(np.clip(np.radians(2.0) * r ** 2, np.radians(0.7), np.radians(12.0)))


def true_pose(data):
    """GROUND TRUTH - evaluation only."""
    x, y = data.qpos[0], data.qpos[1]
    w, qx, qy, qz = data.qpos[3:7]
    yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return np.array([x, y, yaw])


class Simulation:
    def __init__(self, blocks=(), seed=0, disturbed_start=False, use_heading=True):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.blocks = list(blocks)
        for i in range(3):                                   # place or park each block
            x, y = self.blocks[i] if i < len(self.blocks) else PARK[i]
            mid = self.model.body(f"block{i + 1}").mocapid[0]
            self.data.mocap_pos[mid] = [x, y, 0.5]
        mujoco.mj_forward(self.model, self.data)

        self.renderer = mujoco.Renderer(self.model, IMG_H, IMG_W)
        self.K_cam = camera_matrix(CAM_FOVY_DEG, IMG_W, IMG_H)
        self.cam_opt = mujoco.MjvOption()
        self.cam_opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False   # no LiDAR rays in camera images
        # Floor reflections and shadows are not needed to find the marker (and the floor
        # reflection draws a mirrored ghost marker), so skip them: much faster rendering.
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

        self.odom = Odometry(x0=0.0, y0=0.0, theta0=0.0, seed=seed)
        self.odom.update(*read_wheel_angles(self.model, self.data))
        if disturbed_start:   # filter starts in the wrong place and must correct itself
            self.ekf = EKF([0.2, 0.2, np.radians(10)], np.diag([0.3 ** 2, 0.3 ** 2, np.radians(15) ** 2]))
        else:
            self.ekf = EKF([0.0, 0.0, 0.0], np.diag([0.01 ** 2, 0.01 ** 2, np.radians(1) ** 2]))
        self.nav = Navigator(WAYPOINTS, dock_heading=DOCK_POSE[2])
        self.use_heading = use_heading

        self.dt = self.model.opt.timestep
        self.ctrl_every = int(round(1.0 / (CTRL_RATE_HZ * self.dt)))
        self.cam_period = 1.0 / CAM_RATE_HZ
        self.next_cam = 0.0
        self.cmd = (0.0, 0.0)
        self.step_count = 0
        self.done_time = None
        self.marker_seen = False

        self.log = []            # per-step rows for results_plot.py
        self.nees = []           # evaluation only
        self.cam_resid = []      # evaluation only: camera error vs truth
        self.collision_steps = 0

        self._robot_bodies = {self.model.body(n).id for n in ("base", "wheel_left", "wheel_right")}
        self._floor = self.model.geom("floor").id

    # ------------------------------------------------------------------ stepping
    @property
    def finished(self):
        return self.data.time > SIM_TIME or (self.done_time is not None
                                             and self.data.time - self.done_time > SETTLE_TIME)

    def step(self):
        m, d = self.model, self.data

        # Control at 50 Hz from the ESTIMATE + LiDAR only
        if self.step_count % self.ctrl_every == 0:
            self.cmd = self.nav.command(self.ekf.x, get_scan(m, d))
            if self.nav.done and self.done_time is None:
                self.done_time = d.time
        d.ctrl[0], d.ctrl[1] = self.cmd
        mujoco.mj_step(m, d)
        self.step_count += 1

        # Odometry -> EKF predict
        pose_odom = self.odom.update(*read_wheel_angles(m, d))
        self.ekf.predict(self.odom.last_ds, self.odom.last_dtheta)

        # Camera -> EKF update
        if d.time >= self.next_cam:
            self._camera_update()
            self.next_cam += self.cam_period

        self._count_collisions()
        truth = true_pose(d)
        self.log.append([d.time, *self.ekf.x, *np.sqrt(np.diag(self.ekf.P)), *pose_odom, *truth,
                         self.nav.mode_code, float(self.marker_seen)])

    def _camera_update(self):
        d = self.data
        self.renderer.update_scene(d, camera=CAM_NAME, scene_option=self.cam_opt)
        meas = measure_marker(self.renderer.render(), self.K_cam)
        self.marker_seen = meas is not None
        if meas is not None:
            rng, bearing, alpha = meas
            self.ekf.update((rng, bearing))
            if self.use_heading and rng < HEADING_MAX_RANGE:
                self.ekf.update_heading(alpha, heading_sigma(rng))

            # --- evaluation only: camera error and filter consistency vs ground truth ---
            t = true_pose(d)
            dx, dy = MARKER_XY - t[:2]
            self.cam_resid.append([np.hypot(dx, dy), rng - np.hypot(dx, dy),
                                   wrap(bearing - wrap(np.arctan2(dy, dx) - t[2])),
                                   wrap(wrap(MARKER_NORMAL - alpha) - t[2])])
        e = self.ekf.x - true_pose(d)
        e[2] = wrap(e[2])
        self.nees.append(float(e @ np.linalg.solve(self.ekf.P, e)))

    def _count_collisions(self):
        """Evaluation only: count steps where the robot touches anything except the floor."""
        m, d = self.model, self.data
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            in1, in2 = b1 in self._robot_bodies, b2 in self._robot_bodies
            if in1 != in2 and self._floor not in (c.geom1, c.geom2):
                self.collision_steps += 1
                return

    # ------------------------------------------------------------------- results
    def result(self):
        """Metrics for one finished run (uses ground truth - evaluation only)."""
        log = np.array(self.log)
        t, est, odo, tru = log[:, 0], log[:, 1:4], log[:, 7:10], log[:, 10:13]
        final = tru[-1]
        dock = np.array(DOCK_POSE)
        pos_err = float(np.hypot(*(final[:2] - dock[:2])))
        head_err = float(abs(np.degrees(wrap(final[2] - dock[2]))))
        ekf_pos = np.hypot(*(est - tru)[:, :2].T)
        odo_pos = np.hypot(*(odo - tru)[:, :2].T)
        resid = np.array(self.cam_resid) if self.cam_resid else np.zeros((0, 4))
        return dict(
            success=bool(self.nav.done and pos_err < 0.10 and head_err < 10.0 and self.collision_steps == 0),
            reached_dock=bool(self.nav.done),
            time_s=float(self.done_time if self.done_time is not None else t[-1]),
            final_pos_err_m=pos_err,
            final_head_err_deg=head_err,
            collided=self.collision_steps > 0,
            ekf_rmse_m=float(np.sqrt(np.mean(ekf_pos ** 2))),
            odom_rmse_m=float(np.sqrt(np.mean(odo_pos ** 2))),
            ekf_final_err_m=float(ekf_pos[-1]),
            mean_nees=float(np.mean(self.nees)) if self.nees else float("nan"),
            marker_seen_frac=float(np.mean(log[:, 14])),
            rejected=self.ekf.n_rejected,
            resid=resid,
        )

    def close(self):
        self.renderer.close()
