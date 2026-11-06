"""Waypoint navigation with LiDAR gap-following, ending in a camera-aided dock.

Every control step the navigator
  1. picks the free LiDAR direction closest to the direction of the next waypoint
     (follow-the-gap, src/perception/lidar_sensing.best_heading);
  2. if that direction IS the goal direction -> TRACK: LQR path following along the
     line to the waypoint (cross-track + heading error);
     otherwise                               -> AVOID: LQR heading control toward the gap;
     if nothing is free                      -> TURN: spin in place toward the most open side;
  3. scales speed by distance-to-goal, heading error and obstacle clearance;
  4. at the dock waypoint, rotates in place until the estimated heading matches
     the dock heading (ALIGN), then stops (DONE).

Only the EKF estimate x_hat and the LiDAR scan are used - never ground truth.
"""
import numpy as np

from src.control.controller import LQRTracker, wheel_speeds, wrap
from src.perception.lidar_sensing import (best_heading, corridor_clearance,
                                          get_points, obstacle_in_front)

# --- tuning ---------------------------------------------------------------------
MAX_V = 0.16            # m/s cruise speed (wheel limit ~0.21 m/s with the calibrated radius)
K_V = 0.6               # 1/s, speed ~ K_V * distance-to-goal (smooth slow-down)
MIN_V = 0.04            # m/s, keeps the robot creeping into the last centimetres
LOOK = 1.0              # m, LiDAR look-ahead for choosing a direction
HALF_W = 0.23           # m, corridor half-width (robot half-width 0.153 + margin)
SLOW_W = 0.33           # m, wider corridor used only to slow down near obstacles
STOP_CLEAR, SLOW_CLEAR = 0.30, 0.90   # m, clearance at which speed is 0 / full
MIN_SCALE = 0.25
GOAL_TOL = np.radians(4)     # chosen gap within this of the goal bearing -> TRACK
REACHED = 0.08          # m, intermediate waypoint reached
DOCK_REACHED = 0.03     # m, dock waypoint reached
ALIGN_TOL = np.radians(1.5)
ALIGN_W = 0.6           # rad/s max while aligning
SPIN_W = 1.0            # rad/s when boxed in
BLOCKED_WP_R = 0.28     # m, skip an intermediate waypoint if an obstacle sits on it

MODES = {"TRACK": 0, "AVOID": 1, "TURN": 2, "ALIGN": 3, "DONE": 4}


class Navigator:
    def __init__(self, waypoints, dock_heading=0.0):
        self.wps = [np.array(w, dtype=float) for w in waypoints]
        self.dock_heading = dock_heading
        self.tracker = LQRTracker()
        self.i = 0
        self.seg_origin = None
        self.prev_phi = 0.0
        self.was_deviating = False
        self.mode = "TRACK"
        self.done = False

    @property
    def mode_code(self):
        return MODES[self.mode]

    # -- helpers -----------------------------------------------------------------
    def _waypoint_blocked(self, scan, pos, th, goal):
        """True if a LiDAR point lies on top of an intermediate waypoint."""
        x, y = get_points(scan)
        wx = pos[0] + x * np.cos(th) - y * np.sin(th)
        wy = pos[1] + x * np.sin(th) + y * np.cos(th)
        return bool(np.any(np.hypot(wx - goal[0], wy - goal[1]) < BLOCKED_WP_R))

    def _next_waypoint(self, origin):
        """Advance to the next waypoint; the next line starts at `origin`."""
        self.i += 1
        self.seg_origin = np.array(origin, dtype=float)
        self.was_deviating = False

    # -- main entry --------------------------------------------------------------
    def command(self, x_hat, scan):
        """Return (left, right) wheel speeds in rad/s from the estimate and the LiDAR scan."""
        x, y, th = x_hat
        pos = np.array([x, y])
        if self.seg_origin is None:
            self.seg_origin = pos.copy()
        if self.done:
            return 0.0, 0.0

        last = self.i == len(self.wps) - 1
        goal = self.wps[self.i]

        # Intermediate waypoint covered by an obstacle: skip it.
        if not last and self._waypoint_blocked(scan, pos, th, goal):
            self._next_waypoint(pos)              # restart the line from where we are
            return self.command(x_hat, scan)

        d = float(np.linalg.norm(goal - pos))

        # --- reached the waypoint ---
        if d < (DOCK_REACHED if last else REACHED) and self.mode != "ALIGN":
            if last:
                self.mode = "ALIGN"
            else:
                self._next_waypoint(goal)         # next line starts at the waypoint just reached
                return self.command(x_hat, scan)

        # --- ALIGN: rotate in place to the dock heading, then stop ---
        if self.mode == "ALIGN":
            if d > 0.12:                      # drifted off the dock pose: approach again
                self.mode = "TRACK"
            else:
                err = wrap(th - self.dock_heading)
                if abs(err) < ALIGN_TOL:
                    self.mode, self.done = "DONE", True
                    return 0.0, 0.0
                return wheel_speeds(0.0, self.tracker.omega_heading(err, ALIGN_W))

        # --- choose a direction from the LiDAR ---
        psi_goal = np.arctan2(goal[1] - y, goal[0] - x)
        g = wrap(psi_goal - th)
        look = min(LOOK, d + 0.15)            # ignore things beyond the goal (the dock itself)
        phi, found = best_heading(scan, g, self.prev_phi, look, HALF_W)
        self.prev_phi = phi

        if not found:                         # boxed in: spin toward the most open direction
            self.mode = "TURN"
            return wheel_speeds(0.0, SPIN_W * np.sign(phi if phi != 0 else 1.0))

        deviating = abs(wrap(phi - g)) > GOAL_TOL
        if deviating:
            # AVOID: face the chosen gap
            self.mode = "AVOID"
            self.was_deviating = True
            e_h = -phi
            w = self.tracker.omega_heading(e_h)
        else:
            # TRACK: LQR along the line origin -> goal. After a detour restart the line
            # from the current position so the robot is not pulled back across the obstacle.
            self.mode = "TRACK"
            if self.was_deviating:
                self.seg_origin = pos.copy()
                self.was_deviating = False
            seg = goal - self.seg_origin
            psi_s = np.arctan2(seg[1], seg[0])
            e_th = wrap(th - psi_s)
            e_y = -(x - self.seg_origin[0]) * np.sin(psi_s) + (y - self.seg_origin[1]) * np.cos(psi_s)
            w = self.tracker.omega_path(e_y, e_th)
            e_h = wrap(th - psi_goal)

        # --- speed ---
        v = max(min(K_V * d, MAX_V), MIN_V)
        v *= max(np.cos(e_h), 0.0) ** 2
        clear = corridor_clearance(scan, phi, look, SLOW_W)
        v *= np.clip((clear - STOP_CLEAR) / (SLOW_CLEAR - STOP_CLEAR), MIN_SCALE, 1.0)
        if obstacle_in_front(scan) and not (last and d < 0.15):
            v = 0.0                           # emergency stop; the robot still turns away
        return wheel_speeds(v, w)
