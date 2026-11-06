"""EKF for the robot pose [x, y, theta].

Predict : wheel-odometry increments (ds, dtheta) through the unicycle model.
Update 1: camera range + bearing to the ArUco marker (known world position).
Update 2: camera-derived heading from the marker's orientation (close range only).

Both updates are gated on the normalised innovation, so a bad detection is rejected
instead of dragging the estimate.
"""
import numpy as np

from src.config import MARKER_XY, MARKER_NORMAL

MARKER = MARKER_XY


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    def __init__(self, x0, P0, gate=13.8):
        self.x = np.array(x0, dtype=float)               # estimate [x, y, theta]
        self.P = np.array(P0, dtype=float)               # uncertainty
        self.Q = np.diag([0.0005, 0.0005, 0.001]) ** 2   # odometry noise per step (std: m, m, rad)
        self.R = np.diag([0.05, np.radians(1)]) ** 2     # camera noise: range (m), bearing (rad)
        self.gate = gate                                 # chi-square gate (2 dof, 99.9% = 13.8)
        self.n_accepted = 0
        self.n_rejected = 0
        self.max_streak = 5                              # after this many rejections in a row the
        self._streak = 0                                 # filter, not the camera, is wrong: accept

    # ------------------------------------------------------------------ predict
    def predict(self, ds, dtheta):
        th = self.x[2] + dtheta / 2
        self.x += [ds * np.cos(th), ds * np.sin(th), dtheta]
        self.x[2] = wrap(self.x[2])

        F = np.array([[1, 0, -ds * np.sin(th)],
                      [0, 1,  ds * np.cos(th)],
                      [0, 0, 1]])
        self.P = F @ self.P @ F.T + self.Q

    # ------------------------------------------------------- shared update step
    def _apply(self, error, H, R):
        S = H @ self.P @ H.T + R
        d2 = float(error @ np.linalg.solve(S, error))
        if d2 > self.gate and self._streak < self.max_streak:
            self.n_rejected += 1
            self._streak += 1
            return False
        self._streak = 0
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ error
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T    # Joseph form (stays symmetric / PSD)
        self.n_accepted += 1
        return True

    # ----------------------------------------------------- camera: range, bearing
    def update(self, z):
        if not np.all(np.isfinite(z)):
            return False
        dx, dy = MARKER - self.x[:2]
        r = np.hypot(dx, dy)

        expected = np.array([r, wrap(np.arctan2(dy, dx) - self.x[2])])
        error = np.array(z, dtype=float) - expected
        error[1] = wrap(error[1])

        H = np.array([[-dx / r,    -dy / r,    0],
                      [ dy / r**2, -dx / r**2, -1]])
        return self._apply(error, H, self.R)

    # --------------------------------------------- camera: heading from marker normal
    def update_heading(self, marker_angle, sigma):
        """theta = (world angle of marker normal) - (marker normal angle in the robot frame)."""
        if not np.isfinite(marker_angle):
            return False
        z = wrap(MARKER_NORMAL - marker_angle)
        error = np.array([wrap(z - self.x[2])])
        H = np.array([[0.0, 0.0, 1.0]])
        R = np.array([[sigma ** 2]])
        # 3-sigma gate on this scalar measurement
        S = float((H @ self.P @ H.T + R)[0, 0])
        if error[0] ** 2 > 9.0 * S:
            self.n_rejected += 1
            return False
        K = self.P @ H.T / S
        self.x += (K @ error).ravel()
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.n_accepted += 1
        return True
