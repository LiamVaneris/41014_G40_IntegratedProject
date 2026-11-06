"""LQR path-tracking controller for the unicycle (differential-drive) model.

Model (see state_space_model.py): states (x, y, theta), inputs (v, omega).
Linearising about a straight segment at speed v gives the error dynamics

    e_y'     = v * e_theta        (cross-track error, + = left of the path)
    e_theta' = omega              (heading error relative to the path)

    A = [[0, v], [0, 0]],  B = [[0], [1]],   omega = -K [e_y, e_theta]

K comes from the continuous-time algebraic Riccati equation, so the gains are
derived from the model rather than hand tuned. The controller is only ever
given the EKF estimate - never ground truth.
"""
import numpy as np
from scipy.linalg import solve_continuous_are

from src.config import WHEEL_RADIUS, TRACK_WIDTH, WHEEL_MAX

V_NOM = 0.12                       # m/s, speed the linearisation is done at
Q_LQR = np.diag([10.0, 1.0])       # penalty on [cross-track (m^2), heading (rad^2)]
R_LQR = np.array([[0.5]])          # penalty on turn rate (rad/s)^2
W_MAX = 1.5                        # rad/s, turn-rate limit


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def design_lqr(v=V_NOM, Q=Q_LQR, R=R_LQR):
    """Return (K, A, B) for the cross-track / heading error model."""
    A = np.array([[0.0, v], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K.ravel(), A, B


class LQRTracker:
    def __init__(self):
        self.K, _, _ = design_lqr()

    def omega_path(self, e_y, e_theta):
        """Turn rate to follow a line: uses both cross-track and heading error."""
        return float(np.clip(-(self.K[0] * e_y + self.K[1] * e_theta), -W_MAX, W_MAX))

    def omega_heading(self, e_theta, limit=W_MAX):
        """Turn rate to face a direction (cross-track term dropped)."""
        return float(np.clip(-self.K[1] * e_theta, -limit, limit))


def wheel_speeds(v, w):
    """(v m/s, omega rad/s) -> (left, right) wheel speeds in rad/s, desaturated.

    If either wheel would exceed WHEEL_MAX both are scaled together so the
    robot keeps the commanded curvature instead of clipping one wheel.
    """
    left = (v - w * TRACK_WIDTH / 2) / WHEEL_RADIUS
    right = (v + w * TRACK_WIDTH / 2) / WHEEL_RADIUS
    m = max(abs(left), abs(right))
    if m > WHEEL_MAX:
        left, right = left * WHEEL_MAX / m, right * WHEEL_MAX / m
    return left, right
