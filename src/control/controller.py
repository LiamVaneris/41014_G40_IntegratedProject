import numpy as np

WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.288

K_V = 0.5
K_W = 2.0
MAX_V = 0.15
REACHED = 0.05


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Controller:
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.i = 0

    def command(self, x_hat):
        """Wheel speeds from the EKF estimate x_hat = (x, y, theta). Never uses ground truth."""
        if self.i >= len(self.waypoints):
            return 0.0, 0.0

        x, y, theta = x_hat
        gx, gy = self.waypoints[self.i]
        distance = np.hypot(gx - x, gy - y)
        heading_error = wrap(np.arctan2(gy - y, gx - x) - theta)

        if distance < REACHED:
            self.i += 1
            return 0.0, 0.0

        # State feedback
        w = K_W * heading_error
        v = min(K_V * distance, MAX_V) * np.cos(heading_error)
        v = max(v, 0.0)

        # Convert to wheel speeds
        left = (v - w * TRACK_WIDTH / 2) / WHEEL_RADIUS
        right = (v + w * TRACK_WIDTH / 2) / WHEEL_RADIUS
        return left, right