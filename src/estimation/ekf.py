import numpy as np

MARKER = np.array([2.999, -1.0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKF:
    def __init__(self, x0, P0):
        self.x = np.array(x0, dtype=float)   # estimate [x, y, theta]
        self.P = np.array(P0, dtype=float)        # uncertainty
        self.Q = np.diag([0.0005, 0.0005, 0.001]) ** 2    #odometry noise per step
        self.R = np.diag([0.05, np.radians(1)]) ** 2     # camera noise: range, bearing

    def predict(self, ds, dtheta):
        th = self.x[2] + dtheta / 2
        self.x += [ds * np.cos(th), ds * np.sin(th), dtheta]
        self.x[2] = wrap(self.x[2])

        F = np.array([[1, 0, -ds * np.sin(th)],
                      [0, 1,  ds * np.cos(th)],
                      [0, 0, 1]])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        dx, dy = MARKER - self.x[:2]
        r = np.hypot(dx, dy)

        expected = np.array([r, wrap(np.arctan2(dy, dx) - self.x[2])])
        error = np.array(z) - expected
        error[1] = wrap(error[1])

        H = np.array([[-dx / r,    -dy / r,    0],
                      [ dy / r**2, -dx / r**2, -1]])
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)

        self.x += K @ error
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P