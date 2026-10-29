import numpy as np
from math import pi, sin, cos

WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.288

def inside_pi_angles(i):
    #keeping every angle between -pi and pi
    return (i + pi) % (2 * pi) - pi

def read_wheel_angles(model, data):
    left = data.qpos[model.joint("wheel_left").qposadr[0]]
    right = data.qpos[model.joint("wheel_right").qposadr[0]]
    return left, right

class Odometry:
    def __init__(self, x0=0.0, y0=0.0, theta0=0.0, wheel_radius=WHEEL_RADIUS, track_width=TRACK_WIDTH,
                noise_std=0.02, scale_error_std=0.01, seed=None):
        self.pose = np.array([x0, y0, theta0], dtype=float)
        self.r = wheel_radius
        self.b = track_width

        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.gain_left = 1.0 + self.rng.normal(0.0, scale_error_std)
        self.gain_right = 1.0 + self.rng.normal(0.0, scale_error_std)
        self.prev = None

    def update(self, left_angle, right_angle):
        if self.prev is None:
            self.prev = (left_angle, right_angle)
            return self.pose.copy()

        d_left = left_angle - self.prev[0]
        d_right = right_angle - self.prev[1]
        self.prev = (left_angle, right_angle)

        d_left *= self.gain_left * (1.0 + self.rng.normal(0.0, self.noise_std))
        d_right *= self.gain_right * (1.0 + self.rng.normal(0.0, self.noise_std))

        s_left = self.r * d_left
        s_right = self.r * d_right

        ds = (s_right + s_left)/ 2.0
        dtheta = (s_right - s_left) /self.b

        x, y, theta = self.pose
        self.pose[0] = x + ds * cos(theta + dtheta / 2.0)
        self.pose[1] = y + ds * sin(theta + dtheta / 2.0)
        self.pose[2] = inside_pi_angles(theta + dtheta)

        return self.pose.copy()