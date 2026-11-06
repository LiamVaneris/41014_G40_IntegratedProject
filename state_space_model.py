""" Part A unicycle model, and anaylsis. 
State [x, y, theta], input [v, omega]
x_dot = v cos(theta), y_dot = v sin(theta), theta_dot = omega
Wheels: v = r (wL + wR) / 2, omega = r (wR - wL) / b, r = 0.033 m, b = 0.288 
We assume no wheel slip, flat floor, wheels reach the commanded speed instantly """

import numpy as np

rank = np.linalg.matrix_rank
MARKER = np.array([2.999, -1.0])

# heading 0, speed v
B = np.array([[1, 0], [0, 0], [0, 1]])
def A(v):
    return np.array([[0, 0, 0], [0, 0, v], [0, 0, 0]])

# Controllability
print("Controllability, moving: ", rank(np.hstack([B, A(0.1) @ B])))
print("Controllability, stopped:", rank(np.hstack([B, A(0.0) @ B])))

# Camera measurement Jacobian (from range and bearing)
def H(x, y):
    dx, dy = MARKER - [x, y]
    r = np.hypot(dx, dy)
    return np.array([[-dx / r, -dy / r, 0], [dy / r**2, -dx / r**2, -1]])

# Observability: camera readings while driving straight for 10 s
O = np.vstack([H(0.1 * t, 0) @ (np.eye(3) + A(0.1) * t) for t in range(10)])
print("Observability, range + bearing:", rank(O))
print("Observability, + heading:      ", rank(np.vstack([O, [0, 0, 1]])))