"""State-space model, controllability / observability analysis and LQR gains.

Unicycle model:   x' = v cos(theta),  y' = v sin(theta),  theta' = omega
State (x, y, theta), input (v, omega). Linearised about heading theta0 and speed v:

    A = [[0, 0, -v sin(theta0)],      B = [[cos(theta0), 0],
         [0, 0,  v cos(theta0)],           [sin(theta0), 0],
         [0, 0,  0           ]]            [0,           1]]

Camera measurement of the marker at (mx, my):  z = [range, bearing]
    range   = sqrt(dx^2 + dy^2),            dx = mx - x, dy = my - y
    bearing = atan2(dy, dx) - theta
    H = [[-dx/r,   -dy/r,   0],
         [ dy/r^2, -dx/r^2, -1]]
"""
import numpy as np

from src.config import MARKER_XY
from src.control.controller import design_lqr, Q_LQR, R_LQR, V_NOM


def linearise(v, theta0=0.0):
    A = np.array([[0, 0, -v * np.sin(theta0)],
                  [0, 0, v * np.cos(theta0)],
                  [0, 0, 0]])
    B = np.array([[np.cos(theta0), 0],
                  [np.sin(theta0), 0],
                  [0, 1]])
    return A, B


def marker_H(x, y):
    dx, dy = MARKER_XY - np.array([x, y])
    r = np.hypot(dx, dy)
    return np.array([[-dx / r, -dy / r, 0], [dy / r ** 2, -dx / r ** 2, -1]])


def ctrb(A, B):
    return np.hstack([np.linalg.matrix_power(A, k) @ B for k in range(A.shape[0])])


def obsv(A, H):
    return np.vstack([H @ np.linalg.matrix_power(A, k) for k in range(A.shape[0])])


rank = np.linalg.matrix_rank

print("--- Controllability of the linearised model (3 states) ---")
for v in (V_NOM, 0.0):
    A, B = linearise(v)
    print(f"v = {v}: rank = {rank(ctrb(A, B))}")
print("(v = 0: rank 2 - the robot cannot move sideways, so (x,y,theta) is not locally controllable"
      " by linear feedback at standstill; it is controllable once moving.)")

print("\n--- Observability with the camera (range + bearing to one marker) ---")
H = marker_H(0.0, 0.0)
for v in (V_NOM, 0.0):
    A, _ = linearise(v)
    print(f"v = {v}: rank [H; HA; HA^2] = {rank(obsv(A, H))}")
print("Range + bearing is only 2 measurements for 3 states, so a single instant is rank 2."
      " The pose becomes observable (rank 3) only while the robot MOVES and the marker stays in view.")
print("Odometry alone (no marker) measures nothing absolute: x, y, theta are unobservable (rank 0),"
      " so the error grows without bound between marker sightings.")

print("\n--- LQR for the path-tracking error model  e_y' = v e_theta,  e_theta' = omega ---")
K, A2, B2 = design_lqr()
print(f"Q = diag{tuple(float(q) for q in np.diag(Q_LQR))}  R = {R_LQR[0, 0]}   v = {V_NOM} m/s")
print(f"K = {K}   ->   omega = -{K[0]:.3f} e_y - {K[1]:.3f} e_theta")
print("closed-loop eigenvalues:", np.linalg.eigvals(A2 - B2 @ K.reshape(1, 2)))
