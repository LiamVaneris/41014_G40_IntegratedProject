import numpy as np

#unicyle model with state (x,y,theta) and input (v, omega)
B = np.array([[1, 0], [0, 0], [0, 1]])
for v in [0.1, 0.0]:
    A = np.array([[0, 0, 0], [0, 0, v], [0, 0, 0]])
    print("Controllability rank, v =", v, ":", np.linalg.matrix_rank(np.hstack([B, A @ B, A @ A @ B])))

#camera, range and bearing
dx, dy = 2.999, -1.0
r = np.hypot(dx, dy)
H = np.array([[-dx / r, -dy / r, 0], [dy / r**2, -dx / r**2, -1]])
print("Observability rank, range + bearing:", np.linalg.matrix_rank(H))
print("Observability rank, + heading:", np.linalg.matrix_rank(np.vstack([H, [0, 0, 1]])))