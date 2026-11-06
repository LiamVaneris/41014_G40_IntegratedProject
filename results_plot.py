"""Plots for the last main.py run (reads results/run.npy). Ground truth is for evaluation only."""
import numpy as np
import matplotlib.pyplot as plt

from src.config import DOCK_POSE, MARKER_XY

d = np.load("results/run.npy")
try:
    blocks = np.load("results/blocks.npy")
except FileNotFoundError:
    blocks = np.zeros((0, 2))

t = d[:, 0]
est, std, odom, true = d[:, 1:4].copy(), d[:, 4:7].copy(), d[:, 7:10].copy(), d[:, 10:13].copy()
mode, seen = d[:, 13], d[:, 14]

err_xy = np.hypot(*(est - true)[:, :2].T)
odom_err = np.hypot(*(odom - true)[:, :2].T)
head_err = np.degrees(np.abs((est[:, 2] - true[:, 2] + np.pi) % (2 * np.pi) - np.pi))

# Heading in degrees for plotting
for a in (est, std, odom, true):
    a[:, 2] = np.degrees(a[:, 2])

# 1) estimate vs truth with uncertainty; shaded strips = marker NOT visible
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
for i, name in enumerate(["x (m)", "y (m)", "theta (deg)"]):
    axes[i].plot(t, true[:, i], "k", label="Truth")
    axes[i].plot(t, odom[:, i], "--", label="Odometry only")
    axes[i].plot(t, est[:, i], label="EKF")
    axes[i].fill_between(t, est[:, i] - 2 * std[:, i], est[:, i] + 2 * std[:, i], alpha=0.3, label="EKF ±2σ")
    axes[i].set_ylabel(name)
    hidden = seen < 0.5
    axes[i].fill_between(t, *axes[i].get_ylim(), where=hidden, color="grey", alpha=0.12,
                         label="marker not visible" if i == 0 else None)
axes[0].legend(ncol=3, fontsize=8)
axes[2].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig("results/estimate_vs_truth.png", dpi=150)

# 2) path in the room with blocks, marker and dock pose
fig2, ax = plt.subplots(figsize=(7, 5))
ax.plot(true[:, 0], true[:, 1], "k", label="True path")
ax.plot(est[:, 0], est[:, 1], "g--", label="EKF path")
for k, (bx, by) in enumerate(blocks):
    ax.add_patch(plt.Rectangle((bx - 0.1, by - 0.1), 0.2, 0.2, color="tab:blue", label="Block" if k == 0 else None))
ax.plot(*MARKER_XY, "rs", label="ArUco marker")
ax.plot(DOCK_POSE[0], DOCK_POSE[1], "m*", ms=12, label="Dock pose")
ax.plot(true[0, 0], true[0, 1], "ko", label="Start")
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig("results/path.png", dpi=150)

# 3) estimation error against the filter's own 2-sigma bound
fig3, ax3 = plt.subplots(figsize=(8, 3.5))
ax3.plot(t, err_xy * 100, label="EKF position error")
ax3.plot(t, odom_err * 100, "--", label="Odometry-only error")
ax3.plot(t, 2 * np.maximum(std[:, 0], std[:, 1]) * 100, "g:", label="EKF 2σ (position)")
ax3.set_xlabel("time (s)")
ax3.set_ylabel("cm")
ax3.legend(fontsize=8)
fig3.tight_layout()
fig3.savefig("results/error.png", dpi=150)

final = true[-1]
print(f"EKF position RMSE:       {np.sqrt(np.mean(err_xy ** 2)) * 100:.1f} cm")
print(f"Odometry position RMSE:  {np.sqrt(np.mean(odom_err ** 2)) * 100:.1f} cm")
print(f"EKF heading RMSE:        {np.sqrt(np.mean(head_err ** 2)):.2f} deg")
print(f"Final true pose: x={final[0]:.3f} y={final[1]:.3f} theta={final[2]:.1f} deg  "
      f"(dock {DOCK_POSE[0]}, {DOCK_POSE[1]}, 0 deg)")
print("saved results/estimate_vs_truth.png, results/path.png, results/error.png")
plt.show()
