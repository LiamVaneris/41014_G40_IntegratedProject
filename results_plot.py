import numpy as np
import matplotlib.pyplot as plt

d = np.load("results/run.npy")
t = d[:, 0]
est, std, odom, true = d[:, 1:4], d[:, 4:7], d[:, 7:10], d[:, 10:13]

# Show heading in degrees instead of radians
for a in (est, std, odom, true):
    a[:, 2] = np.degrees(a[:, 2])

fig, axes = plt.subplots(3, 1, sharex=True)
for i, name in enumerate(["x (m)", "y (m)", "theta (deg)"]):
    axes[i].plot(t, true[:, i], "k", label="Truth")
    axes[i].plot(t, odom[:, i], "--", label="Odometry")
    axes[i].plot(t, est[:, i], label="EKF")
    axes[i].fill_between(t, est[:, i] - 2 * std[:, i], est[:, i] + 2 * std[:, i], alpha=0.3, label="EKF ±2σ")
    axes[i].set_ylabel(name)
axes[0].legend()
axes[2].set_xlabel("time (s)")
plt.savefig("results/estimate_vs_truth.png")

ekf_err = np.hypot(*(est - true)[:, :2].T)
odom_err = np.hypot(*(odom - true)[:, :2].T)
print(f"EKF RMSE:      {np.sqrt(np.mean(ekf_err**2)):.3f} m")
print(f"Odometry RMSE: {np.sqrt(np.mean(odom_err**2)):.3f} m")
plt.show()