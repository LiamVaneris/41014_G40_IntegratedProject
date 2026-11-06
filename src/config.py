"""Shared constants for the TurtleBot3 docking project (one place to change them)."""
import numpy as np

# --- Scene geometry (must match scene.xml) -----------------------------------
MARKER_XY = np.array([2.999, -1.0])   # ArUco marker centre in the world (m)
MARKER_NORMAL = np.pi                 # world angle of the marker's outward normal (it faces -x)

# Waypoints: [dogleg, pre-dock point on the dock axis, dock stand-off].
# The last waypoint IS the dock pose; the robot should end there facing +x (heading 0).
WAYPOINTS = [(0.8, -1.0), (2.1, -1.0), (2.6, -1.0)]
DOCK_POSE = (2.6, -1.0, 0.0)          # x, y, heading

# --- Robot (TurtleBot3 Waffle Pi) --------------------------------------------
# Effective odometry parameters from calibrate_odometry.py. The geometric values are
# 0.033 m / 0.288 m, but in the simulation the tyres slip, so the robot moves ~15% less
# and turns ~27% less than the nominal wheel geometry implies. Using the calibrated values
# removes that systematic odometry error.
WHEEL_RADIUS = 0.0281                 # m  (nominal 0.033)
TRACK_WIDTH = 0.342                   # m  (nominal 0.288)
WHEEL_MAX = 7.5                       # rad/s (actuator limit in the XML is 7.88)
LIDAR_X = -0.01                       # LiDAR offset from the base origin along x (m)

# --- Timing --------------------------------------------------------------------
SIM_TIME = 90.0                       # s, hard stop
CTRL_RATE_HZ = 50                     # navigation / LiDAR rate
CAM_RATE_HZ = 10                      # camera + EKF update rate

# --- Camera --------------------------------------------------------------------
CAM_NAME = "front_cam"
IMG_W, IMG_H = 640, 480
CAM_FOVY_DEG = 50
