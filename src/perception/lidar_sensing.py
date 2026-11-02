import numpy as np
import mujoco


# Returns a Nx2 array:
# Column 0 = angle (radians)
# Column 1 = distance (metres)
def get_scan(m, d):

    ids = np.where(
        m.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER
    )[0]

    dist = d.sensordata[m.sensor_adr[ids]]

    # Replace "no hit" values with infinity
    dist = np.where(dist < 0, np.inf, dist)

    angles = np.arange(len(ids)) * 2 * np.pi / len(ids)

    return np.column_stack((angles, dist))


# Convert LiDAR polar coordinates to robot-frame x,y coordinates
def get_points(scan):

    angles = scan[:, 0]
    distances = scan[:, 1]

    xpoints = distances * np.cos(angles)
    ypoints = distances * np.sin(angles)

    return xpoints, ypoints


# Returns True if an obstacle is inside:
# 1m forward
# 20cm left
# 20cm right
def obstacle_in_front(scan):

    xpoints, ypoints = get_points(scan)

    return np.any(
        (xpoints >= 0.0) &
        (xpoints <= 0.25) &
        (ypoints >= -0.2) &
        (ypoints <= 0.2)
    )
