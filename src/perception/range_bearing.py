import cv2
import numpy as np

MARKER_ID = 0
MARKER_SIZE = 0.2 * 400 / 600              # m, black square only (~0.133 m)
CAM_IN_BASE = np.array([0.09, 0.0, 0.107])  # camera position on the robot (from the XML)

detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def camera_matrix(fovy_deg, width, height):
    """Camera calibration: focal length and image centre from the field of view."""
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
    return np.array([[f, 0, width / 2],
                     [0, f, height / 2],
                     [0, 0, 1]])


def measure_range_bearing(image, K):
    """Return (range in m, bearing in rad) to the marker, or None if it isn't seen."""
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or MARKER_ID not in ids.flatten():
        return None
    i = list(ids.flatten()).index(MARKER_ID)

    # Real corner positions of the marker (metres), in the order OpenCV detects them
    s = MARKER_SIZE / 2
    real_corners = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

    # Where the marker is relative to the camera
    ok, _, tvec = cv2.solvePnP(real_corners, corners[i].reshape(4, 2), K, None,
                               flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    x_cam, y_cam, z_cam = tvec.flatten()   # camera frame: x right, y down, z forward

    # Convert to the robot frame: x forward, y left
    x_robot = z_cam + CAM_IN_BASE[0]
    y_robot = -x_cam + CAM_IN_BASE[1]

    return np.hypot(x_robot, y_robot), np.arctan2(y_robot, x_robot)