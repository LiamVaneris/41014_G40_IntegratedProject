"""ArUco marker detection -> range, bearing and marker orientation in the robot frame."""
import cv2
import numpy as np

MARKER_ID = 0
MARKER_SIZE = 0.2 * 400 / 600              # m, black square only (~0.133 m)
CAM_IN_BASE = np.array([0.09, 0.0, 0.107])  # camera position on the robot (from the XML)

detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))

# Real corner positions of the marker (metres), in the order OpenCV detects them
_s = MARKER_SIZE / 2
_REAL_CORNERS = np.array([[-_s, _s, 0], [_s, _s, 0], [_s, -_s, 0], [-_s, -_s, 0]], dtype=np.float32)


def camera_matrix(fovy_deg, width, height):
    """Camera calibration: focal length and image centre from the field of view.

    Ideal pinhole model (square pixels, no lens distortion), which is exactly what
    the MuJoCo camera renders, so no distortion coefficients are needed.
    """
    f = (height / 2) / np.tan(np.radians(fovy_deg) / 2)
    return np.array([[f, 0, width / 2],
                     [0, f, height / 2],
                     [0, 0, 1]])


def measure_marker(image, K):
    """Return (range m, bearing rad, marker_angle rad) or None if the marker isn't seen.

    range, bearing : horizontal distance and direction to the marker centre from the
                     robot origin (x forward, y left).
    marker_angle   : direction of the marker's outward normal in the robot frame. With the
                     marker's known world orientation this gives the robot heading directly,
                     but it is only accurate at close range (see EKF.update_heading).
    """
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or MARKER_ID not in ids.flatten():
        return None
    i = list(ids.flatten()).index(MARKER_ID)

    ok, rvec, tvec = cv2.solvePnP(_REAL_CORNERS, corners[i].reshape(4, 2), K, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    x_cam, _, z_cam = tvec.flatten()       # camera frame: x right, y down, z forward

    # Camera frame -> robot frame (x forward, y left)
    x_robot = z_cam + CAM_IN_BASE[0]
    y_robot = -x_cam + CAM_IN_BASE[1]

    # Marker normal (pointing back toward the camera), in the robot frame
    normal = cv2.Rodrigues(rvec)[0][:, 2].copy()
    if normal[2] > 0:
        normal = -normal
    marker_angle = np.arctan2(-normal[0], normal[2])

    out = (np.hypot(x_robot, y_robot), np.arctan2(y_robot, x_robot), marker_angle)
    if not np.all(np.isfinite(out)):          # degenerate pose from solvePnP: treat as not seen
        return None
    return out


def measure_range_bearing(image, K):
    """Backwards-compatible wrapper: (range, bearing) or None."""
    m = measure_marker(image, K)
    return None if m is None else (m[0], m[1])
