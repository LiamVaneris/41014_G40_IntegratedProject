"""2D LiDAR processing: 360 rangefinder beams, beam 0 straight ahead, 1 deg apart, CCW."""
import numpy as np
import mujoco

from src.config import LIDAR_X

_ids_cache = {}


def get_scan(m, d):
    """Return an Nx2 array: column 0 = beam angle (rad), column 1 = range (m, inf = no hit)."""
    key = id(m)
    if key not in _ids_cache:
        _ids_cache[key] = np.where(m.sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER)[0]
    ids = _ids_cache[key]

    dist = d.sensordata[m.sensor_adr[ids]]
    dist = np.where(dist < 0, np.inf, dist)            # "no hit" -> infinity
    angles = np.arange(len(ids)) * 2 * np.pi / len(ids)
    return np.column_stack((angles, dist))


def get_points(scan):
    """Scan -> (x, y) points in the robot frame (x forward, y left). Misses are dropped."""
    ok = np.isfinite(scan[:, 1])
    ang, dist = scan[ok, 0], scan[ok, 1]
    return dist * np.cos(ang) + LIDAR_X, dist * np.sin(ang)


def obstacle_in_front(scan, forward=0.25, half_width=0.2):
    """Emergency check: True if anything is within `forward` m ahead and `half_width` m either side."""
    x, y = get_points(scan)
    return bool(np.any((x >= 0.0) & (x <= forward) & (np.abs(y) <= half_width)))


def corridor_clearance(scan, phi, max_range, half_width):
    """Distance to the first obstacle inside a corridor pointing at angle phi (robot frame).

    The corridor is `half_width` either side of the line and `max_range` long.
    Returns inf if the corridor is empty.
    """
    x, y = get_points(scan)
    along = x * np.cos(phi) + y * np.sin(phi)
    lateral = -x * np.sin(phi) + y * np.cos(phi)
    hit = (along > 0.0) & (along < max_range) & (np.abs(lateral) < half_width)
    return float(along[hit].min()) if np.any(hit) else float("inf")


def best_heading(scan, goal_bearing, prev_phi, max_range, half_width,
                 span=np.radians(120), step=np.radians(2), hysteresis=0.3):
    """Follow-the-gap steering: pick the free direction closest to the goal.

    A direction phi (robot frame) is free if its corridor is empty. Among free
    directions the one with the lowest cost is chosen:
        cost = |phi - goal_bearing| + hysteresis * |phi - previous choice|
    The second term stops the robot dithering between two equally good gaps.

    Returns (phi, found). found=False means every direction is blocked; phi is
    then the direction with the most room (used to spin out of a corner).
    """
    x, y = get_points(scan)
    phis = np.arange(-span, span + 1e-9, step)
    c, s = np.cos(phis)[:, None], np.sin(phis)[:, None]
    along = x[None, :] * c + y[None, :] * s
    lateral = -x[None, :] * s + y[None, :] * c
    hit = (along > 0.0) & (along < max_range) & (np.abs(lateral) < half_width)
    blocked = hit.any(axis=1)

    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    cost = np.abs(wrap(phis - goal_bearing)) + hysteresis * np.abs(wrap(phis - prev_phi))

    if not blocked.all():
        return float(phis[np.argmin(np.where(blocked, np.inf, cost))]), True

    room = np.where(hit, along, max_range).min(axis=1)
    return float(phis[np.argmax(room - 0.01 * cost)]), False
