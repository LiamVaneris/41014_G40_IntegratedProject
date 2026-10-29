import time
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

from src.estimation.odometry import Odometry, read_wheel_angles, inside_pi_angles
from src.perception.range_bearing import camera_matrix, measure_range_bearing

ROOT = Path(__file__).parent
CAM_NAME = "front_cam"
IMG_W, IMG_H = 640,480
CAM_FOVY_DEG = 50
SIM_TIME = 10.0
CAM_RATE_HZ = 10
WHEEL_SPEED = 3.0 #rad/s
REAL_TIME = True

def true_pose(data):
    x, y = data.qpos[0], data.qpos[1]
    w, qx, qy, qz = data.qpos[3:7]
    yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return np.array([x,y,yaw])

model = mujoco.MjModel.from_xml_path(str(ROOT/"scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, IMG_H, IMG_W)
c_matrix = camera_matrix(CAM_FOVY_DEG, IMG_W, IMG_H)

odom = Odometry (x0=0.0, y0=0.0, theta0=0.0, seed=0)
odom.update(*read_wheel_angles(model,data))
current_pose = odom.pose.copy()

cam_period = 1.0 / CAM_RATE_HZ
next_cam_time = 0.0
next_print_time = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    wall_start = time.time()

    while viewer.is_running() and data.time <= SIM_TIME:
        data.ctrl[0] = WHEEL_SPEED #left wheel
        data.ctrl[1] = WHEEL_SPEED #right wheel

        mujoco.mj_step(model, data)

        current_pose = odom.update(*read_wheel_angles(model,data))

        if data.time >= next_cam_time:
            renderer.update_scene(data, camera=CAM_NAME)
            image = renderer.render()
            measurement = measure_range_bearing(image, c_matrix)
            next_cam_time += cam_period

            if data.time >= next_print_time:
                truth = true_pose(data)
                if measurement is None:
                    cam_text = "marker not seen"
                else:
                    rng, bearing = measurement
                    print(f" ***CAMERA READINGS***          time={data.time:5.2f}s      range= {rng:.2f}m   bearing={np.degrees(bearing):6.1f}deg")
                    print(f"***ODOMETRY READINGS***         x={current_pose[0]:.3f}m         y={current_pose[1]:.3f}m        th={np.degrees(current_pose[2]):6.1f}deg")
                    print(f" ***EVALUATION ONLY***  GTRUTH  x={truth[0]:.3f}m         y={truth[1]:.3f}m           th={np.degrees(truth[2]):6.1f}deg\n")
                      
                
                next_print_time += 1.0

        viewer.sync()

        # slow the sim down to max with real time
        if REAL_TIME:
            ahead = data.time - (time.time() - wall_start)
            if ahead > 0:
                time.sleep(ahead) # pause sim until real time = sim time

renderer.close()

# FOR EVALUATION ONLY - GROUND TRUTH
truth = true_pose(data)
error = current_pose - truth
error[2] = inside_pi_angles(error[2])
print(f"    ***EVALUATION ONLY***\nFinal true pose x={truth[0]:.3f} y={truth[1]:.3f} th={np.degrees(truth[2]):.1f} deg")
print(f"Final odometry error x={error[0]:.4f} m y={error[1]:.4f} m th={np.degrees(error[2]):.2f} deg")