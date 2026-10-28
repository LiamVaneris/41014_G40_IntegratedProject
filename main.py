import time
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path

from src.perception.range_bearing import camera_matrix, measure_range_bearing

ROOT = Path(__file__).parent
CAM_NAME = "front_cam"
IMG_W, IMG_H = 640,480
CAM_FOVY_DEG = 50
SIM_TIME = 10.0
CAM_RATE_HZ = 10
WHEEL_SPEED = 3.0 #rad/s
REAL_TIME = True

model = mujoco.MjModel.from_xml_path(str(ROOT/"scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, IMG_H, IMG_W)
c_matrix = camera_matrix(CAM_FOVY_DEG, IMG_W, IMG_H)

cam_period = 1.0 / CAM_RATE_HZ
next_cam_time = 0.0
next_print_time = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    wall_start = time.time()

    while viewer.is_running() and data.time < SIM_TIME:
        data.ctrl[0] = WHEEL_SPEED #left wheel
        data.ctrl[1] = WHEEL_SPEED #right wheel

        mujoco.mj_step(model, data)

        if data.time >= next_cam_time:
            renderer.update_scene(data, camera=CAM_NAME)
            image = renderer.render()
            measurement = measure_range_bearing(image, c_matrix)
            next_cam_time += cam_period

            if data.time >= next_print_time:
                if measurement is None:
                    print(f"t= {data.time:5.2f}s marker not seen")
                else:
                    rng, bearing = measurement
                    print(f"t= {data.time:5.2f}s range = {rng:.2f}m "
                        f"bearing = {np.degrees(bearing):6.1f}deg")
                next_print_time += 1.0

        viewer.sync()

        # slow the sim down to max with real time
        if REAL_TIME:
            ahead = data.time - (time.time() - wall_start)

            if ahead > 0:
                time.sleep(ahead) # pause sim until real time = sim time

renderer.close()

# FOR EVALUATION ONLY - GROUND TRUTH
x, y = data.qpos[0], data.qpos[1]
print(f"***EVALUATION ONLY***\nfinal true position: x= {x:.2f}m y= {y:.2f}m")