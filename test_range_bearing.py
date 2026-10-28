import mujoco
import numpy as np
from src.perception.range_bearing import camera_matrix, measure_range_bearing

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, 480, 640)
renderer.update_scene(data, camera="front_cam")
image = renderer.render()
renderer.close()

K = camera_matrix(50, 640, 480)
result = measure_range_bearing(image, K)

if result is None:
    print("Marker not detected")
else:
    distance, angle = result
    print(f"Range: {distance:.2f} m   Bearing: {np.degrees(angle):.1f} deg")