import cv2
import mujoco

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, 480, 640)
renderer.update_scene(data, camera="front_cam")
image = renderer.render()
renderer.close()

cv2.imwrite("camera_view.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
corners, ids, _ = detector.detectMarkers(image)
print("Marker IDs found:", ids)