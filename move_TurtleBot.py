from pathlib import Path
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "scene.xml"))
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[0] = 3.0   # left wheel speed (rad/s)
        data.ctrl[1] = 3.0   # right wheel speed (rad/s)
        mujoco.mj_step(model, data)
        viewer.sync()