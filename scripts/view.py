"""Interactive MuJoCo viewer for the BimanualAssist scene (run this yourself; needs a display).

Arms hold their pose (zero OSC delta action); you can drag objects with the mouse
(double-click a body, then Ctrl + right-drag to apply a force). Close the window to quit.
Pass --still to only show the scene without stepping the controllers.
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from _common import make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", action="store_true", help="don't step the simulation")
    args = ap.parse_args()

    env = make_env()
    env.reset()
    m, d = env.sim.model._model, env.sim.data._data
    dt = 1.0 / env.control_freq
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.opt.geomgroup[0] = 0  # hide collision geoms (robosuite puts them in group 0)
        viewer.cam.lookat[:] = (0.0, 0.0, 0.9)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 210, -30
        while viewer.is_running():
            t0 = time.perf_counter()
            if not args.still:
                obs, _, _, _ = env.step(np.zeros(env.action_dim))
            viewer.sync()
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    env.close()


if __name__ == "__main__":
    main()
