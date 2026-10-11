"""Close-up renders of the fork and the jar (free cameras, no change to the scene):
  outputs/closeup_fork.png : fork from above-front | low side view (handle / neck / tine profile)
  outputs/closeup_jar.png  : jar + knurled lid | same view with the lid hidden (shows the hollow jar)"""
import os

import imageio.v2 as imageio
import mujoco
import numpy as np
from _common import OUT, make_env

W, H = 640, 480


def render(model, data, lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, camera=cam)
        return r.render().copy()


def main(settle_steps=20):
    os.makedirs(OUT, exist_ok=True)
    env = make_env()
    env.reset()
    for _ in range(settle_steps):
        env.step(np.zeros(env.action_dim))
    m, d = env.sim.model._model, env.sim.data._data
    fork = d.xpos[env.fork_body_id].copy()
    jar = d.xpos[env.jar_body_id].copy()

    fork_imgs = [render(m, d, fork + (0.02, 0, 0), 0.32, 120, -40),   # from the front-left, above
                 render(m, d, fork + (0.02, 0, 0), 0.26, 90, -6)]     # low side view
    jar_imgs = [render(m, d, jar + (0, 0, 0.01), 0.32, 200, -25)]
    lid_body = m.body(env.jar.lid_body).id
    lid_geoms = np.where(m.geom_bodyid == lid_body)[0]
    alpha = m.geom_rgba[lid_geoms, 3].copy()
    m.geom_rgba[lid_geoms, 3] = 0.0  # hide the lid to show the cavity
    jar_imgs.append(render(m, d, jar + (0, 0, 0.01), 0.30, 200, -55))
    m.geom_rgba[lid_geoms, 3] = alpha
    env.close()

    imageio.imwrite(os.path.join(OUT, "closeup_fork.png"), np.concatenate(fork_imgs, axis=1))
    imageio.imwrite(os.path.join(OUT, "closeup_jar.png"), np.concatenate(jar_imgs, axis=1))
    print("saved closeup_fork.png, closeup_jar.png to", OUT)


if __name__ == "__main__":
    main()
