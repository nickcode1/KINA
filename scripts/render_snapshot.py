"""Offscreen-render the RGB-D camera (rgb + depth) and overview cameras to outputs/*.png."""
import os

import cv2
import imageio.v2 as imageio
import numpy as np
from _common import OUT, make_env
from robosuite.utils.camera_utils import get_real_depth_map

W, H = 640, 480
CAMERAS = ["rgbd_cam", "overview", "frontview", "birdview"]


def depth_vis(depth_m, near=None, far=2.5):
    near = depth_m.min() if near is None else near
    d = np.clip((depth_m - near) / (far - near), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((255 * (1 - d)).astype(np.uint8), cv2.COLORMAP_TURBO),
                        cv2.COLOR_BGR2RGB)


def main(settle_steps=20):
    os.makedirs(OUT, exist_ok=True)
    env = make_env(has_offscreen_renderer=True)
    env.reset()
    for _ in range(settle_steps):  # let objects settle (1 s)
        env.step(np.zeros(env.action_dim))

    # Render all cameras first, then post-process (interleaving post-processing with renders
    # gave black frames on Windows/GLFW in testing). OpenGL images are bottom-up -> flip vertically.
    frames = {}
    for cam in CAMERAS:
        rgb, depth = env.sim.render(camera_name=cam, width=W, height=H, depth=True)
        frames[cam] = (rgb[::-1].copy(), depth[::-1].copy())
    depth_m = get_real_depth_map(env.sim, frames["rgbd_cam"][1])  # metres
    env.close()

    for cam, (rgb, depth) in frames.items():
        imageio.imwrite(os.path.join(OUT, f"{cam}.png" if cam != "rgbd_cam" else "rgbd_cam_rgb.png"), rgb)
    imageio.imwrite(os.path.join(OUT, "rgbd_cam_depth.png"), depth_vis(depth_m))
    np.save(os.path.join(OUT, "rgbd_cam_depth_m.npy"), depth_m.astype(np.float32))
    print(f"rgbd_cam depth range: {depth_m.min():.3f} - {depth_m.max():.3f} m")
    print("mean intensity:", {c: round(float(f[0].mean()), 1) for c, f in frames.items()})
    print("saved PNGs to", OUT)


if __name__ == "__main__":
    main()
