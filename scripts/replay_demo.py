"""Replay a recorded teleop episode (data/demos/*.hdf5) and save an mp4 to outputs/.

    .venv\\Scripts\\python scripts\\replay_demo.py                       # newest file in data/demos
    .venv\\Scripts\\python scripts\\replay_demo.py data\\demos\\demo_X.hdf5 --mode actions

--mode states  (default) sets the recorded flattened MuJoCo state at every step (exact kinematic replay)
--mode actions resets to the first recorded state and re-applies the recorded 14-d actions open-loop;
               prints the end-effector drift w.r.t. the recording.
Each video frame = replayed render (camera --camera) | recorded rgbd_cam RGB | recorded depth.
A PNG of the last frame is also saved next to the mp4.
"""
import argparse
import glob
import os

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
from _common import OUT, ROOT, make_env


def depth_vis(d, size):
    d = np.nan_to_num(d, nan=0.0)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    u8 = (255 * np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)).astype(np.uint8)
    col = cv2.applyColorMap(255 - u8, cv2.COLORMAP_JET)[:, :, ::-1]
    return cv2.resize(col, (size, size), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="hdf5 episode (default: newest in data/demos)")
    ap.add_argument("--mode", choices=["states", "actions"], default="states")
    ap.add_argument("--camera", default="overview")
    ap.add_argument("--out", default=None, help="output mp4 (default outputs/replay_<name>.mp4)")
    args = ap.parse_args()

    path = args.file
    if path is None:
        files = sorted(glob.glob(os.path.join(ROOT, "data", "demos", "*.hdf5")), key=os.path.getmtime)
        assert files, "no demos in data/demos"
        path = files[-1]
    name = os.path.splitext(os.path.basename(path))[0]
    out = args.out or os.path.join(OUT, f"replay_{name}.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    with h5py.File(path, "r") as f:
        xml = f.attrs["model_file"]
        states, actions = f["states"][:], f["actions"][:]
        rgb, depth = f["obs/rgbd_cam_rgb"][:], f["obs/rgbd_cam_depth"][:]
        eef_rec = [f["obs/robot0_eef_pos"][:], f["obs/robot1_eef_pos"][:]]
        success = bool(f.attrs["success"])
    T = len(actions)
    print(f"{path}: {T} steps, success={success}, mode={args.mode}")

    env = make_env(has_offscreen_renderer=True, hard_reset=False)
    env.reset()
    env.reset_from_xml_string(xml)  # exact model used during recording
    sim = env.sim
    H, W, S = 480, 640, 240

    def frame(t):
        img, _ = sim.render(camera_name=args.camera, width=W, height=H, depth=True)
        img = img[::-1]
        side = np.concatenate([cv2.resize(rgb[t], (S, S), interpolation=cv2.INTER_NEAREST),
                               depth_vis(depth[t], S)], axis=0)
        fr = np.concatenate([img, side], axis=1).copy()
        cv2.putText(fr, f"{name} t={t}/{T} ({args.mode})", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return fr

    frames, drift = [], []
    sim.set_state_from_flattened(states[0])
    sim.forward()
    for t in range(T):
        if args.mode == "states":
            sim.set_state_from_flattened(states[t])
            sim.forward()
        else:
            if t > 0:
                env.step(actions[t - 1])
            obs = env._get_observations(force_update=True)
            drift.append([np.linalg.norm(obs[f"robot{i}_eef_pos"] - eef_rec[i][t]) for i in range(2)])
        frames.append(frame(t))
    env.close()

    imageio.mimsave(out, frames, fps=20)
    png = os.path.splitext(out)[0] + "_last.png"
    imageio.imwrite(png, frames[-1])
    imageio.imwrite(os.path.splitext(out)[0] + "_first.png", frames[0])
    print(f"saved {out} ({len(frames)} frames) and {png}")
    if drift:
        drift = np.array(drift)
        print(f"action replay eef drift [m]: right max {drift[:, 0].max():.4f}, left max {drift[:, 1].max():.4f}")


if __name__ == "__main__":
    main()
