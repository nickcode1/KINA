"""Headless smoke test of the teleop pipeline (no window, no real keyboard).

Feeds a scripted sequence of synthetic key presses/releases into the SAME TeleopKeyboard ->
Device.input2action -> TeleopSession.step -> EpisodeRecorder path that scripts/teleop.py uses,
records one episode to data/demos/selftest_*.hdf5, checks that the arms moved as commanded,
and (unless --no-replay) runs replay_demo.py on the file.
"""
import argparse
import os
import subprocess
import sys

import h5py
import numpy as np
from _common import ROOT, make_env
from teleop import DEMO_DIR, TeleopKeyboard, TeleopSession

# (n_steps, keys held during these steps, discrete key taps at the start of the segment)
SCRIPT = [
    (10, [], []),                      # settle
    (25, ["up"], []),                  # right arm +x
    (20, ["e"], []),                   # right arm up
    (10, [], ["f"]),                   # close right gripper
    (20, ["a"], []),                   # yaw +
    (5, [], ["n"]),                    # switch to LEFT arm
    (25, ["left"], []),                # left arm +y
    (20, ["q", "shift"], []),          # left arm down, slow mode
    (5, [], ["y"]),                    # mark success
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-replay", action="store_true")
    args = ap.parse_args()

    env = make_env(has_offscreen_renderer=True, hard_reset=False)
    dev = TeleopKeyboard(env)
    sess = TeleopSession(env, dev, record=True, prefix="selftest")

    def eef(i):
        return np.array(sess.obs[f"robot{i}_eef_pos"])

    marks = {}
    for k, (n, held, taps) in enumerate(SCRIPT):
        for t in taps:  # same entry points the pynput listener calls
            dev.press(t)
            dev.release(t)
        for h in held:
            dev.press(h)
        marks[k] = (eef(0), eef(1))
        for _ in range(n):
            sess.step()
        for h in held:
            dev.release(h)
    marks[len(SCRIPT)] = (eef(0), eef(1))
    n_steps = len(sess.recorder)
    dev.press("enter")  # save + reset
    dev.release("enter")
    sess.step()
    assert sess.saved, "episode was not saved"
    path = sess.saved[-1]
    env.close()

    def disp(seg, arm):
        return marks[seg + 1][arm] - marks[seg][arm]

    checks = {
        "right +x (Up)": disp(1, 0)[0],
        "right +z (E)": disp(2, 0)[2],
        "left +y (Left)": disp(6, 1)[1],
        "left -z (Q+Shift)": -disp(7, 1)[2],
        "right still while left active": -np.linalg.norm(disp(6, 0)) + 0.01,
        "left still while right active": -np.linalg.norm(disp(1, 1)) + 0.01,
    }
    print(f"recorded {n_steps} steps -> {path}")
    ok = True
    for name, val in checks.items():
        good = val > 0.0
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {val * 100:+.1f} cm")

    with h5py.File(path, "r") as f:
        print("  file attrs: success=%s num_steps=%d" % (f.attrs["success"], f.attrs["num_steps"]))
        acts = f["actions"][:]
        g = f["obs"]
        print("  actions", acts.shape, "states", f["states"].shape,
              "rgb", g["rgbd_cam_rgb"].shape, g["rgbd_cam_rgb"].dtype,
              "depth", g["rgbd_cam_depth"].shape, "depth range [%.2f, %.2f] m" % (g["rgbd_cam_depth"][0].min(), g["rgbd_cam_depth"][0].max()))
        print("  obs keys:", ", ".join(sorted(g.keys())))
        assert f.attrs["success"], "success flag not set"
        assert acts.shape == (n_steps, 14)
        grip_r = acts[:, 6]
        print(f"  right gripper action: first {grip_r[0]:+.0f}, last {grip_r[-1]:+.0f}; active_robot last = {f['active_robot'][-1]}")
        ok &= grip_r[0] < 0 < grip_r[-1] and f["active_robot"][-1] == 1
        ok &= bool(np.abs(acts[:, :6]).max() <= 1.0)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    if not args.no_replay:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "replay_demo.py"), path], check=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
