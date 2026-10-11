"""Record a short scripted motion to outputs/scripted_motion.mp4 (+ .gif):
right arm reaches over the fork handle, descends, closes the gripper and lifts;
left arm moves above the jar lid. Uses simple P-control on the OSC_POSE position deltas."""
import os

import imageio.v2 as imageio
import numpy as np
from _common import OUT, make_env

CAM, W, H = "overview", 640, 480
OSC_MAX = 0.05  # metres per unit action (default OSC_POSE output_max for position)


def arm_action(eef, target, grip, gain=2.0):
    """7-dim per-arm action: [dx, dy, dz, drx, dry, drz, gripper(-1 open, +1 close)]."""
    a = np.zeros(7)
    a[:3] = np.clip(gain * (target - eef) / OSC_MAX, -1, 1)
    a[6] = grip
    return a


def main():
    os.makedirs(OUT, exist_ok=True)
    env = make_env(has_offscreen_renderer=True)
    obs = env.reset()
    fork_handle = obs["fork_pos"].copy()
    lid_top = obs["jar_pos"] + np.array([0, 0, 0.051])
    up = np.array([0, 0, 1.0])
    # (n_steps, right-arm target, right gripper, left-arm target, left gripper)
    plan = [
        (40, fork_handle + 0.12 * up, -1, lid_top + 0.15 * up, -1),
        (30, fork_handle + 0.005 * up, -1, lid_top + 0.06 * up, -1),
        (15, fork_handle + 0.005 * up, 1, lid_top + 0.06 * up, -1),
        (40, fork_handle + 0.20 * up, 1, lid_top + 0.06 * up, -1),
    ]
    frames, offsets = [], []
    for n, tgt_r, g_r, tgt_l, g_l in plan:
        for _ in range(n):
            action = np.concatenate([arm_action(obs["robot0_eef_pos"], tgt_r, g_r),
                                     arm_action(obs["robot1_eef_pos"], tgt_l, g_l)])
            obs, _, _, _ = env.step(action)
            rgb, _ = env.sim.render(camera_name=CAM, width=W, height=H, depth=True)
            frames.append(rgb[::-1].copy())
        offsets.append(obs["fork_pos"] - obs["robot0_eef_pos"])  # fork pose relative to the gripper
    slip = np.linalg.norm(offsets[-1] - offsets[-2])  # change during the lift phase
    print(f"fork lifted by {100 * (obs['fork_pos'][2] - fork_handle[2]):.1f} cm; "
          f"slip in gripper during lift: {1000 * slip:.1f} mm; fork tip at {np.round(obs['fork_tip_pos'], 3)}")
    env.close()
    mp4 = os.path.join(OUT, "scripted_motion.mp4")
    imageio.mimsave(mp4, frames, fps=20)
    imageio.mimsave(os.path.join(OUT, "scripted_motion.gif"), frames[::2], duration=0.1, loop=0)
    imageio.imwrite(os.path.join(OUT, "scripted_motion_last.png"), frames[-1])
    print("saved", mp4, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
