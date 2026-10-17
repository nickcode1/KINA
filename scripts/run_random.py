"""Headless smoke test: reset, hold still for 2.5 s (stability check), then small random actions.
Prints observation keys and the real-time factor (sim time / wall time)."""
import time

import numpy as np
from _common import make_env


def main(n_still=50, n_random=150, seed=0):
    env = make_env()
    obs = env.reset()
    print("action_dim:", env.action_dim, "| timestep:", env.sim.model.opt.timestep,
          "| control_freq:", env.control_freq)
    print("observation keys:")
    for k, v in obs.items():
        print(f"  {k:28s} {np.shape(v)}")

    obj_keys = ["fork_pos", "plate_pos", "jar_pos", "food0_pos", "food1_pos", "food2_pos"]
    start = {k: obs[k].copy() for k in obj_keys}
    rng = np.random.default_rng(seed)
    t0, sim0 = time.perf_counter(), env.sim.data.time
    for i in range(n_still + n_random):
        if i < n_still:
            action = np.zeros(env.action_dim)
        else:
            action = rng.uniform(-0.2, 0.2, env.action_dim)
        obs, _, _, _ = env.step(action)
        if i == n_still - 1:
            print(f"\nobject drift after {env.sim.data.time - sim0:.2f} s with zero action:")
            for k in obj_keys:
                print(f"  {k:12s} {1000 * np.linalg.norm(obs[k] - start[k]):7.2f} mm")
            print(f"  lid_angle    {float(np.ravel(obs['lid_angle'])[0]):.4f} rad")
    wall = time.perf_counter() - t0
    sim_t = env.sim.data.time - sim0
    print(f"\nsteps: {n_still + n_random}, sim time: {sim_t:.2f} s, wall time: {wall:.2f} s, "
          f"real-time factor: {sim_t / wall:.2f}x")
    print("fork tip:", np.round(obs["fork_tip_pos"], 3), "| mouth:", np.round(obs["mouth_pos"], 3),
          "| lid angle:", np.round(obs["lid_angle"], 3))
    env.close()


if __name__ == "__main__":
    main()
