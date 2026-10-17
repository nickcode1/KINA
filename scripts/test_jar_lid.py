"""Tiny lid test: apply a torque to the lid hinge (qfrc_applied) and check that the lid angle
increases and lid_open becomes True after LID_TURNS_TO_OPEN (=2) turns.
The jar is NOT held: the hinge reaction torque (friction + damping) must be resisted by the
jar-table contact, so the test also checks that the jar does not move.
Run:  .venv\\Scripts\\python scripts\\test_jar_lid.py   (or: pytest scripts/test_jar_lid.py)"""
import numpy as np
from _common import make_env

TAU = 0.1  # N m on the lid (joint frictionloss is 0.05 N m, damping 0.02 N m s/rad)


def test_lid_opens_after_two_turns(max_steps=400, verbose=True):
    env = make_env()
    env.reset()
    m, d = env.sim.model._model, env.sim.data._data
    dof = m.jnt_dofadr[m.joint(env.jar.lid_joint).id]
    assert env.LID_TURNS_TO_OPEN == 2.0 and not env.is_jar_open()

    # 1) no torque: lid must stay put (joint friction holds it)
    for _ in range(20):
        env.step(np.zeros(env.action_dim))
    assert abs(env.lid_angle) < 1e-3, env.lid_angle

    # 2) apply torque until open
    jar_pos0 = env.sim.data.body_xpos[env.jar_body_id].copy()
    jar_yaw0 = env.sim.data.body_xquat[env.jar_body_id].copy()
    angles, t_open, opened_at = [], None, None
    for k in range(max_steps):
        d.qfrc_applied[dof] = TAU
        obs, _, _, _ = env.step(np.zeros(env.action_dim))
        angles.append(env.lid_angle)
        if env.is_jar_open() and t_open is None:
            t_open, opened_at = (k + 1) / env.control_freq, env.lid_angle
            break
    d.qfrc_applied[dof] = 0.0
    angles = np.array(angles)
    assert np.all(np.diff(angles) > 0), "lid angle must increase monotonically"
    assert t_open is not None and env.lid_open and bool(obs["lid_open"][0] if np.ndim(obs["lid_open"]) else obs["lid_open"])
    assert angles[-2] < 2 * np.pi * 2 <= angles[-1]
    jar_rot = np.abs(env.sim.data.body_xquat[env.jar_body_id] - jar_yaw0).max()
    jar_shift = np.linalg.norm(env.sim.data.body_xpos[env.jar_body_id] - jar_pos0)
    assert jar_shift < 2e-3 and jar_rot < 1e-2, (jar_shift, jar_rot)
    if verbose:
        for t in (0.5, 1.0, 2.0, 3.0, 4.0):
            i = int(t * env.control_freq) - 1
            if i < len(angles):
                print(f"  t={t:3.1f}s  lid_angle={angles[i]:6.2f} rad ({angles[i] / (2 * np.pi):4.2f} turns)  "
                      f"open={abs(angles[i]) >= 4 * np.pi}")
        print(f"lid_open -> True after {t_open:.2f} s at {opened_at:.2f} rad "
              f"({opened_at / (2 * np.pi):.2f} turns); jar moved {1000 * jar_shift:.2f} mm, "
              f"jar quat change {jar_rot:.1e}")

    # 3) configurable threshold
    env.LID_TURNS_TO_OPEN = 3.0
    assert not env.is_jar_open()
    env.close()


if __name__ == "__main__":
    test_lid_opens_after_two_turns()
    print("PASS")
