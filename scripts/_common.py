"""Shared helpers for the scripts (path setup + env factory)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
sys.path.insert(0, ROOT)

import robosuite as suite  # noqa: E402

import assistive_sim  # noqa: E402,F401  (registers BimanualAssist)


def make_env(**kwargs):
    cfg = dict(robots=["Panda", "Panda"], env_configuration="parallel",
               has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
               control_freq=20, ignore_done=True)
    cfg.update(kwargs)
    return suite.make("BimanualAssist", **cfg)
