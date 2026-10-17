"""Bimanual assistive (feeding / jar opening) robosuite environment.

Importing this package registers the env, so that
    suite.make("BimanualAssist", robots=["Panda", "Panda"], ...)
works.
"""
from .env import BimanualAssist, BimanualAssistEnv  # noqa: F401  (import registers the env)

