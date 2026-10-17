import numpy as np
import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

obs = env.reset()
for i in range(300):
    action = np.zeros(env.action_dim)
    action[2] = -0.3 if i < 100 else 0.3   # move down, then up
    obs, reward, done, info = env.step(action)
    env.render()

env.close()

print("Environment closed successfully.")