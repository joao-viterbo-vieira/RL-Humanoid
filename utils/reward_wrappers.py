import gymnasium as gym
import numpy as np

class EffortAndLateralWrapper(gym.Wrapper):
    """
    Small shaping to stabilize gait:
    - action L2 penalty (discourage frantic torques)
    - lateral |y| penalty (world-frame drift from the corridor centerline)

    Note: earlier versions also defined an upright bonus, but it called the
    mujoco-py-only API `model.body_name2id` inside a silent try/except and
    never executed under the mujoco>=2.2 bindings used for every logged run.
    It was removed as dead code; do not re-add it when reproducing published
    results.
    """
    def __init__(self, env, effort_w=0.001, lateral_w=1.0):
        super().__init__(env)
        self.effort_w = float(effort_w)
        self.lateral_w = float(lateral_w)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Penalize effort
        reward -= self.effort_w * float(np.sum(np.square(action)))

        # Penalize lateral deviation (y-axis)
        # MuJoCo: data.qpos[0:3] = [x, y, z] of root body
        y_pos = float(self.env.unwrapped.data.qpos[1])
        reward -= self.lateral_w * abs(y_pos)
        return obs, reward, terminated, truncated, info
