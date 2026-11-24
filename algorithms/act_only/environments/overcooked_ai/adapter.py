"""Run the fixed Forced Coordination layout for a full episode."""

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


LAYOUT_NAME = "forced_coordination"


class OvercookedKitchen:
    def __init__(self, horizon):
        # The pinned neurips2019 release calls this layout "random0".
        self.mdp = OvercookedGridworld.from_layout_name("random0")
        self.env = OvercookedEnv(self.mdp, horizon=horizon)

    def reset(self):
        self.env.reset()
        return self.env.state

    def step(self, joint_action):
        state, reward, done, info = self.env.step(joint_action)
        return state, reward, done, {
            **info,
            "delivered": reward > 0,
            "grid_steps": self.env.t,
        }
