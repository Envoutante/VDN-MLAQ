"""Run the classic Forced Coordination kitchen."""

from copy import copy

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


LAYOUT_NAME = "forced_coordination"


class OvercookedFirstSoup:
    def __init__(self, horizon=400, stop_on_delivery=True):
        # The neurips2019 release calls Forced Coordination "random0".
        self.mdp = OvercookedGridworld.from_layout_name("random0")
        self.env = OvercookedEnv(self.mdp, horizon=horizon)
        self.stop_on_delivery = stop_on_delivery
        self.delivered = False

    def reset(self):
        self.env.reset()
        self.delivered = False
        return self.env.state

    def fork(self, state, grid_steps):
        """Start a search branch without changing the rollout's simulator."""
        branch = copy(self)
        branch.env = copy(self.env)
        branch.env.state = state.deepcopy()
        branch.env.t = grid_steps
        branch.env.cumulative_sparse_rewards = 0
        branch.env.cumulative_shaped_rewards = 0
        branch.delivered = False
        return branch

    def step(self, joint_action):
        if (self.stop_on_delivery and self.delivered) or self.env.is_done():
            raise RuntimeError("Reset the kitchen before starting another episode.")

        state, sparse_reward, done, info = self.env.step(joint_action)
        # The pinned simulator awards sparse reward only on delivery.
        self.delivered = sparse_reward > 0
        return state, sparse_reward, done or (self.stop_on_delivery and self.delivered), {
            **info,
            "delivered": self.delivered,
            "grid_steps": self.env.t,
        }
