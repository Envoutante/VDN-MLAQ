"""Execute concurrent fixture interactions using the classic grid simulator."""

from collections import deque
from dataclasses import dataclass
import re

from overcooked_ai_py.mdp.actions import Action, Direction


AGENTS = ("RightChef", "LeftChef")
FIXTURES = {"O": "onion dispenser", "D": "plate dispenser", "P": "pot",
            "S": "serving station", "X": "counter"}


@dataclass
class Transition:
    state: object
    grid_steps: int
    reward: float
    done: bool
    delivered: bool


class KitchenController:
    def __init__(self, mdp):
        self.floor = set(mdp.get_valid_player_positions())
        self.fixtures = {
            (x, y): FIXTURES[tile]
            for y, row in enumerate(mdp.terrain_mtx)
            for x, tile in enumerate(row)
            if tile in FIXTURES
        }

    def commands(self, player, command):
        if command == "WAIT":
            return [Action.STAY]
        match = re.fullmatch(r"USE (\d+),(\d+)", command)
        if match is None:
            raise ValueError(f"Invalid kitchen action {command!r}")
        target = tuple(map(int, match.groups()))
        if target not in self.fixtures:
            raise ValueError(f"No fixture at {target}")

        # Forced Coordination keeps the cooks in separate floor regions.
        start = player.position, player.orientation
        pending = deque([(start, [])])
        visited = {start}
        while pending:
            (position, facing), moves = pending.popleft()
            if Action.move_in_direction(position, facing) == target:
                return moves + [Action.INTERACT]
            for direction in Direction.ALL_DIRECTIONS:
                candidate = Action.move_in_direction(position, direction)
                next_position = candidate if candidate in self.floor else position
                pose = next_position, direction
                if pose not in visited:
                    visited.add(pose)
                    pending.append((pose, moves + [direction]))
        raise ValueError(f"Fixture {target} is out of reach")

    def reachable_fixtures(self, player):
        reachable = []
        for x, y in self.fixtures:
            try:
                self.commands(player, f"USE {x},{y}")
            except ValueError:
                continue
            reachable.append(f"{x},{y} {self.fixtures[x, y]}")
        return ", ".join(reachable)

    def plan(self, state, joint_action):
        plans = []
        for agent, player, command in zip(AGENTS, state.players, joint_action):
            try:
                plans.append(self.commands(player, command))
            except ValueError as error:
                raise ValueError(f"{agent} cannot execute {command}: {error}") from error
        return plans

    def execute(self, kitchen, joint_action, replay_states=None):
        plans = self.plan(kitchen.env.state, joint_action)
        return self.execute_plan(kitchen, plans, replay_states)

    def execute_plan(self, kitchen, plans, replay_states=None):
        for tick in range(max(map(len, plans))):
            actions = tuple(plan[tick] if tick < len(plan) else Action.STAY for plan in plans)
            state, reward, done, info = kitchen.step(actions)
            if replay_states is not None:
                replay_states.append(state.deepcopy())
            # A delivery ends this joint command. In a full episode, the next
            # command starts from the delivered state without resetting the kitchen.
            if done or info["delivered"]:
                break
        return Transition(state, info["grid_steps"], reward, done, info["delivered"])
