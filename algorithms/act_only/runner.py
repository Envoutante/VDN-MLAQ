"""Choose each joint action directly from the current Overcooked state."""

from itertools import count
import json
import re
from textwrap import indent

from algorithms.act_only.environments.overcooked_ai.adapter import OvercookedKitchen
from algorithms.act_only.environments.overcooked_ai.controller import AGENTS, KitchenController
from tools.overcooked_replay import save_replay


TASK_RULES = (
    "Two cooks occupy separate sides of the Forced Coordination kitchen. LeftChef has "
    "onions and plates. RightChef has pots and the serving station. Pass items via a shared "
    "counter. The team aims to deliver as many soups as possible before the episode ends. "
    "Each soup needs three onions in one pot. Use a plate on a ready pot, then serve the soup. "
    "Both pots can be used. While one pot cooks, LeftChef can bring onions or stage a plate on a "
    "shared counter, and RightChef can fetch ingredients and fill the other pot. Coordinate prep "
    "with the next handoff and serve ready soup in time. "
    "Grid positions are (x,y). "
    "USE x,y walks to that fixture and interacts once. WAIT waits. "
    "Both cooks start together, and a cook who finishes early waits for the other. "
    "Pot object state records [ingredient, count, cooking ticks]. A full pot cooks "
    "automatically and is ready at 20 cooking ticks."
)
COMMAND = re.compile(r"WAIT|USE [0-9]+,[0-9]+")


def state_record(state, grid_steps):
    record = state.to_dict()
    record["objects"].sort(key=lambda obj: obj["position"])
    record["grid_steps"] = grid_steps
    return record


class ActOnlyRunner:
    def __init__(self, args, case_dir, query):
        self.args = args
        self.case_dir = case_dir
        self.query = query
        self.kitchen = OvercookedKitchen(args.grid_horizon)
        self.controller = KitchenController(self.kitchen.mdp)
        self.fixtures = ", ".join(f"{x},{y} {name}" for (x, y), name in self.controller.fixtures.items())
        self.usage = dict(requests=0, input_tokens=0, output_tokens=0,
                          requests_without_usage=0, failed=0)

    def propose(self, player, agent, observation, remaining, directory, teammate_action):
        prompt = (
            f"{TASK_RULES}\nFixtures: {self.fixtures}\n"
            f"Remaining episode grid steps: {remaining} of {self.args.grid_horizon}.\n"
            f"State: {observation}\nYou control {agent}.\n"
            f"Fixtures you can reach: {self.controller.reachable_fixtures(player)}.\n"
            "Choose a reachable fixture or WAIT. Choose an action that helps the team deliver "
            "more soups. Consider preparing a later soup while a pot cooks. "
            "Output exactly one command on its own line: WAIT or USE x,y.\n"
        )
        if teammate_action is not None:
            prompt += f"RightChef has proposed {teammate_action}.\n"

        directory = directory / f"{agent}_query" / "trial_1"
        directory.mkdir(parents=True)
        (directory / "main_prompt.txt").write_bytes(prompt.encode("utf-8"))
        self.usage["requests"] += 1
        response, usage = self.query(prompt, self.args.llm_source, self.args.temperature)
        (directory / "main_response.txt").write_bytes(response.encode("utf-8"))
        if usage is None:
            self.usage["requests_without_usage"] += 1
        else:
            self.usage["input_tokens"] += usage.get("prompt_tokens", 0)
            self.usage["output_tokens"] += usage.get("completion_tokens", 0)
        self.usage["failed"] += int(not response.strip())
        return response.strip()

    def run_episode(self, episode_id):
        state = self.kitchen.reset()
        replay_states = [state.deepcopy()]
        directory = self.case_dir / "interact" / f"round_{episode_id}"
        directory.mkdir(parents=True)
        log_path = directory / "log.txt"
        invalid_actions = dict.fromkeys(AGENTS, 0)
        delivery_ticks = []
        horizon = self.args.grid_horizon

        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            log.write(f"[Start execution | Grid step 0 of {horizon}]\n"
                      f"[Initial state | Players: {', '.join(AGENTS)}]\n"
                      f"{json.dumps(state_record(state, 0), indent=2)}\n")
            for step in count(1):
                start_tick = self.kitchen.env.t
                observation = json.dumps(state_record(state, start_tick), sort_keys=True)
                step_dir = directory / f"timestep_{step}"
                commands = []
                plans = []
                rejections = []
                for agent, player in zip(AGENTS, state.players):
                    teammate_action = commands[0] if commands else None
                    command = self.propose(player, agent, observation, horizon - start_tick,
                                           step_dir, teammate_action)
                    try:
                        if COMMAND.fullmatch(command) is None:
                            raise ValueError("Expected one WAIT or USE x,y command.")
                        plan = self.controller.commands(player, command)
                    except ValueError as error:
                        invalid_actions[agent] += 1
                        rejections.append(f"[Invalid action | {agent}] {error} Using WAIT.")
                        command = "WAIT"
                        plan = self.controller.commands(player, command)
                    commands.append(command)
                    plans.append(plan)

                transition = self.controller.execute_plan(self.kitchen, plans, replay_states)
                state = transition.state
                if transition.delivered:
                    delivery_ticks.append(transition.grid_steps)
                end_tick = transition.grid_steps
                grid_range = (f"Grid step {end_tick}" if end_tick == start_tick + 1
                              else f"Grid steps {start_tick + 1}–{end_tick}")
                action = "EXECUTE\n" + "\n".join(
                    f"NAME {agent} ACTION {command}" for agent, command in zip(AGENTS, commands)
                )
                log.write("\n".join([
                    f"\n========== Step {step} | {grid_range} ==========",
                    *rejections,
                    "[Executed action]",
                    indent(action, "  "),
                    f"[Reward] {transition.reward:+g} | Total {self.kitchen.env.cumulative_sparse_rewards:g}",
                    f"[Delivered] {str(transition.delivered).lower()}",
                ]) + "\n")
                if transition.done:
                    break

            reward = self.kitchen.env.cumulative_sparse_rewards
            log.write(f"\n[End execution | horizon_reached | Grid step {end_tick} of {horizon}]\n"
                      f"[Summary] Total reward {reward:g} | Soups delivered {len(delivery_ticks)} | "
                      f"Joint actions {step} | Invalid actions {sum(invalid_actions.values())}\n")

        save_replay(directory / "replay_video.mp4", self.kitchen.mdp, replay_states, "horizon_reached")
        return dict(outcome="horizon_reached", decisions=step, grid_steps=end_tick,
                    reward=reward, soups_delivered=len(delivery_ticks),
                    delivery_grid_steps=delivery_ticks, invalid_actions=invalid_actions)

    def run(self):
        self.case_dir.mkdir(parents=True)
        print(f"Act-only results: {self.case_dir}")
        summaries = [self.run_episode(episode_id) for episode_id in range(1, self.args.episodes + 1)]
        result = dict(finished=True, episodes=summaries, model=self.args.llm_source,
                      temperature=self.args.temperature, grid_horizon=self.args.grid_horizon,
                      full_horizon=True, critic_feedback=False, usage={"acting": self.usage})
        report = json.dumps(result, ensure_ascii=False, indent=2)
        (self.case_dir / "case_result.json").write_text(report + "\n", encoding="utf-8")
        print(report)
        return result
