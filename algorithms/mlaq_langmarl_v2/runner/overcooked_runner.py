"""MLAQ's joint-action search and rollout on Forced Coordination."""

import json
from pathlib import Path
import re

import openai

from environments.overcooked_ai.adapter import LAYOUT_NAME, OvercookedFirstSoup
from environments.overcooked_ai.controller import AGENTS, KitchenController
from prompting.lang_tree import LangTree, parse_joint_action
from tools.overcooked_replay import save_replay


GOAL = "First onion soup delivered"
RULES = (
    "Two cooks occupy separate sides of the Forced Coordination kitchen. LeftChef has "
    "onions and plates. RightChef has pots and the serving station. Pass items via a shared "
    "counter. First fill a single pot with three onions, let it cook, use a plate on "
    "the ready pot, then serve the soup. Grid positions are (x,y). "
    "USE x,y walks to that fixture and interacts once. WAIT waits. "
    "Both cooks start together, and a cook who finishes early waits for the other. "
    "Pot object state records [ingredient, count, cooking ticks]. A full pot cooks "
    "automatically and is ready at 20 cooking ticks."
)
COMMAND = re.compile(r"(?m)^\s*(WAIT|USE \d+,\d+)\s*$")


def state_key(state, grid_steps):
    record = state.to_dict()
    record["objects"].sort(key=lambda obj: obj["position"])
    record["grid_steps"] = grid_steps
    return json.dumps(record, sort_keys=True)


def query_model(prompt, model, temperature):
    chunks = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=temperature, stream=True, stream_options={"include_usage": True},
        request_timeout=45,
    )
    content = []
    usage = None
    finish_reason = None
    has_reasoning = False
    for chunk in chunks:
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk["choices"]:
            delta = choice["delta"]
            finish_reason = choice.get("finish_reason") or finish_reason
            has_reasoning = has_reasoning or bool(delta.get("reasoning_content"))
            if delta.get("content"):
                content.append(delta["content"])
    return "".join(content), usage, finish_reason, has_reasoning


class OvercookedRunner:
    def __init__(self, args, batch_dir, feedback, query=query_model):
        self.args = args
        self.kitchen = OvercookedFirstSoup(horizon=args.grid_horizon)
        self.controller = KitchenController(self.kitchen.mdp)
        self.feedback = feedback
        self.query = query
        config = batch_dir / "overcooked_config.json"
        settings = {"layout": LAYOUT_NAME, "grid_horizon": args.grid_horizon}
        if config.exists() and json.loads(config.read_text(encoding="utf-8")) != settings:
            raise ValueError("Kitchen layout and grid horizon must match the existing batch.")
        batch_dir.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        self.case_dir = batch_dir / f"run_{args.start_run_id}" / "case_0"
        self.case_dir.mkdir(parents=True, exist_ok=False)
        self.graph = LangTree(load=args.tree_load, if_temp_tree=True,
                              tree_path=str(batch_dir / "lang_tree.pth"), agent_names=AGENTS)
        self.node(GOAL).value = 1
        self.usage = {role: dict(requests=0, input_tokens=0, output_tokens=0,
                                 requests_without_usage=0, failed=0)
                      for role in ("planning", "evaluation")}
        self.fixtures = ", ".join(f"{x},{y} {name}" for (x, y), name in self.controller.fixtures.items())

    def node(self, key):
        node = self.graph.build_node(key)
        node.visit_num = max(node.visit_num, 1)
        return node

    def ask(self, prompt, evaluation=False):
        totals = self.usage["evaluation" if evaluation else "planning"]
        totals["requests"] += 1
        try:
            response, usage, finish_reason, has_reasoning = self.query(
                prompt, self.args.llm_source, self.args.temperature
            )
        except openai.error.OpenAIError as error:
            if not evaluation:
                raise
            self.feedback.log("evaluation_api_error", error_type=type(error).__name__)
            response, usage, finish_reason, has_reasoning = "", None, None, False
        failed = not response.strip()
        totals["failed"] += int(failed)
        if usage is None:
            totals["requests_without_usage"] += 1
        else:
            totals["input_tokens"] += usage.get("prompt_tokens", 0)
            totals["output_tokens"] += usage.get("completion_tokens", 0)
        self.feedback.record_usage(
            evaluation, 1, (usage or {}).get("prompt_tokens", 0),
            (usage or {}).get("completion_tokens", 0), False, failed,
        )
        if evaluation:
            details = (usage or {}).get("completion_tokens_details") or {}
            self.feedback.log("evaluation_stream", finish_reason=finish_reason,
                              has_reasoning=has_reasoning, content_chars=len(response),
                              reasoning_tokens=details.get("reasoning_tokens"))
        return response

    def propose(self, key, directory, known_actions):
        self.feedback.dialog_versions.clear()
        commands = []
        for agent in AGENTS:
            prompt = (
                f"{RULES}\nFixtures: {self.fixtures}\n"
                f"State: {key}\nYou control {agent}. "
                "Choose your next action toward delivering the first soup quickly. "
                "End your response with one command on its own line: WAIT or USE x,y.\n"
            )
            if known_actions:
                prompt += "Known joint actions. Try a different combination if useful:\n"
                prompt += "\n".join(action.strip() for action in known_actions) + "\n"
            if commands:
                prompt += f"RightChef has proposed {commands[0]}.\n"
            prompt += self.feedback.advice_for_prompt(GOAL, key, agent)
            query_dir = directory / f"{agent}_query" / "trial_1"
            query_dir.mkdir(parents=True, exist_ok=True)
            (query_dir / "main_prompt.txt").write_bytes(prompt.encode("utf-8"))
            response = self.ask(prompt)
            (query_dir / "main_response.txt").write_bytes(response.encode("utf-8"))
            matches = COMMAND.findall(response)
            if not matches:
                raise ValueError(f"{agent} did not return a kitchen command")
            commands.append(matches[-1].strip())
        return "EXECUTE\n" + "".join(f"NAME {agent} ACTION {action}\n"
                                    for agent, action in zip(AGENTS, commands))

    def execute(self, kitchen, action, replay_states=None):
        commands = parse_joint_action(action, AGENTS)
        if commands is None:
            raise ValueError("Expected a joint action with one command for each cook.")
        return self.controller.execute(kitchen, tuple(commands[agent] for agent in AGENTS), replay_states)

    def imagine(self, initial, directory):
        branch = self.kitchen.fork(initial, 0)
        directory.mkdir(parents=True, exist_ok=True)
        notes = []
        for timestep in range(1, self.args.tsteps + 1):
            key = state_key(branch.env.state, branch.env.t)
            node = self.node(key)
            self.graph.build_virtual_node(key)
            self.graph.compute_ucb_values()
            action, child = node.get_best(with_ucb=True)
            proposed = action == "action_virtual"
            if proposed:
                child.visit_num += 1
            try:
                if proposed:
                    known_actions = [edge["action"] for edge in node.children
                                     if edge["action"] != "action_virtual"]
                    action = self.propose(key, directory / f"timestep_{timestep}", known_actions)
                transition = self.execute(branch, action)
            except ValueError as error:
                notes.append(f"Step {timestep}: rejected candidate: {error}")
                break
            next_key = GOAL if transition.delivered else state_key(transition.state, transition.grid_steps)
            new_edge = node.get_child(action) is None
            self.node(next_key).visit_num += int(not new_edge)
            self.graph.update([key, action, next_key])
            if proposed:
                self.feedback.record_proposal(key, action, "accepted" if new_edge else "known")
                if new_edge:
                    self.feedback.record_new_edge(GOAL, key, action)
            source = "LLM" if proposed else "memory"
            notes.append(f"Step {timestep}: {source}, grid step {transition.grid_steps}\n{action.strip()}")
            if transition.done:
                break
        (directory / "log.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    def rollout(self, episode_id):
        state = self.kitchen.reset()
        replay_states = [state.deepcopy()]
        self.graph.compute_values(GOAL, loop=self.args.tsteps)
        steps = []
        lines = [f"Fixtures: {self.fixtures}", f"Initial state: {state_key(state, 0)}"]
        outcome = "decision_limit"
        for step in range(1, self.args.tsteps + 1):
            start_tick = self.kitchen.env.t
            key = state_key(state, start_tick)
            node = self.graph.get_node(key)
            known = [] if node is None else [
                edge for edge in node.children if edge["action"] != "action_virtual"
            ]
            if not known:
                outcome = "no_candidate"
                break
            action = max(known, key=lambda edge: edge["child"].value)["action"]
            transition = self.execute(self.kitchen, action, replay_states)
            state = transition.state
            next_key = state_key(state, transition.grid_steps)
            versions = self.feedback.versions_for_execution(GOAL, key, action)
            steps.append(dict(step=step, state=key, joint_action=action, next_state=next_key,
                              reward=transition.reward, done=transition.done, execution="complete",
                              feedback_versions=versions, grid_steps=transition.grid_steps - start_tick))
            lines.append(f"Step {step} (grid steps {start_tick + 1}–{transition.grid_steps}):\n"
                         f"{action.strip()}\nState: {next_key}\nReward: {transition.reward}")
            if transition.done:
                outcome = "goal_reached" if transition.delivered else "grid_step_limit"
                break
        lines.append(f"Outcome: {outcome}; {len(steps)} decisions, {self.kitchen.env.t} grid steps.")
        round_log = "\n".join(lines) + "\n"
        directory = self.case_dir / "interact" / f"round_{episode_id}"
        directory.mkdir(parents=True)
        (directory / "log.txt").write_text(round_log, encoding="utf-8")
        save_replay(directory / "replay_video.mp4", self.kitchen.mdp, replay_states, outcome)
        episode = dict(episode=episode_id, outcome=outcome, steps=steps)
        summary = dict(outcome=outcome, decisions=len(steps), grid_steps=self.kitchen.env.t,
                       reward=self.kitchen.env.cumulative_sparse_rewards)
        return episode, round_log, summary

    def run(self):
        summaries = []
        self.feedback.reset(self.case_dir)
        round_id = 0
        for episode_id in range(self.args.episodes):
            initial = self.kitchen.reset()
            for _ in range(self.args.imagination_rounds):
                round_id += 1
                directory = self.case_dir / "simulate" / f"round_{round_id}"
                self.imagine(initial, directory)
            episode, round_log, summary = self.rollout(episode_id)
            summaries.append(summary)
            self.graph.save()
            self.feedback.review(GOAL, episode, round_log,
                                 lambda prompt: self.ask(prompt, evaluation=True), directory)
        delivered = [item for item in summaries if item["outcome"] == "goal_reached"]
        result = dict(finished=True, episodes=summaries, usage=self.usage,
                      model=self.args.llm_source, imagination_rounds=self.args.imagination_rounds,
                      decision_limit=self.args.tsteps, grid_horizon=self.args.grid_horizon,
                      first_delivery_grid_steps=delivered[0]["grid_steps"] if delivered else None,
                      best_delivery_grid_steps=min((item["grid_steps"] for item in delivered), default=None))
        result["trajectory_feedback"] = self.feedback.summary()
        (self.case_dir / "case_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Overcooked results: {self.case_dir}")
        print(json.dumps(result, ensure_ascii=False))
        return result


def run_overcooked(args, data_dir, epoch, feedback):
    if args.start_case_id != 0 or args.num_cases < 1 or args.compare_q or args.comm_mode != "dialog":
        raise ValueError("Overcooked supports case_0 in dialog mode with MLAQ's original joint values.")
    if min(args.episodes, args.imagination_rounds, args.tsteps, args.grid_horizon) < 1:
        raise ValueError("Kitchen planning and rollout budgets must be positive.")
    batch = Path(data_dir) / "overcooked_dialog" / LAYOUT_NAME / epoch
    return OvercookedRunner(args, batch, feedback=feedback).run()
