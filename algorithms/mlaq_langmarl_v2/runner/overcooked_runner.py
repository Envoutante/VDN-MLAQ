"""MLAQ's joint-action search and rollout on Forced Coordination."""

from copy import deepcopy
import json
from itertools import count
from math import log, sqrt
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from textwrap import fill, indent

import openai
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
import torch

from environments.overcooked_ai.adapter import LAYOUT_NAME, OvercookedFirstSoup
from environments.overcooked_ai.controller import AGENTS, KitchenController
from prompting.lang_tree import LangTree, Nodes, parse_joint_action, preprocess_state
from tools.overcooked_replay import save_replay


FULL_HORIZON_GOAL = "Deliver as many onion soups as possible"
MAX_FEEDBACK_ATTEMPTS = 2
MODEL_TRIALS = 3
KITCHEN_LAYOUT = (
    "Two cooks occupy separate sides of the Forced Coordination kitchen. LeftChef has "
    "onions and plates. RightChef has pots and the serving station. Pass items via a shared "
    "counter. "
)
KITCHEN_ACTIONS = (
    "Grid positions are (x,y). "
    "USE x,y walks to that fixture and interacts once. WAIT waits. "
    "Both cooks start together, and a cook who finishes early waits for the other. "
    "Pot object state records [ingredient, count, cooking ticks]. A full pot cooks "
    "automatically and is ready at 20 cooking ticks."
)
FULL_HORIZON_RULES = (
    KITCHEN_LAYOUT + "The team aims to deliver as many soups as possible before the episode ends. "
    "Each soup needs three onions in one pot. Use a plate on a ready pot, then serve the soup. "
    "Both pots can be used. While one pot cooks, LeftChef can bring onions or stage a plate on a "
    "shared counter, and RightChef can fetch ingredients and fill the other pot. Coordinate prep "
    "with the next handoff and serve ready soup in time. " + KITCHEN_ACTIONS
)
COMMAND = re.compile(r"(?m)^\s*(WAIT|USE \d+,\d+)\s*$")
VERDICT = re.compile(r"(?mi)^[ \t]*\[(VALID|ERROR)\][ \t]*([^\r\n]*)\r?$")
ORIENTATION_NAMES = {(0, -1): "north", (1, 0): "east", (0, 1): "south", (-1, 0): "west"}


def append_round_log(log_path, message):
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(message + "\n")


def state_key(state, grid_steps):
    record = state.to_dict()
    record["objects"].sort(key=lambda obj: obj["position"])
    record["grid_steps"] = grid_steps
    return json.dumps(record, sort_keys=True)


def describe_item(item):
    if item is None:
        return "empty"
    if item["name"] == "soup":
        ingredient, count, ticks = item["state"]
        return f"soup ({count} {ingredient}, {ticks} cooking ticks)"
    return item["name"]


def describe_state_changes(before, after, delivered):
    lines = ["[State changes]"]
    for agent, previous, current in zip(AGENTS, before["players"], after["players"]):
        changes = []
        if previous["position"] != current["position"]:
            changes.append(f"position {previous['position']} → {current['position']}")
        if previous["orientation"] != current["orientation"]:
            previous_direction = ORIENTATION_NAMES.get(tuple(previous["orientation"]), previous["orientation"])
            current_direction = ORIENTATION_NAMES.get(tuple(current["orientation"]), current["orientation"])
            changes.append(f"facing {previous_direction} → {current_direction}")
        previous_item = describe_item(previous["held_object"])
        current_item = describe_item(current["held_object"])
        if previous_item != current_item:
            previous_hand = "empty-handed" if previous["held_object"] is None else f"holding {previous_item}"
            current_hand = "empty-handed" if current["held_object"] is None else f"holding {current_item}"
            changes.append(f"{previous_hand} → {current_hand}")
        if changes:
            lines.append(f"  {agent}: {', '.join(changes)}")
    previous_objects = {tuple(obj["position"]): obj for obj in before["objects"]}
    current_objects = {tuple(obj["position"]): obj for obj in after["objects"]}
    for position in sorted(previous_objects.keys() | current_objects.keys()):
        previous_item = describe_item(previous_objects.get(position))
        current_item = describe_item(current_objects.get(position))
        if previous_item != current_item:
            lines.append(f"  At {position}: {previous_item} → {current_item}")
    lines.append(f"  Delivered: {json.dumps(delivered)}")
    return "\n".join(lines)


def normalize_predicted_item(item, label):
    name = item["name"]
    value = item["state"]
    if name in ("onion", "tomato", "dish"):
        if value == []:
            item["state"] = None
        elif value is not None:
            raise ValueError(f'{label}.state must be null for "{name}".')
    elif name == "soup":
        if (not isinstance(value, list) or len(value) != 3
                or value[0] not in ("onion", "tomato")
                or type(value[1]) is not int or not 1 <= value[1] <= 3
                or type(value[2]) is not int or not 0 <= value[2] <= 20):
            raise ValueError(f'{label}.state for "soup" must be [ingredient, count (1–3), cooking ticks (0–20)].')
    else:
        raise ValueError(f'{label}.name "{name}" is not an item. Use onion, tomato, dish or soup.')


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


class KitchenLangTree(LangTree):
    """Keep full-horizon paths flat when copying or saving the search graph."""

    def __getstate__(self):
        graph = vars(self).copy()
        graph["nodes"] = [
            {**vars(node), "children": [
                {**edge, "child": edge["child"].state} for edge in node.children
            ]}
            for node in self.nodes
        ]
        return graph

    def __setstate__(self, graph):
        records = graph["nodes"]
        nodes = {record["state"]: Nodes(record["state"]) for record in records}
        for record in records:
            node = nodes[record["state"]]
            vars(node).update(record)
            node.children = [{**edge, "child": nodes[edge["child"]]} for edge in record["children"]]
        vars(self).update(graph)
        self.nodes = list(nodes.values())

    def save(self):
        destination = Path(self.tree_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=destination.parent, suffix=".pth", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save(self, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


class OvercookedRunner:
    def __init__(self, args, batch_dir, feedback, query=query_model):
        self.args = args
        self.feedback_goal = FULL_HORIZON_GOAL
        self.kitchen = OvercookedFirstSoup(horizon=args.grid_horizon, stop_on_delivery=False)
        self.round_id = 0
        self.controller = KitchenController(self.kitchen.mdp)
        self.feedback = feedback
        self.feedback_attempts = {}
        self.feedback_search = dict(imagination_expansions=0, rollout_expansions=0,
                                    rollout_searches=0, rollout_planning_requests=0,
                                    new_candidates=0, duplicate_candidates=0,
                                    rejected_candidates=0)
        self.query = query
        config = batch_dir / "overcooked_config.json"
        settings = {"layout": LAYOUT_NAME, "grid_horizon": args.grid_horizon,
                    "full_horizon": True, "transition_model": "llm_episode_horizon"}
        if config.exists() and json.loads(config.read_text(encoding="utf-8")) != settings:
            raise ValueError("Kitchen configuration differs from the existing batch. Start a new batch.")
        tree_path = batch_dir / "lang_tree.pth"
        if args.tree_load and tree_path.exists() and not config.exists():
            raise ValueError("Cannot load a kitchen graph without its matching configuration.")
        batch_dir.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        self.case_dir = batch_dir / f"run_{args.start_run_id}" / "case_0"
        self.case_dir.mkdir(parents=True, exist_ok=False)
        self.memory = KitchenLangTree(load=args.tree_load, tree_path=str(tree_path), agent_names=AGENTS)
        self.reset_search_graph()
        self.usage = {role: dict(requests=0, input_tokens=0, output_tokens=0,
                                 requests_without_usage=0, failed=0)
                      for role in ("planning", "evaluation")}
        self.fixtures = ", ".join(f"{x},{y} {name}" for (x, y), name in self.controller.fixtures.items())

    def reset_search_graph(self):
        self.graph = deepcopy(self.memory)
        self.graph.if_temp_tree = True
        for node in self.graph.nodes:
            node.action_exhausted = False
            node.value = 0
            node.ucb_value = 0
            node.visit_num = 1

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

    def query_trial(self, directory, prompt, property_name="main"):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{property_name}_prompt.txt").write_bytes(prompt.encode("utf-8"))
        response = self.ask(prompt)
        (directory / f"{property_name}_response.txt").write_bytes(response.encode("utf-8"))
        return response

    def propose(self, key, state, grid_step, directory, known_actions, trial, feedbacks):
        self.feedback.dialog_versions.clear()
        commands = []
        remaining = self.args.grid_horizon - grid_step
        for agent, player in zip(AGENTS, state.players):
            reachable = self.controller.reachable_fixtures(player)
            prompt = (
                f"{FULL_HORIZON_RULES}\nFixtures: {self.fixtures}\n"
                f"Remaining episode grid steps: {remaining} of {self.args.grid_horizon}.\n"
                f"State: {key}\nYou control {agent}.\n"
                f"Fixtures you can reach: {reachable}.\n"
                "Choose a reachable fixture or WAIT. Choose an action that helps the team deliver "
                "more soups. Consider preparing a later soup while a pot cooks. "
                "End your response with one command on its own line: WAIT or USE x,y.\n"
            )
            if known_actions:
                prompt += "Known joint actions. Try a different combination if useful:\n"
                prompt += "\n".join(action.strip() for action in known_actions) + "\n"
            if commands:
                prompt += f"RightChef has proposed {commands[0]}.\n"
            if feedbacks:
                prompt += "Previous proposals at this state were rejected:\n" + "\n".join(feedbacks) + "\n"
            prompt += self.feedback.advice_for_prompt(self.feedback_goal, key, agent)
            response = self.query_trial(directory / f"{agent}_query" / f"trial_{trial}", prompt)
            matches = COMMAND.findall(response)
            if not matches:
                raise ValueError(f"{agent} did not return a kitchen command")
            commands.append(matches[-1].strip())
        return "EXECUTE\n" + "".join(f"NAME {agent} ACTION {action}\n"
                                    for agent, action in zip(AGENTS, commands))

    def plan_action(self, state, action):
        commands = parse_joint_action(action, AGENTS)
        if commands is None:
            raise ValueError("Expected a joint action with one command for each cook.")
        return self.controller.plan(state, tuple(commands[agent] for agent in AGENTS))

    def execute(self, kitchen, action, replay_states=None):
        plans = self.plan_action(kitchen.env.state, action)
        return self.controller.execute_plan(kitchen, plans, replay_states)

    def check_with_llm(self, directory, context, question):
        analysis = self.query_trial(directory, f"{context}\n{question}")
        verdict = self.query_trial(
            directory, f"Question: {question}\nChecker analysis:\n{analysis}\n"
            "Summarize the check in one line. Output [VALID] if it passes, "
            "or [ERROR] followed by the specific reason if it fails.",
            property_name="feedback",
        )
        matches = VERDICT.findall(verdict)
        if len(matches) != 1:
            return "The checker did not return a clear verdict."
        label, reason = matches[0]
        return None if label.upper() == "VALID" else reason.strip() or "The checker rejected it."

    def predict_action(self, key, state, grid_step, directory, known_actions):
        feedbacks = []
        log_path = directory.parent / "log.txt"
        for trial in range(1, MODEL_TRIALS + 1):
            attempt = f"attempt {trial}/{MODEL_TRIALS}"
            append_round_log(log_path, f"\n[Action proposal | {attempt}]")
            action = None
            checker_dir = directory / "action_checker_query" / f"trial_{trial}"
            try:
                action = self.propose(key, state, grid_step, directory, known_actions, trial, feedbacks)
                append_round_log(log_path, indent(action.strip(), "  "))
                append_round_log(log_path, f"[Program reachability check | {attempt}]")
                plans = self.plan_action(state, action)
                append_round_log(log_path, "  PASSED")
                if action not in known_actions or getattr(self.args, "skip_check_memory", False):
                    append_round_log(log_path, f"[LLM action check | {attempt}]")
                    error = self.check_with_llm(
                        checker_dir,
                        f"{FULL_HORIZON_RULES}\nFixtures: {self.fixtures}\n"
                        f"State: {key}\nJoint action:\n{action}",
                        "Verify the action's reachability, required held items and interaction timing. "
                        "Account for travel and concurrent actions. Waiting and movement can be valid.",
                    )
                    if error:
                        raise ValueError(error)
                    append_round_log(log_path, "  PASSED")
                else:
                    append_round_log(log_path, "[Action check] Reusing previous check from memory.")
            except ValueError as error:
                checker_dir.mkdir(parents=True, exist_ok=True)
                (checker_dir / "rejection.txt").write_text(str(error), encoding="utf-8")
                append_round_log(log_path, indent(f"REJECTED: {error}", "  "))
                feedbacks.append(f"{action.strip() if action else 'Malformed proposal'}\nReason: {error}")
                if action is not None:
                    self.feedback.record_proposal(key, action, "action_rejected")
                continue
            return action, plans
        raise ValueError(f"Action checking failed after {MODEL_TRIALS} attempts. {feedbacks[-1]}")

    def predict_next_state(self, key, action, plans, grid_step, directory):
        schedule = "\n".join(
            f"{agent}: " + " ".join(Action.to_char(primitive) for primitive in plan)
            for agent, plan in zip(AGENTS, plans)
        )
        remaining = self.args.grid_horizon - grid_step
        current_tick = json.loads(key)["grid_steps"]
        duration = min(remaining, max(map(len, plans)))
        context = (
            f"{FULL_HORIZON_RULES}\nFixtures: {self.fixtures}\nState: {key}\n"
            f"Joint action:\n{action}\nPrimitive plans (arrows move/turn):\n{schedule}\n"
            "The plans run concurrently and the shorter one is padded with waits. "
            "Same-tick interactions resolve RightChef first. Interactions happen before the cooking tick. "
            "A full pot gains one cooking tick, including the tick the third onion is added. "
            "Plating removes soup from the pot. The joint action ends on delivery or at the grid horizon. "
            "After a delivery the episode continues from the state after serving. "
            f"Absolute grid steps remaining: {self.args.grid_horizon - grid_step}. "
            "The JSON grid_steps field is the absolute episode clock. Advance it by the elapsed grid steps."
        )
        feedback = ""
        log_path = directory.parent / "log.txt"
        for trial in range(1, MODEL_TRIALS + 1):
            attempt = f"attempt {trial}/{MODEL_TRIALS}"
            append_round_log(log_path, f"\n[Next-state prediction | {attempt}]")
            predictor_dir = directory / "world_model_query" / f"trial_{trial}"
            response = self.query_trial(
                predictor_dir, f"{context}\nPredict the next state. Return one JSON object with "
                "the same fields and coordinate format as the current state plus a boolean "
                "delivered that is true only if this joint action served a soup. "
                'Static fixtures stay on the map. A nonempty pot is represented by an object '
                'with name "soup" at the pot position, even when partly filled. Its state is '
                '["onion", count, cooking_ticks]. Empty pots have no entry in objects. '
                'Each held_object is either null or an object with name, position and state. '
                'Its position matches the player, and held items are excluded from objects. '
                'For onions, tomatoes and dishes, state is null. For example, an onion at (1,2) is '
                '{"name":"onion","position":[1,2],"state":null}.\n'
                f"{feedback}",
            )
            try:
                raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", response, flags=re.I)
                predicted = json.loads(raw)
                delivered = predicted.pop("delivered")
                tick = predicted.pop("grid_steps")
                if type(delivered) is not bool or type(tick) is not int:
                    raise ValueError("Use a boolean delivered and an integer grid_steps.")
                elapsed = tick - current_tick
                if not 1 <= elapsed <= duration or (not delivered and elapsed != duration):
                    raise ValueError("The elapsed grid steps do not match the joint action duration.")
                for agent, player in zip(AGENTS, predicted["players"]):
                    if tuple(player["position"]) not in self.controller.floor:
                        raise ValueError(f"{agent}.position must be a walkable tile.")
                    held = player["held_object"]
                    if held is not None and not isinstance(held, dict):
                        raise ValueError(f"{agent}.held_object must be null or an object with name, position and state.")
                    if held is not None:
                        normalize_predicted_item(held, f"{agent}.held_object")
                        if held["position"] != player["position"]:
                            raise ValueError(f"{agent}.held_object.position must match {agent}.position.")
                occupied = set()
                for index, obj in enumerate(predicted["objects"]):
                    position = tuple(obj["position"])
                    if position not in self.controller.fixtures or position in occupied:
                        raise ValueError(f"objects[{index}].position must be an unoccupied fixture position.")
                    occupied.add(position)
                    if obj["name"] == "pot":
                        if self.controller.fixtures[position] != "pot" or not obj["state"]:
                            raise ValueError("A pot is a fixed fixture. Record only its contents as a soup object.")
                        # Some replies name the contents after the fixture. An occupied pot has only one object.
                        obj["name"] = "soup"
                    normalize_predicted_item(obj, f"objects[{index}]")
                state = OvercookedState.from_dict(predicted)
                if len(state.players) != len(AGENTS) or state.order_list is not None:
                    raise ValueError("Keep the two cooks and the layout's unlimited order list.")
                # Check the representation and map constraints without advancing the environment.
                self.kitchen.mdp._check_valid_state(state)
                prediction = state_key(state, tick)
            except (ValueError, TypeError, KeyError, AttributeError, AssertionError, IndexError) as error:
                reason = str(error) or "The prediction violates the kitchen state constraints."
                append_round_log(log_path, f"[Program validation | {attempt}]")
            else:
                append_round_log(log_path, f"[LLM prediction check | {attempt}]")
                reason = self.check_with_llm(
                    directory / "prediction_checker_query" / f"trial_{trial}",
                    f"{context}\nPredicted next state: {prediction}\n"
                    f"Predicted delivery: {json.dumps(delivered)}",
                    "Check the predicted movements, handoffs, pot timers, elapsed grid steps, "
                    "and whether this action actually serves a soup.",
                )
                if reason is None:
                    append_round_log(log_path, "  PASSED")
                    return prediction, delivered
            (predictor_dir / "rejection.txt").write_text(reason, encoding="utf-8")
            append_round_log(log_path, indent(f"REJECTED: {reason}", "  "))
            feedback = f"The previous prediction was rejected. {reason}"
        raise ValueError(f"Next-state checking failed after {MODEL_TRIALS} predictions. {feedback}")

    def pending_feedback_expansion(self, key):
        versions = self.feedback.active_versions_for_state(self.feedback_goal, key)
        if versions:
            expansion = key, tuple(versions.items())
            if self.feedback_attempts.get(expansion, 0) < MAX_FEEDBACK_ATTEMPTS:
                return expansion
        return None

    def record_feedback_expansion(self, expansion, trigger, outcome, grid_step, **details):
        attempt = self.feedback_attempts.get(expansion, 0) + 1
        if outcome in ("new_candidate", "duplicate_candidate"):
            # A valid proposal uses the opportunity even if the action is already known.
            self.feedback_attempts[expansion] = MAX_FEEDBACK_ATTEMPTS
        else:
            self.feedback_attempts[expansion] = attempt
        key, versions = expansion
        self.feedback_search[f"{trigger}_expansions"] += 1
        self.feedback_search[f"{outcome}s"] += 1
        self.feedback.log("feedback_expansion", state=key, feedback_versions=dict(versions),
                          trigger=trigger, outcome=outcome, attempt=attempt, grid_step=grid_step,
                          **details)

    def remember_transition(self, key, action, next_state, reward):
        start_tick = json.loads(key)["grid_steps"]
        end_tick = json.loads(next_state)["grid_steps"]
        if not start_tick < end_tick <= self.args.grid_horizon:
            raise ValueError("A transition must advance the episode clock within the grid horizon.")
        metadata = {"reward": reward, "grid_steps": end_tick - start_tick}
        self.node(next_state)
        self.memory.update([key, action, next_state], edge_metadata=metadata)
        self.graph.update([key, action, next_state], edge_metadata=metadata)

    def route_values(self):
        # Time strictly increases on each edge, so a backward pass is exact on the known graph.
        nodes = [(json.loads(node.state)["grid_steps"], node) for node in self.graph.nodes
                 if not node.state.endswith("_sub_virtual\n")]
        partial, complete = {}, {}
        for tick, node in sorted(nodes, key=lambda item: item[0], reverse=True):
            if tick == self.args.grid_horizon:
                partial[node.state] = complete[node.state] = 0
                continue
            edges = [edge for edge in node.children if edge["action"] != "action_virtual"]
            partial[node.state] = max(
                (edge["reward"] + partial[edge["child"].state] for edge in edges), default=0)
            complete_scores = [edge["reward"] + complete[edge["child"].state] for edge in edges
                               if edge["child"].state in complete]
            if complete_scores:
                complete[node.state] = max(complete_scores)
        return partial, complete

    def imagine(self, initial, directory, start_tick=0, *, trigger="imagination", reason="initial"):
        state = initial
        grid_step = start_tick
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "log.txt"
        initial_state = json.dumps(json.loads(state_key(state, grid_step)), indent=2)
        log_path.write_text(
            f"[Start imagination | absolute grid step {start_tick} → {self.args.grid_horizon}]\n"
            f"[Reason] {reason}\n"
            "[Initial state | players: RightChef, LeftChef | grid_steps: absolute]\n"
            f"{initial_state}\n",
            encoding="utf-8",
        )
        for timestep in count(1):
            if grid_step == self.args.grid_horizon:
                break
            if (timestep - 1) % self.args.tsteps == 0:
                partial, _ = self.route_values()
                batch = (timestep - 1) // self.args.tsteps + 1
                append_round_log(log_path, f"\n[Search batch {batch} | absolute grid step {grid_step}]")
            append_round_log(log_path, f"\n========== Decision {timestep} | timestep_{timestep} | "
                                       f"absolute grid step {grid_step} ==========")
            key = state_key(state, grid_step)
            node = self.node(key)
            self.graph.build_virtual_node(key)
            visits = sum(edge["child"].visit_num for edge in node.children)

            def ucb_score(edge):
                child = edge["child"]
                value = child.value if edge["action"] == "action_virtual" else (
                    edge["reward"] + partial.get(child.state, 0)
                ) / self.kitchen.mdp.delivery_reward
                return value / child.visit_num + 2 * sqrt(log(visits) / child.visit_num)

            expansion = self.pending_feedback_expansion(key)
            best = max(node.children, key=ucb_score)
            ucb_action = action = best["action"]
            child = best["child"]
            if expansion is not None:
                action = "action_virtual"
                child = node.get_child(action)
            proposed = action == "action_virtual"
            if proposed and child is not None:
                child.visit_num += 1
            source = "LLM" if proposed else "Memory"
            if expansion is not None:
                source += " (previous episode feedback)"
            append_round_log(log_path, f"[Source] {source}")
            prediction = None
            try:
                if proposed:
                    known_actions = [edge["action"] for edge in node.children
                                     if edge["action"] != "action_virtual"]
                    step_dir = directory / f"timestep_{timestep}"
                    action, plans = self.predict_action(key, state, grid_step, step_dir, known_actions)
                    if action not in known_actions:
                        prediction = self.predict_next_state(key, action, plans, grid_step, step_dir)
            except ValueError as error:
                if expansion is not None:
                    self.record_feedback_expansion(expansion, trigger, "rejected_candidate",
                                                   grid_step, ucb_action=ucb_action, error=str(error))
                append_round_log(log_path, f"[Candidate rejected] {error}")
                known = self.known_edges(key)
                if not known:
                    break
                action = max(known, key=ucb_score)["action"]
                proposed = False
                expansion = None
                append_round_log(log_path, "[Source] Memory (fallback)")

            new_edge = prediction is not None
            if not proposed:
                append_round_log(log_path, "[Action]\n" + indent(action.strip(), "  "))
            if new_edge:
                next_state, delivered = prediction
                reward = self.kitchen.mdp.delivery_reward if delivered else 0
            else:
                edge = next(edge for edge in node.children if edge["action"] == action)
                next_state = edge["child"].state
                reward = edge["reward"]
                delivered = reward > 0
            next_record = json.loads(next_state)
            next_grid_step = next_record.pop("grid_steps")
            if new_edge:
                self.remember_transition(key, action, next_state, reward)
            else:
                edge["child"].visit_num += 1
            if proposed:
                self.feedback.record_proposal(key, action, "accepted" if new_edge else "known")
                if new_edge:
                    self.feedback.record_new_edge(self.feedback_goal, key, action)
            if expansion is not None:
                outcome = "new_candidate" if new_edge else "duplicate_candidate"
                self.record_feedback_expansion(expansion, trigger, outcome,
                                               grid_step, ucb_action=ucb_action, joint_action=action)
            result = "Transition saved" if new_edge else "Memory transition reused"
            append_round_log(log_path, f"\n[Result] {result} | predicted absolute grid steps "
                                       f"{grid_step} → {next_grid_step} | predicted reward {reward}")
            if new_edge:
                append_round_log(log_path, describe_state_changes(json.loads(key), next_record, delivered))
            grid_step = next_grid_step
            state = OvercookedState.from_dict(next_record)
        outcome = "horizon_reached" if grid_step == self.args.grid_horizon else "plan_incomplete"
        append_round_log(log_path, f"\n[End imagination | {outcome} | absolute grid step {grid_step}]")
        return grid_step

    def plan_from(self, state, start_tick=0, *, trigger="imagination", reason="initial"):
        furthest_tick = start_tick
        start_state = preprocess_state(state_key(state, start_tick))
        rounds = 1 if reason == "feedback" else self.args.imagination_rounds
        for _ in range(rounds):
            self.round_id += 1
            directory = self.case_dir / "simulate" / f"round_{self.round_id}"
            last_tick = self.imagine(state, directory, start_tick, trigger=trigger, reason=reason)
            furthest_tick = max(furthest_tick, last_tick)
            _, complete = self.route_values()
            if start_state in complete:
                break
        predicted_reward = complete.get(start_state)
        self.feedback.log("kitchen_plan", trigger=trigger, reason=reason, start_grid_step=start_tick,
                          furthest_grid_step=furthest_tick, target_grid_step=self.args.grid_horizon,
                          complete=predicted_reward is not None, predicted_reward=predicted_reward)
        return complete

    def known_edges(self, key):
        node = self.graph.get_node(key)
        return [] if node is None else [edge for edge in node.children if edge["action"] != "action_virtual"]

    def rollout(self, episode_id, complete):
        state = self.kitchen.env.state
        replay_states = [state.deepcopy()]
        directory = self.case_dir / "interact" / f"round_{episode_id}"
        directory.mkdir(parents=True)
        log_path = directory / "log.txt"
        initial_key = state_key(state, 0)
        initial_record = json.dumps(json.loads(initial_key), indent=2)
        fixtures = fill(self.fixtures, width=100, initial_indent="  ", subsequent_indent="  ")
        initial_log = (
            f"[Start execution | Grid step 0 of {self.args.grid_horizon}]\n"
            f"[Kitchen fixtures]\n{fixtures}\n"
            f"\n[Initial state | Players: {', '.join(AGENTS)}]\n{initial_record}\n"
        )
        log_path.write_text(initial_log, encoding="utf-8")
        steps = []
        delivery_ticks = []
        outcome = "horizon_reached"
        prediction_changed = False
        initial_predicted_reward = complete.get(preprocess_state(initial_key))
        for step in count(1):
            start_tick = self.kitchen.env.t
            key = state_key(state, start_tick)
            state_id = preprocess_state(key)
            has_feedback = self.pending_feedback_expansion(key) is not None
            if step > 1 and (has_feedback or state_id not in complete):
                if has_feedback:
                    reason = "feedback"
                elif prediction_changed:
                    reason = "execution_prediction_mismatch"
                else:
                    reason = "missing_complete_route"
                append_round_log(log_path, f"\n[Replan | Grid step {start_tick} → {self.args.grid_horizon}] {reason}")
                requests_before = self.usage["planning"]["requests"]
                complete = self.plan_from(state, start_tick, trigger="rollout", reason=reason)
                self.feedback_search["rollout_searches"] += 1
                requests = self.usage["planning"]["requests"] - requests_before
                self.feedback_search["rollout_planning_requests"] += requests
                if has_feedback:
                    self.feedback.log("rollout_feedback_search", state=key, episode=episode_id,
                                      grid_step=start_tick, step=step, planning_requests=requests)
            known = [edge for edge in self.known_edges(key) if edge["child"].state in complete]
            if not known:
                outcome = "plan_incomplete"
                append_round_log(log_path, f"\n[Execution stopped] No complete route from grid step {start_tick} "
                                           f"to {self.args.grid_horizon}.")
                break
            chosen = max(known, key=lambda edge: edge["reward"] + complete[edge["child"].state])
            action = chosen["action"]
            transition = self.execute(self.kitchen, action, replay_states)
            state = transition.state
            next_key = state_key(state, transition.grid_steps)
            predicted = chosen["child"].state
            predicted_reward = chosen["reward"]
            prediction_changed = (preprocess_state(next_key) != predicted
                                  or transition.reward != predicted_reward)
            if prediction_changed:
                self.remember_transition(key, action, next_key, transition.reward)
                self.feedback.log("execution_transition_corrected", state=key, joint_action=action,
                                  prediction=predicted, actual_state=next_key,
                                  predicted_reward=predicted_reward, reward=transition.reward,
                                  predicted_delivery=predicted_reward > 0, delivered=transition.delivered)
                _, complete = self.route_values()
            if transition.delivered:
                delivery_ticks.append(transition.grid_steps)
            versions = self.feedback.versions_for_execution(self.feedback_goal, key, action)
            steps.append(dict(step=step, state=key, joint_action=action, next_state=next_key,
                              reward=transition.reward, done=transition.done, execution="complete",
                              feedback_versions=versions, grid_steps=transition.grid_steps - start_tick))
            end_tick = transition.grid_steps
            grid_range = (f"Grid step {end_tick}" if end_tick == start_tick + 1
                          else f"Grid steps {start_tick + 1}–{end_tick}")
            step_log = [
                f"\n========== Step {step} | {grid_range} ==========",
                "[Executed action]",
                indent(action.strip(), "  "),
                describe_state_changes(json.loads(key), json.loads(next_key), transition.delivered),
                f"[Reward] {transition.reward:+g} | Total {self.kitchen.env.cumulative_sparse_rewards:g}",
            ]
            if prediction_changed:
                predicted_tick = json.loads(predicted)["grid_steps"]
                step_log.append(f"[Prediction mismatch] Expected grid step {predicted_tick}, "
                                f"reward {predicted_reward:g}. Actual grid step {end_tick}, "
                                f"reward {transition.reward:g}. See the recorded state changes above.")
            append_round_log(log_path, "\n".join(step_log))
            if transition.done:
                break
        append_round_log(log_path, f"\n[End execution | {outcome} | Grid step {self.kitchen.env.t} "
                                   f"of {self.args.grid_horizon}]\n"
                                   f"[Summary] Total reward {self.kitchen.env.cumulative_sparse_rewards:g} | "
                                   f"Soups delivered {len(delivery_ticks)} | Joint actions {len(steps)}")
        round_log = log_path.read_text(encoding="utf-8")
        save_replay(directory / "replay_video.mp4", self.kitchen.mdp, replay_states, outcome)
        episode = dict(episode=episode_id, outcome=outcome, steps=steps)
        summary = dict(outcome=outcome, decisions=len(steps), grid_steps=self.kitchen.env.t,
                       reward=self.kitchen.env.cumulative_sparse_rewards,
                       soups_delivered=len(delivery_ticks), delivery_grid_steps=delivery_ticks,
                       initial_plan_complete=initial_predicted_reward is not None,
                       initial_predicted_reward=initial_predicted_reward)
        return episode, round_log, summary

    def run(self):
        summaries = []
        self.feedback.reset(self.case_dir)
        for episode_id in range(1, self.args.episodes + 1):
            initial = self.kitchen.reset()
            self.reset_search_graph()
            complete = self.plan_from(initial)
            episode, round_log, summary = self.rollout(episode_id, complete)
            summaries.append(summary)
            self.memory.save()
            directory = self.case_dir / "interact" / f"round_{episode_id}"
            self.feedback.review(self.feedback_goal, episode, round_log,
                                 lambda prompt: self.ask(prompt, evaluation=True), directory)
        first_deliveries = [
            item["delivery_grid_steps"][0] for item in summaries if item["delivery_grid_steps"]
        ]
        result = dict(finished=True, episodes=summaries, usage=self.usage,
                      model=self.args.llm_source, imagination_rounds=self.args.imagination_rounds,
                      decision_limit=None, grid_horizon=self.args.grid_horizon,
                      full_horizon=True, planning_batch_size=self.args.tsteps,
                      first_delivery_grid_steps=first_deliveries[0] if first_deliveries else None,
                      best_delivery_grid_steps=min(first_deliveries, default=None))
        result["trajectory_feedback"] = self.feedback.summary()
        result["feedback_search"] = self.feedback_search
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
