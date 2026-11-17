"""Case-local, evidence-linked advice for MLAQ's joint action proposals."""

import json
from itertools import count
from pathlib import Path
from uuid import uuid4

from prompting.lang_tree import preprocess_action, preprocess_state


REVIEW_INSTRUCTION = """Read the execution log. Could the robots reach the same goal in fewer steps?
Even if they succeeded, check whether a robot that waited could safely move a cube
in parallel or prepare a handoff one step earlier. Suggest the exact move and any
teammate move it depends on. Use completed moves as evidence and treat alternatives as untested.

Alice can pick and place on panels 1–3, Bob on 3–5, Chad on 5–7.
Each robot can move one cube or wait per step. Joint moves must pick different cubes
and use different destinations. A destination can be freed by a pick in the same step.

Give one individual recommendation for each robot. If a robot acted well, say
what specific action it should keep doing and why. For a coordinated change, use
the same step for the affected robots and give each its own part of the plan.
Check that the recommendations work together, including any later handoff.

Return only JSON. Each robot must appear exactly once, using this format:
{"updates":[{"step":1,"agent":"Alice","decision":"add","evidence_step":1,"reason":"why this could help","condition":"when it applies","advice":"what to do"}]}
Use completed step numbers from the log. The suggestion applies before "step",
and "evidence_step" supports it. Use "add" when the robot has no existing advice
at that state. Include the full set of robots even if the route needs no change.
"""

REVISE_INSTRUCTION = """If the selected state and robot already have advice, give "previous_version".
Use "decision":"keep" with a reason to leave that advice active. To replace it,
use "decision":"revise", give the replacement condition and advice, and set
"followed":true only if the old version was shown, its condition held, and the
robot actually followed it. Use "keep" if those conditions have not been verified.
"""


def empty_usage():
    return dict(requests=0, input_tokens=0, output_tokens=0, estimated=0, failed=0)


class TrajectoryFeedback:
    def __init__(self, agents):
        self.agents = tuple(agents)
        self.reset()

    def reset(self, case_dir=None):
        self.log_path = Path(case_dir) / 'trajectory_feedback.jsonl' if case_dir else None
        self.session = uuid4().hex
        self.active = {}
        self.edge_versions = {}
        self.shown_versions = set()
        self.dialog_versions = {}
        self.reviewed = set()
        self.next_version = 1
        self.counts = dict(added=0, revised=0, kept=0, prompt_uses=0, proposals=0,
                           new_edges=0, executed_edges=0, reviews=0, invalid_reviews=0)
        self.usage = {'planning': empty_usage(), 'evaluation': empty_usage()}

    def log(self, event, **data):
        if self.log_path is not None:
            row = dict(session=self.session, event=event, **data)
            with self.log_path.open('a', encoding='utf-8') as file:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')

    def record_usage(self, evaluation, requests, input_tokens, output_tokens, estimated, failed):
        usage = self.usage['evaluation' if evaluation else 'planning']
        usage['requests'] += requests
        usage['input_tokens'] += input_tokens
        usage['output_tokens'] += output_tokens
        usage['estimated'] += int(estimated)
        usage['failed'] += int(failed)

    def advice_for_prompt(self, goal, state, agent):
        entry = self.active.get((preprocess_state(goal), preprocess_state(state), agent))
        if entry is None:
            return ''
        version = entry['version']
        self.dialog_versions[agent] = version
        self.shown_versions.add(version)
        self.counts['prompt_uses'] += 1
        self.log('advice_shown', agent=agent, state=state, version=version)
        return (
            '\n[Advice from a previous executed episode]\n'
            'Consider this only if the condition holds. Confirm any required teammate action.\n'
            f"Condition: {entry['condition']}\nAdvice: {entry['advice']}\n"
        )

    def record_proposal(self, state, action, status):
        if self.dialog_versions:
            self.counts['proposals'] += 1
            self.log('imagined_proposal', state=state, joint_action=action,
                     status=status, feedback_versions=self.dialog_versions)

    def record_new_edge(self, goal, state, action):
        key = (preprocess_state(goal), preprocess_state(state), preprocess_action(action))
        if self.dialog_versions:
            self.edge_versions[key] = self.dialog_versions.copy()
            self.counts['new_edges'] += 1
            self.log('imagined_edge_added', state=state, joint_action=action,
                     feedback_versions=self.dialog_versions)
        else:
            self.edge_versions.pop(key, None)

    def versions_for_execution(self, goal, state, action):
        key = (preprocess_state(goal), preprocess_state(state), preprocess_action(action))
        versions = self.edge_versions.get(key, {}).copy()
        self.counts['executed_edges'] += bool(versions)
        return versions

    def review(self, goal, episode, round_log, query, round_save_dir):
        self.log('episode', goal=goal, **episode)
        steps = {step['step']: step for step in episode['steps'] if step['execution'] == 'complete'}
        if not steps:
            return

        goal = preprocess_state(goal)
        visited = {preprocess_state(step['state']) for step in steps.values()}
        previous = [entry for (entry_goal, state, _), entry in self.active.items()
                    if entry_goal == goal and state in visited]
        prompt = (
            REVIEW_INSTRUCTION + '\nRequired robots: ' + ', '.join(self.agents)
            + '\n\nExecution log:\n' + round_log.strip() + '\n'
        )
        if previous:
            prompt += '\n' + REVISE_INSTRUCTION + '\nPrevious advice:\n'
            for step in steps.values():
                state = preprocess_state(step['state'])
                for entry in previous:
                    if entry['state'] != state:
                        continue
                    agent = entry['agent']
                    version = entry['version']
                    shown = step['feedback_versions'].get(agent) == version
                    prompt += (
                        f"Step {step['step']}, {agent}, version {version}, "
                        f"shown when proposing this action: {'yes' if shown else 'no'}.\n"
                        f"When: {entry['condition']}\nAdvice: {entry['advice']}\n"
                    )
        if prompt in self.reviewed:
            self.log('review_skipped', episode=episode['episode'], reason='duplicate_evidence')
            return

        query_dir = None
        if self.log_path is not None:
            critic_dir = Path(round_save_dir) / 'critic_query'
            for trial in count(1):
                query_path = critic_dir / f'trial_{trial}'
                try:
                    query_path.mkdir(parents=True)
                except FileExistsError:
                    continue
                break
            query_dir = query_path.relative_to(self.log_path.parent).as_posix()
            (query_path / 'main_prompt.txt').write_bytes(prompt.encode('utf-8'))

        response = query(prompt)
        if self.log_path is not None:
            (query_path / 'main_response.txt').write_bytes(response.encode('utf-8'))
        self.counts['reviews'] += 1
        self.log('review', episode=episode['episode'], previous_advice=previous, response=response,
                 query_dir=query_dir)
        try:
            updates = self.parse(response)
            validated = [(self.validate_update(goal, steps, update), update) for update in updates]
        except ValueError as error:
            self.counts['invalid_reviews'] += 1
            self.log('invalid_review', episode=episode['episode'], reason=str(error))
            return

        self.reviewed.add(prompt)
        for (key, reason), update in validated:
            self.apply_update(key, reason, update, episode['episode'])

    def parse(self, response):
        response = response.strip()
        if not response:
            raise ValueError('evaluation returned no answer content')
        if response.startswith(('```json\n', '```\n')) and response.endswith('```'):
            response = response.split('\n', 1)[1][:-3].strip()
        data = json.loads(response)
        if not isinstance(data, dict) or not isinstance(data.get('updates'), list):
            raise ValueError('expected a JSON object with an updates list')
        updates = data['updates']
        agents = [update.get('agent') for update in updates if isinstance(update, dict)]
        if (len(updates) != len(self.agents) or len(agents) != len(self.agents)
                or any(agents.count(agent) != 1 for agent in self.agents)):
            raise ValueError('review must include exactly one recommendation for each agent')
        return updates

    def validate_update(self, goal, steps, update):
        step = update.get('step')
        evidence_step = update.get('evidence_step')
        agent = update.get('agent')
        decision = update.get('decision')
        if type(step) is not int or step not in steps:
            raise ValueError('advice must reference a completed execution step')
        if type(evidence_step) is not int or evidence_step not in steps:
            raise ValueError('evidence must reference a completed execution step')
        if decision not in ('add', 'revise', 'keep'):
            raise ValueError('unknown decision')

        key = (goal, preprocess_state(steps[step]['state']), agent)
        previous = self.active.get(key)
        if decision == 'add':
            if previous is not None:
                raise ValueError('there is already active advice')
        elif (previous is None or type(update.get('previous_version')) is not int
              or update['previous_version'] != previous['version']):
            raise ValueError('previous_version must identify the active advice')
        if decision == 'revise' and (update.get('followed') is not True
                                     or steps[step]['feedback_versions'].get(agent) != previous['version']):
            raise ValueError('the active version was not shown and reported followed at this step')

        reason = self.required_text(update, 'reason')
        if decision != 'keep':
            self.required_text(update, 'condition')
            self.required_text(update, 'advice')
        return key, reason

    def apply_update(self, key, reason, update, episode):
        decision = update['decision']
        previous = self.active.get(key)
        current = previous
        if decision != 'keep':
            current = dict(
                version=self.next_version, state=key[1], agent=key[2],
                condition=update['condition'].strip(), advice=update['advice'].strip(),
            )
            self.next_version += 1
            self.active[key] = current

        self.counts[{'add': 'added', 'revise': 'revised', 'keep': 'kept'}[decision]] += 1
        self.log('advice_updated', goal=key[0], episode=episode, step=update['step'],
                 evidence_step=update['evidence_step'], decision=decision, reason=reason,
                 previous=previous, current=current)

    @staticmethod
    def required_text(update, field):
        value = update.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 600:
            raise ValueError(f'{field} must be a short, nonempty string')
        return value.strip()

    def summary(self):
        return dict(self.counts, session=self.session, active=len(self.active),
                    active_unused=sum(entry['version'] not in self.shown_versions
                                      for entry in self.active.values()),
                    usage=self.usage)
