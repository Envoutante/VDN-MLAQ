"""Estimate action advantages and exploration returns from the search graph."""

from collections import Counter, defaultdict, deque
from math import log, sqrt

from prompting.lang_tree import parse_joint_action, preprocess_state


def select_agent(tree, context, pending):
    """Prefer untried directions, then rank by improvement rate and uncertainty."""
    agents = tree.agent_names
    in_progress = Counter(event['agent'] for event in pending if event['context'] == context)
    counts = {agent: tree.exploration_stats.get((*context, agent), (0, 0)) for agent in agents}
    effective = {agent: counts[agent][0] + in_progress[agent] for agent in agents}
    if any(n == 0 for n in effective.values()):
        return min(agents, key=effective.get), 'single_cold_start', {}
    if any(n == 0 for n, _ in counts.values()):
        return min(agents, key=effective.get), 'single_pending', {}
    total = sum(effective.values())
    scores = {agent: successes / n + sqrt(2 * log(total) / effective[agent])
              for agent, (n, successes) in counts.items()}
    return max(agents, key=scores.get), 'single_value', scores


def known_branch_steps(tree, target, if_fail):
    """Length through each known joint edge, counting the edge itself."""
    reverse = defaultdict(set)
    edges = {}
    terminal = {target, preprocess_state('Success'), preprocess_state('Failure')}
    rejected = {node.state for node in tree.nodes if node.state not in terminal and if_fail(node.state)}
    for node in tree.nodes:
        if node.state in terminal or node.state in rejected:
            continue
        for edge in node.children:
            action, successor = edge['action'], edge['child'].state
            if successor in rejected or successor in terminal - {target}:
                continue
            if parse_joint_action(action, tree.agent_names) is None:
                continue
            edges[node.state, action] = successor
            reverse[successor].add(node.state)
    dist, queue = {target: 0}, deque([target])
    while queue:
        successor = queue.popleft()
        for source in reverse[successor]:
            if source not in dist:
                dist[source] = dist[successor] + 1
                queue.append(source)
    return {edge: 1 + dist[successor] for edge, successor in edges.items() if successor in dist}


def finish_round(tree, target, attempts, if_fail):
    """Refresh graph advantages and credit each completed request exactly once."""
    if not attempts and not any(key[0] == target for key in tree.action_advantages):
        return
    steps = known_branch_steps(tree, target, if_fail)
    for event in attempts:
        if event['candidate'] is not None:
            tree.action_advantages[(*event['context'], event['agent'], event['candidate'])] = None
    for key in tree.action_advantages:
        goal, state, anchor, _, candidate = key
        if goal != target:
            continue
        old_steps = steps.get((state, anchor))
        new_steps = steps.get((state, candidate))
        advantage = None
        if old_steps is not None and new_steps is not None:
            advantage = tree.q_gamma ** (new_steps - 1) - tree.q_gamma ** (old_steps - 1)
        tree.action_advantages[key] = dict(anchor_steps=old_steps, candidate_steps=new_steps, advantage=advantage)
    for event in attempts:
        _, state, anchor = event['context']
        stat_key = (*event['context'], event['agent'])
        old_steps = steps.get((state, anchor))
        new_steps = steps.get((state, event['candidate']))
        improvement = new_steps is not None and (old_steps is None or new_steps < old_steps)
        n, successes = tree.exploration_stats.get(stat_key, (0, 0))
        tree.exploration_stats[stat_key] = (n + 1, successes + int(improvement))
        event.update(anchor_steps=old_steps, candidate_steps=new_steps, improvement=improvement)
        event['advantage'] = None if event['candidate'] is None else tree.action_advantages[(*stat_key, event['candidate'])]['advantage']
