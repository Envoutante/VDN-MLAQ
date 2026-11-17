"""Replay a hand-written route in the official Forced Coordination layout."""

from overcooked_ai_py.mdp.actions import Action, Direction

from .adapter import OvercookedFirstSoup


N, S, E, W = Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST
WAIT, INTERACT = Action.STAY, Action.INTERACT
COUNTER = (2, 2)
POT = (3, 0)


def deliver_first_soup(kitchen):
    state = kitchen.reset()
    assert tuple(player.position for player in state.players) == ((3, 1), (1, 2))

    def advance(right, left):
        state, reward, done, _ = kitchen.step((right, left))
        assert reward == 0 and not done
        return state

    onion_steps = []
    for onion in range(1, 4):
        advance(S, W)
        advance(W, INTERACT)
        advance(WAIT, E)
        state = advance(WAIT, INTERACT)
        assert state.get_object(COUNTER).name == "onion"
        state = advance(INTERACT, WAIT)
        assert state.players[0].get_object().name == "onion"
        advance(N, WAIT)
        state = advance(INTERACT, WAIT)
        assert state.get_object(POT).state[:2] == ("onion", onion)
        onion_steps.append(kitchen.env.t)

    advance(S, S)
    advance(W, W)
    advance(WAIT, INTERACT)
    advance(WAIT, N)
    advance(WAIT, E)
    state = advance(WAIT, INTERACT)
    assert state.get_object(COUNTER).name == "dish"
    dish_step = kitchen.env.t
    advance(INTERACT, WAIT)
    state = advance(N, WAIT)

    cooking_ticks = state.get_object(POT).state[2]
    for _ in range(kitchen.mdp.soup_cooking_time - cooking_ticks):
        state = advance(WAIT, WAIT)
    assert state.get_object(POT).state == ("onion", 3, kitchen.mdp.soup_cooking_time)
    ready_step = kitchen.env.t

    state = advance(INTERACT, WAIT)
    assert state.players[0].get_object().name == "soup"
    advance(S, WAIT)
    advance(S, WAIT)
    state, reward, done, info = kitchen.step((INTERACT, WAIT))
    assert done and info["delivered"] and reward == 20
    assert not state.players[0].has_object()
    return onion_steps, dish_step, ready_step, info["grid_steps"], reward


if __name__ == "__main__":
    kitchen = OvercookedFirstSoup()
    result = deliver_first_soup(kitchen)
    assert result == deliver_first_soup(kitchen), "Reset must restore the initial kitchen"
    onions, dish, ready, delivery, reward = result
    print(f"Onions in pot at grid steps {onions}; plate on shared counter at {dish}.")
    print(f"Soup ready at grid step {ready}; delivered at grid step {delivery}; sparse reward {reward}.")

    deadline = OvercookedFirstSoup(horizon=1)
    _, reward, done, info = deadline.step((WAIT, WAIT))
    assert done and reward == 0 and not info["delivered"] and info["grid_steps"] == 1
    print("PASS: deterministic reset and separate delivery/timeout outcomes.")
