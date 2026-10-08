"""Enemy melee roles, AI reachability and saved-map compatibility."""
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stoix.tests.number_grid_fixtures import MAP
from stoix.envs.number_grid import NumberGrid, SHOOT, DEFEND, CONTINUE, HIT, MISS, GUARD


# These regressions retain the saved v10 combat contract. Version 2 has its own tests.
MAP = {**MAP, 'combat_rules_version': 1, 'battle_observation_version': 3}


def battle(slot=1, **overrides):
    counts = list(MAP['enemy_units'])
    counts[6] = 6
    roles = [-1] * len(counts)
    roles[6] = slot
    env = NumberGrid(map_config={**MAP, 'enemy_units': counts,
                                'enemy_warrior_slots': roles, **overrides})
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = env._begin_battle(state.replace(enemy=jnp.int32(6)))
    return env, state.replace(actor=jnp.int32(6 + slot))


def reachable(slot, enemies, front):
    if slot >= 3 and front:
        return set()
    near = ({0, 1}, {0, 1, 2}, {1, 2})[slot % 3]
    for row in ({0, 1, 2}, {3, 4, 5}):
        candidates = set(enemies) & row
        adjacent = {i for i in candidates if i % 3 in near}
        if adjacent or candidates:
            return adjacent or candidates
    return set()


def test_current_map_has_one_front_warrior_per_squad_with_unchanged_hp():
    env = NumberGrid(map_config=MAP)
    assert MAP['enemy_warrior_slots'] == [0 if n == 1 else 1 for n in MAP['enemy_units']]
    assert sum(MAP['enemy_units']) == 94 and len(MAP['enemy_warrior_slots']) == 24
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert ts.observation.shape == (178,)
    for enemy, slot in enumerate(MAP['enemy_warrior_slots']):
        started = env._begin_battle(state.replace(enemy=jnp.int32(enemy)))
        assert started.hp[6 + slot] == MAP['enemy_hp'][enemy]
        np.testing.assert_array_equal(started.hp, env.max_hp(started))
        assert env.observation(started)[124 + (6 + slot) * 4] == 1


@pytest.mark.parametrize('slot', range(6))
def test_enemy_ai_obeys_melee_geometry_for_death_escape_and_pending_retreat(slot):
    env, state = battle(slot)
    # Independent oracle: dead, present, preparing retreat, escaped.
    statuses = np.array(list(itertools.product(range(4), repeat=6)))
    def check(es, own):
        hp = state.hp.at[:6].set(jnp.where(es == 0, 0, 45))
        hp = hp.at[6:9].set(jnp.where(own == 0, 0, 35)).at[6 + slot].set(35)
        escaped = state.escaped.at[:6].set(es == 3).at[6:9].set(own == 3).at[6 + slot].set(False)
        retreating = state.retreating.at[:6].set(es == 2).at[6:9].set(own == 2).at[6 + slot].set(False)
        s = state.replace(hp=hp, escaped=escaped, retreating=retreating)
        return env._melee_targets(s, enemy_side=True), env._enemy_action(s, jnp.full(12, .5))
    actual, actions = jax.jit(jax.vmap(jax.vmap(check, in_axes=(None, 0)), in_axes=(0, None)))(
        jnp.asarray(statuses), jnp.arange(4))
    actual, actions = np.asarray(actual), np.asarray(actions)
    for i, status in enumerate(statuses):
        living = set(np.flatnonzero((status == 1) | (status == 2)))
        for own in range(4):
            expected = reachable(slot, living, own in (1, 2))
            assert set(np.flatnonzero(actual[i, own])) == expected
            assert actions[i, own] == (SHOOT + min(expected) if expected else DEFEND)


@pytest.mark.parametrize('accuracy,defended,damage,event', [
    (1., False, 25, HIT), (1., True, 13, HIT), (0., False, 0, MISS),
])
def test_enemy_warrior_uses_sword_stats_and_never_shoots_through_front(accuracy, defended, damage, event):
    env, state = battle(warrior_accuracy=accuracy, archer_accuracy=0.)
    # Rear target has 1 HP, but the front blocks it even while retreating.
    state = state.replace(hp=state.hp.at[:6].set(jnp.array([45, 100, 45, 1, 45, 45])),
                          defended=state.defended.at[:3].set(defended),
                          retreating=state.retreating.at[:3].set(True))
    result, ts = jax.jit(env.step)(state, jnp.int32(CONTINUE))
    assert result.last_target in (0, 2) and result.last_damage == damage
    assert result.last_event == event and result.hp[3] == 1
    assert result.enemy_turns == state.enemy_turns + 1 and ts.extras['enemy_battle_transition']
    assert result.turn_phase[7] == 2
    np.testing.assert_array_equal(np.flatnonzero(env.action_mask(state)), [CONTINUE])


def test_ai_uses_actual_sword_damage_for_kill_priority():
    env, state = battle(warrior_damage=40, warrior_accuracy=1.)
    state = state.replace(hp=state.hp.at[:6].set(jnp.array([21, 30, 45, 1, 45, 45])),
                          defended=state.defended.at[0].set(True))
    result, _ = jax.jit(env.step)(state, jnp.int32(CONTINUE))
    assert result.last_target == 1 and result.hp[1] == 0 and result.hp[0] == 21


def test_rear_warrior_defends_until_all_front_allies_leave():
    env, state = battle(4, warrior_accuracy=1.)
    blocked, _ = jax.jit(env.step)(state, jnp.int32(CONTINUE))
    assert blocked.last_event == GUARD and blocked.defended[10]
    np.testing.assert_array_equal(blocked.hp, state.hp)
    freed = state.replace(escaped=state.escaped.at[6:9].set(True),
                          hp=state.hp.at[:3].set(0))
    result, _ = env.step(freed, jnp.int32(CONTINUE))
    assert result.last_event == HIT and result.last_target >= 3


def test_enemy_archer_still_targets_rear_and_uses_archer_stats():
    env, state = battle(1, archer_accuracy=1., warrior_accuracy=0.)
    state = state.replace(actor=jnp.int32(11), hp=state.hp.at[5].set(5))
    result, _ = jax.jit(env.step)(state, jnp.int32(CONTINUE))
    assert result.last_target == 5 and result.hp[5] == 0 and result.last_event == HIT


def test_enemy_warrior_initiative_is_scaled_at_start_and_each_round():
    env, state = battle()
    draws = jnp.full(13, .5)
    started = env._begin_battle(state, state.battle_key, draws)
    np.testing.assert_array_equal(started.priority, [1.5, 1.25, 1.5, 1.5, 1.5, 1.5, 1.5, 1.25, 1.5, 1.5, 1.5, 1.5])
    last = state.replace(turn_phase=jnp.full(12, 2).at[7].set(0))
    result, _ = jax.jit(env._battle_step)(last, jnp.int32(CONTINUE), last.battle_key, draws)
    assert result.round == last.round + 1
    np.testing.assert_array_equal(result.priority, started.priority)


@pytest.mark.parametrize('roles', [None, [], [0], [True] * 24, [-2] * 24, [6] * 24, [1] * 24])
def test_invalid_enemy_warrior_slots_are_rejected(roles):
    with pytest.raises(ValueError, match='[Ee]nemy warrior'):
        NumberGrid(map_config={**MAP, 'enemy_warrior_slots': roles})


def test_saved_v9_map_keeps_enemy_archers_and_observation():
    path = Path(__file__).resolve().parents[2] / 'maps/number_grid-24x24-v9-warrior-24-squads.json'
    env = NumberGrid(map_config=json.loads(path.read_text()))
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert not env.has_enemy_warriors and ts.observation.shape == (178,)
    state = env._begin_battle(state.replace(enemy=jnp.int32(6)))
    state = state.replace(actor=jnp.int32(7), hp=state.hp.at[5].set(1))
    assert env._enemy_action(state, jnp.zeros(12)) == SHOOT + 5
    assert env._begin_battle(state, state.battle_key, jnp.full(13, .5)).priority[7] == 1.5
