"""Basic combat characteristics, Python-reference arithmetic and turn scheduling."""
import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stoix.envs.number_grid import (
    MAP, NumberGrid, SHOOT, DEFEND, WAIT, RETREAT, CONTINUE, HIT, MISS,
)
from stoix.envs.number_grid_combat import accuracy_hits, HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE


def battle(hero=None, enemy=None, **overrides):
    game_map = copy.deepcopy(MAP)
    game_map['hero_combat_stats'] = hero or [{} for _ in range(6)]
    if enemy:
        game_map['enemy_combat_stats'] = [[{} for _ in range(n)] for n in MAP['enemy_units']]
        game_map['enemy_combat_stats'][11] = enemy
    game_map.update(overrides)
    env = NumberGrid(map_config=game_map)
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = env._begin_battle(state.replace(enemy=jnp.int32(11)))
    return env, state.replace(actor=jnp.int32(0))


def rolls(first=0., second=0., bonus=0.):
    return (jnp.zeros(36).at[12:18].set(first).at[18:24].set(second)
            .at[24:30].set(bonus))


def attack(env, state, values, action=SHOOT):
    return env._battle_step(state, jnp.int32(action), state.battle_key, values)[0]


def test_default_stats_and_observation_are_individual_and_finite():
    env = NumberGrid()
    state, ts = env.reset(jax.random.PRNGKey(0))
    assert ts.observation.shape == env.observation_space().shape == (238,)
    np.testing.assert_array_equal(env.unit_stats(state)[:, HP], env.max_hp(state))
    np.testing.assert_array_equal(env.unit_stats(state)[1], [100, 25, 80, 0, 50])
    np.testing.assert_array_equal(env.unit_stats(state)[5], [45, 20, 80, 0, 60])
    assert np.all(env.unit_stats(state)[6:] == 0)
    env, state = battle(hero=[dict(max_hp=120, damage=40, accuracy=75, armor=30, initiative=55)] + [{}]*5)
    obs = np.asarray(env.observation(state))[124:232].reshape(12, 9)
    np.testing.assert_allclose(obs[0, 4:], [1.2, 40/300, .75, .3, .55])
    assert obs[0, 0] == 1 and np.isfinite(obs).all()
    assert env.max_hp(state)[0] == 120


@pytest.mark.parametrize('chance,count', [(0, 0), (50, 5050), (80, 9220), (100, 10000)])
def test_two_roll_accuracy_matches_exhaustive_discrete_distribution(chance, count):
    values = (jnp.arange(100) + .25) / 100
    hits = jax.jit(accuracy_hits)(chance, values[:, None], values[None, :])
    assert int(jnp.sum(hits)) == count
    expected = (np.arange(100)[:, None] + np.arange(100)[None, :]) / 2 < chance
    np.testing.assert_array_equal(hits, expected)


@pytest.mark.parametrize('damage,armor,bonus,defended,expected', [
    (25, 0, 0, False, 25), (25, 0, 5, False, 30),
    (25, 0, 0, True, 12), (25, 0, 2, True, 14),
    (25, 40, 5, False, 18), (25, 40, 5, True, 9),
    (15, 70, 0, False, 5),  # Python float/round: 4.500000000000001, not 4.
    (300, 90, 5, True, 15), (400, 100, 5, False, 30),
    (0, 0, 5, False, 0),
])
def test_damage_bonus_cap_armor_defend_and_python_rounding(damage, armor, bonus, defended, expected):
    env, state = battle(hero=[dict(damage=damage)] + [{}]*5,
                        enemy=[dict(max_hp=1000, armor=armor)] + [{}]*5)
    state = state.replace(defended=state.defended.at[6].set(defended))
    result = attack(env, state, rolls(bonus=(bonus+.25)/6))
    assert result.last_damage == expected and result.hp[6] == 1000 - expected
    assert result.last_event == HIT


def test_mage_rolls_hit_and_damage_independently_for_each_target():
    env, state = battle(enemy=[dict(max_hp=100)]*6)
    state = state.replace(actor=jnp.int32(5))
    random = rolls().at[12:18].set(jnp.array([0, .99, 0, .99, 0, .99]))
    random = random.at[18:24].set(random[12:18]).at[24:30].set((jnp.arange(6)+.25)/6)
    result = attack(env, state, random)
    np.testing.assert_array_equal(result.hp[6:], [80, 100, 78, 100, 76, 100])
    assert result.last_damage == 66 and result.last_target == -1
    other_target = attack(env, state, random, SHOOT+5)
    np.testing.assert_array_equal(other_target.hp, result.hp)


def test_hp_clamps_at_zero_miss_does_not_damage_and_dead_unit_leaves_queue():
    env, state = battle()
    state = state.replace(hp=state.hp.at[6].set(1))
    result = attack(env, state, rolls())
    assert result.hp[6] == 0 and result.last_damage == 1 and result.actor != 6
    missed = attack(env, state, rolls(.99, .99))
    np.testing.assert_array_equal(missed.hp, state.hp)
    assert missed.last_event == MISS and missed.turn_phase[0] == 2


def test_initiative_is_additive_rerolled_and_resolves_equal_values_without_side_bias():
    env, state = battle()
    low = env._round_priority(jnp.zeros(36), 11)
    high = env._round_priority(jnp.full(36, .999), 11)
    np.testing.assert_array_equal(jnp.floor(low), env.unit_stats(state)[:, INITIATIVE])
    np.testing.assert_array_equal(jnp.floor(high), env.unit_stats(state)[:, INITIATIVE] + 9)
    assert high[1] < low[0]  # 50+9 cannot outrun 60+0.
    r = jnp.zeros(36).at[0].set(.04).at[6].set(.06)
    assert int(jnp.argmax(env._round_priority(r, 11))) == 6
    assert int(jnp.argmax(env._round_priority(r.at[0].set(.08), 11))) == 0
    # Completing the round draws a new priority vector, even if the last unit waited.
    state = state.replace(turn_phase=jnp.full(12, 2).at[0].set(1))
    result = attack(env, state, jnp.full(36, .5), DEFEND)
    assert result.round == 2
    np.testing.assert_array_equal(result.priority, env._round_priority(jnp.full(36, .5), 11))


def test_wait_is_once_per_round_and_reverses_the_rolled_queue():
    env, state = battle()
    state = state.replace(hp=jnp.array([45, 100, 45, 0, 0, 0, 25, 0, 0, 0, 0, 0]),
                          priority=jnp.array([160., 150., 140., 0, 0, 0, 130., 0, 0, 0, 0, 0]))
    for actor in (0, 1, 2):
        assert state.actor == actor
        state = attack(env, state, rolls(), WAIT)
    assert state.actor == 6
    state = attack(env, state, rolls(.99, .99), CONTINUE)
    for actor in (2, 1, 0):
        assert state.actor == actor and not env.action_mask(state)[WAIT]
        rejected, _ = jax.jit(env.step)(state, jnp.int32(WAIT))
        assert rejected.actor == actor
        np.testing.assert_array_equal(rejected.battle_key, state.battle_key)
        state = attack(env, state, rolls(), DEFEND)
    assert state.round == 2 and env.action_mask(state)[WAIT]


def test_defend_survives_round_boundary_and_expires_at_next_activation():
    env, state = battle()
    state = state.replace(hp=jnp.array([45, 100, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0]),
                          priority=jnp.array([60., 50., 0, 0, 0, 0, 61., 0, 0, 0, 0, 0]))
    state = attack(env, state, rolls(), DEFEND)
    assert state.defended[0]
    state = attack(env, state, rolls(.99, .99), CONTINUE)  # enemy misses
    assert state.actor == 1 and state.defended[0]
    state = attack(env, state, rolls(), DEFEND)
    assert state.round == 2 and not state.defended[0] and state.defended[1]
    state = attack(env, state, rolls(), WAIT)
    assert not state.defended[0]  # waiting cannot extend the previous defence.


def test_enemy_uses_its_own_stats_and_armor_aware_kill_targets():
    env, state = battle(hero=[dict(max_hp=10, armor=90), dict(max_hp=20)] + [{}]*4,
                        enemy=[dict(damage=20, accuracy=0)] + [{}]*5,
                        enemy_warrior_slots=[-1]*24)
    state = state.replace(actor=jnp.int32(6))
    assert env._enemy_action(state, jnp.zeros(6)) == SHOOT+1
    result = attack(env, state, rolls(), CONTINUE)
    np.testing.assert_array_equal(result.hp, state.hp)
    assert result.last_event == MISS


def test_recovery_restores_individual_hp_and_retreat_does_not_grant_extra_attack():
    env, state = battle(hero=[dict(max_hp=120)] + [{}]*5)
    state = state.replace(hp=state.hp.at[6:].set(jnp.array([1, 0, 0, 0, 0, 0])))
    result = attack(env, state, rolls())
    np.testing.assert_array_equal(result.hp[:6], [120, 100, 45, 45, 45, 45])
    assert not result.in_battle
    state = state.replace(actor=jnp.int32(0), turn_phase=state.turn_phase.at[0].set(1))
    result = attack(env, state, rolls(), RETREAT)
    assert result.retreating[0] and result.turn_phase[0] == 2


@pytest.mark.parametrize('change', [
    dict(max_hp=0), dict(max_hp=True), dict(damage=-1), dict(damage=float('nan')),
    dict(accuracy=101), dict(armor=-1), dict(initiative=-1), dict(initiative=1.5), dict(immunity=1),
])
def test_invalid_unit_stats_fail_before_training(change):
    with pytest.raises(ValueError):
        battle(hero=[change]+[{}]*5)


def test_versions_and_override_shapes_are_validated():
    for change in [dict(combat_rules_version=3), dict(battle_observation_version=3),
                   dict(hero_combat_stats=[]), dict(enemy_combat_stats=[[]])]:
        with pytest.raises(ValueError):
            NumberGrid(map_config={**MAP, **change})
    saved = json.loads((Path(__file__).resolve().parents[2] / 'maps/number_grid-24x24-v10-enemy-warriors.json').read_text())
    env = NumberGrid(map_config=saved)
    assert not env.basic_combat and env.reset(jax.random.PRNGKey(0))[1].observation.shape == (178,)


def test_jitted_vmap_rollout_has_finite_observations_and_valid_wait_defend_masks():
    env = NumberGrid(max_steps=200)
    states, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(7), 16))
    states = jax.vmap(lambda s: env._begin_battle(s.replace(enemy=jnp.int32(11))))(states)
    def step(carry, key):
        mask = jax.vmap(env.action_mask)(carry)
        actions = jax.random.categorical(key, jnp.where(mask, 0., -jnp.inf))
        following, ts = jax.vmap(env.step)(carry, actions)
        return following, ts.observation
    final, obs = jax.jit(lambda s: jax.lax.scan(step, s, jax.random.split(jax.random.PRNGKey(8), 200)))(states)
    assert np.isfinite(obs).all() and np.all(final.hp >= 0)
    assert obs.shape == (200, 16, 238)


def test_current_map_has_a_winning_route_with_new_combat(monkeypatch):
    # Reuse the independent preparation-only route verifier on the actual v11 map.
    from stoix.tests import number_grid_battle_test as route_tests
    monkeypatch.setattr(route_tests, 'MAP', MAP)
    route_tests.test_fixed_map_has_a_playable_full_route_with_archer_battles()
