"""Siege reward regressions; CUDA-only, run only on an explicit request.

Source: user reward-design request, 2026-10-13; docs/REFERENCE_CAMPAIGNS.md.
These rewards are not claims about original Disciples II mechanics.
"""
from copy import copy

import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, REST, load_map
from stoix.envs import number_grid_reference_rewards as rewards
from stoix.envs.number_grid_combat import HP


@pytest.fixture(scope='module')
def siege():
    env = NumberGrid(map_config=load_map('siege_train'))
    state, _ = env.reset(jax.random.PRNGKey(7))
    baseline = copy(env)
    baseline.siege_rewards = None
    return env, baseline, state


def active_battle(env, state):
    target = env.objective_enemy
    active = state.replace(enemy=jnp.int32(target),
        scenario_spawned=state.scenario_spawned.at[target].set(True),
        alive=state.alive.at[target].set(True))
    return env._begin_battle(active, jax.random.PRNGKey(11), jnp.full(env.random_size, .5))


def battle_reward(env, before, after, hp, victory=False, lost=False, withdrawal=False):
    return rewards.battle(env, before, after, hp, jnp.bool_(victory),
                          jnp.bool_(lost), jnp.bool_(withdrawal), jnp.bool_(False))


def test_preparation_counts_living_max_hp_once_before_enemy_attacks(siege):
    env, baseline, initial = siege
    # A wounded fighter keeps its development value; a dead fighter contributes zero.
    before = initial.replace(hp=initial.hp.at[0].set(1).at[2].set(0),
        potion_bonus=initial.potion_bonus.at[0, HP].add(100).at[2, HP].add(500),
        day=jnp.int32(27))
    after = env.scenario.advance(before.replace(day=jnp.int32(28)))
    _, shaped = rewards.world(env, before, after, REST, jnp.bool_(True))
    _, plain = rewards.world(baseline, before, after, REST, jnp.bool_(True))
    maximum = env.max_hp(before)
    expected = .05*float(maximum[0]+maximum[1]+maximum[4])
    assert float(shaped-plain) == pytest.approx(expected)
    # Further RESTs cannot pay again, even with additional HP growth.
    later = after.replace(day=after.day+1, potion_bonus=after.potion_bonus.at[0, HP].add(200))
    assert float(rewards.world(env, after, later, REST, jnp.bool_(True))[1]) == pytest.approx(
        float(rewards.world(baseline, after, later, REST, jnp.bool_(True))[1]))


@pytest.mark.parametrize('summon', [False, True])
def test_kills_pay_on_defeat_and_survive_retreat_revival_and_reset(siege, summon):
    env, _, initial = siege
    before = active_battle(env, initial).replace(summon_battle=jnp.bool_(summon))
    hp = before.hp.at[6].set(0).at[9].set(0)
    after, reward = battle_reward(env, before, before, hp, lost=True)
    assert float(reward) == pytest.approx(10.-.25)
    assert int(after.siege_rewarded_kills) == (1 << 0) | (1 << 3)
    # Reward is not tied to battle-local damage credit or final victory.
    repeated, reward = battle_reward(env, after, after, hp, withdrawal=True)
    assert float(reward) == pytest.approx(-.25)
    revived = repeated.replace(hp=before.hp, enemy_killed_credit=jnp.zeros(6, bool))
    _, reward = battle_reward(env, revived, revived, hp)
    assert float(reward) == pytest.approx(.0025)
    reset, _ = env.reset(jax.random.PRNGKey(19))
    assert int(reset.siege_rewarded_kills) == 0


def test_empty_slots_escape_and_other_armies_do_not_pay(siege):
    env, _, initial = siege
    before = active_battle(env, initial)
    # Empty slots are already zero; escaped units still have positive HP.
    escaped = before.replace(escaped=before.escaped.at[6:].set(True))
    _, reward = battle_reward(env, escaped, escaped, escaped.hp, withdrawal=True)
    assert float(reward) == pytest.approx(-.25)
    other = before.replace(enemy=jnp.int32(0))
    _, reward = battle_reward(env, other, other, other.hp.at[6].set(0))
    assert float(reward) == pytest.approx(.0025)


def test_spell_kill_and_later_battle_share_episode_credit(siege):
    env, baseline, initial = siege
    active = active_battle(env, initial).replace(in_battle=jnp.bool_(False))
    target = env.objective_enemy
    maximum = env.casting.enemy_max_hp(active)[target]
    struck = active.replace(enemy_wounds=active.enemy_wounds.at[target, 0].set(maximum[0]))
    credited, shaped = rewards.world(env, active, struck, 0, jnp.bool_(False))
    _, plain = rewards.world(baseline, active, struck, 0, jnp.bool_(False))
    assert float(shaped-plain) == pytest.approx(5.)
    battle = active_battle(env, credited)
    # The spell victim is already dead at entry; only the newly killed ranger pays.
    _, reward = battle_reward(env, battle, battle, battle.hp.at[9].set(0))
    assert float(reward) == pytest.approx(5.0025)


def test_only_fixed_siege_victory_components_are_tripled(siege):
    env, baseline, initial = siege
    before = active_battle(env, initial)
    target = env.objective_enemy
    final = before.replace(alive=before.alive.at[target].set(False),
        hp=before.hp.at[6:].set(0), in_battle=jnp.bool_(False),
        won=jnp.bool_(True), done=jnp.bool_(True))
    shaped_state, shaped = battle_reward(env, before, final, final.hp, victory=True)
    plain_state, plain = battle_reward(baseline, before, final, final.hp, victory=True)
    shaped += rewards.finish(env, before, shaped_state)[1]
    plain += rewards.finish(baseline, before, plain_state)[1]
    # Fixed baseline 16.5 becomes 49.5; four kills add another 20.
    # Identical XP, survival and episode completion fractions cancel.
    assert float(shaped-plain) == pytest.approx(49.5-16.5+20.)


def test_other_reference_maps_do_not_enable_siege_shaping():
    for name in ('super_last_stand', 'wotans_retribution'):
        assert 'siege_rewards' not in load_map(name)
