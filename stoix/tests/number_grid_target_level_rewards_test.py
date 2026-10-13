"""Target-only level rewards, requested 2026-10-14; not original game rules.

Extended CUDA regressions. Run only on explicit request through scripts/run_gpu.sh.
See docs/DEFAULT_MAP.md and docs/REFERENCE_CAMPAIGNS.md.
"""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import DEFEND, NumberGrid, load_map
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state


@pytest.fixture(scope='module')
def target_game():
    env = NumberGrid(map_config=load_map('default-blue-dragon'))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    return env, state


def near_enemy(env, state, enemy):
    position = state.enemy_positions[enemy]
    blocked = env.territory.blocked_grid.reshape(env.size, env.size)
    direction, offset = next((i, d) for i, d in enumerate(env.directions.tolist())
        if not bool(blocked[position[0]-d[0], position[1]-d[1]]))
    return state.replace(position=position-jnp.array(offset, jnp.int32),
        movement_points=jnp.int32(100), alive=state.alive.at[enemy].set(True),
        scenario_spawned=state.scenario_spawned.at[enemy].set(True)), jnp.int32(direction)


def enter(env, before, action):
    return compiled_method(env, '_world_step')(before, action,
        jax.random.PRNGKey(19), jnp.full(env.random_size, .5))


def finish(env, battle, outcome='victory'):
    if outcome == 'victory':
        battle = battle.replace(hp=battle.hp.at[6:].set(0))
    elif outcome == 'retreat':
        battle = battle.replace(escaped=battle.escaped.at[:6].set(True))
    else:
        battle = battle.replace(hp=battle.hp.at[:6].set(0))
    battle = battle.replace(actor=jnp.int32(1), round=jnp.int32(1),
        post_victory=jnp.int32(1 if outcome == 'victory' else 2),
        pending_healers=jnp.zeros(12, bool))
    return compiled_method(env, '_battle_step')(battle, jnp.int32(DEFEND),
        jax.random.PRNGKey(20), jnp.full(env.random_size, .5))


def developed_party(env, initial):
    return hero_roster_state(env, initial,
        ['defender_of_faith', 'duke', None, 'white_mage', None, None])


def test_target_snapshot_is_observable_and_paid_only_on_victory(target_game):
    env, initial = target_game
    target = env.objective_enemy
    assert env.army_victory_level_reward == 5.
    assert env.army_victory_level_objective_only
    assert jnp.flatnonzero(env.army_victory_level_targets).tolist() == [target]
    base = developed_party(env, initial)
    # Wounded level-5 mercenary + level-1 hero; dead and empty slots add nothing.
    base = base.replace(hp=base.hp.at[0].set(1).at[3].set(0),
        unit_levels=base.unit_levels.at[2].set(99),
        potion_bonus=base.potion_bonus.at[0, 0].add(100).at[3, 0].add(500))
    before, action = near_enemy(env, base, target)
    arrived, entry_reward = enter(env, before, action)
    claimed, plain_entry = enter(env, before.replace(
        army_victory_rewarded=before.army_victory_rewarded.at[target].set(True)), action)
    assert arrived.in_battle and int(arrived.enemy) == target
    assert float(entry_reward) == pytest.approx(float(plain_entry))
    assert int(arrived.army_battle_level_strength) == 4
    assert not jnp.any(arrived.army_victory_rewarded)
    observation = compiled_method(env, 'observation')(arrived)
    assert observation.shape == (env.observation_size,)
    assert float(observation[-1]) == pytest.approx(.04)
    won, reward = finish(env, arrived)
    _, plain = finish(env, claimed)
    assert won.won and won.done and not won.alive[target]
    assert float(reward-plain) == pytest.approx(20.)
    assert jnp.flatnonzero(won.army_victory_rewarded).tolist() == [target]
    assert int(won.army_battle_level_strength) == 0
    # HP changes, casualties and level gains after entry cannot change the snapshot.
    changed = arrived.replace(hp=arrived.hp.at[0].set(0),
        unit_levels=arrived.unit_levels.at[1].add(3),
        potion_bonus=arrived.potion_bonus.at[1, 0].add(500))
    _, changed_reward = finish(env, changed)
    _, changed_plain = finish(env, changed.replace(army_victory_rewarded=won.army_victory_rewarded))
    assert float(changed_reward-changed_plain) == pytest.approx(20.)
    _, repeated = finish(env, claimed.replace(army_battle_level_strength=jnp.int32(100)))
    assert float(repeated) == pytest.approx(float(plain))
    reset, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(43))
    assert not jnp.any(reset.army_victory_rewarded)
    assert int(reset.army_battle_level_strength) == 0


@pytest.mark.parametrize('outcome', ['retreat', 'defeat'])
def test_unsuccessful_target_battle_discards_snapshot(target_game, outcome):
    env, initial = target_game
    before, action = near_enemy(env, developed_party(env, initial), env.objective_enemy)
    arrived, _ = enter(env, before, action)
    after, reward = finish(env, arrived, outcome)
    _, plain = finish(env, arrived.replace(army_battle_level_strength=jnp.int32(0)), outcome)
    assert float(reward) == pytest.approx(float(plain))
    assert int(after.army_battle_level_strength) == 0
    assert not jnp.any(after.army_victory_rewarded)
    if outcome == 'retreat':
        stronger = after.replace(unit_levels=after.unit_levels.at[1].add(2))
        before, action = near_enemy(env, stronger, env.objective_enemy)
        retried, _ = enter(env, before, action)
        assert int(retried.army_battle_level_strength) == int(arrived.army_battle_level_strength)+2
        _, reward = finish(env, retried)
        _, plain = finish(env, retried.replace(army_battle_level_strength=jnp.int32(0)))
        assert float(reward-plain) == pytest.approx(5.*int(retried.army_battle_level_strength))


def test_other_armies_and_invalid_moves_cannot_earn_target_bonus(target_game):
    env, initial = target_game
    before, action = near_enemy(env, developed_party(env, initial), 0)
    arrived, _ = enter(env, before, action)
    assert arrived.in_battle and int(arrived.enemy) == 0
    assert int(arrived.army_battle_level_strength) == 0
    # The payment boundary also checks the target, even with a stale snapshot.
    after, reward = finish(env, arrived.replace(army_battle_level_strength=jnp.int32(99)))
    _, plain = finish(env, arrived)
    assert float(reward) == pytest.approx(float(plain))
    assert not jnp.any(after.army_victory_rewarded)
    near, action = near_enemy(env, developed_party(env, initial), env.objective_enemy)
    invalid, _ = compiled_method(env, 'step')(near.replace(movement_points=jnp.int32(0)), action)
    assert not invalid.in_battle and not jnp.any(invalid.army_victory_rewarded)
    assert int(invalid.army_battle_level_strength) == 0


def test_summoned_army_cannot_earn_target_level_bonus(target_game):
    env, initial = target_game
    near, _ = near_enemy(env, initial, env.objective_enemy)
    spell_id = 'g000ss0117'
    spell = next(i for i, row in enumerate(env.casting.rows) if row['id'] == spell_id)
    near = near.replace(learned_spells=jnp.uint32(1 << spell), mana=jnp.full(5, 5000, jnp.int32))
    battle, _ = enter(env, near, jnp.int32(env.casting.start+spell))
    assert battle.summon_battle and int(battle.enemy) == env.objective_enemy
    assert int(battle.army_battle_level_strength) == 0
    after, reward = finish(env, battle.replace(army_battle_level_strength=jnp.int32(99)))
    _, plain = finish(env, battle)
    assert float(reward) == pytest.approx(float(plain))
    assert not jnp.any(after.army_victory_rewarded)


@pytest.mark.parametrize('value', [1, 'true', None])
def test_invalid_target_scope_is_rejected(value):
    with pytest.raises(ValueError, match='army_victory_level_objective_only'):
        NumberGrid(map_config={**load_map('default-blue-dragon'), 'army_victory_level_objective_only': value})


def test_target_scope_requires_an_objective_and_enabled_reward():
    with pytest.raises(ValueError, match='requires objective_enemy'):
        NumberGrid(map_config={**load_map('trade_train'), 'army_victory_level_objective_only': True})
    with pytest.raises(ValueError, match='requires objective_enemy'):
        NumberGrid(map_config={**load_map('default-blue-dragon'), 'army_victory_level_reward': 0.})


def test_unrelated_maps_keep_level_rewards_disabled():
    for name in ('default', 'default-green-dragon', 'super_last_stand', 'wotans_retribution', 'siege_train'):
        assert not load_map(name).get('army_victory_level_reward', 0.)
    assert not load_map('trade_train').get('army_victory_level_objective_only', False)


@pytest.mark.parametrize('levels,expected', [([1, 1, 1, 1], 0.), ([2, 2, 3, 1], 20.),
                                          ([4, 4, 4, 4], 60.)])
def test_target_bonus_grows_with_entry_levels(target_game, levels, expected):
    env, initial = target_game
    base = hero_roster_state(env, initial, ['possessed', 'duke', 'possessed', 'cultist', None, None])
    base = base.replace(unit_levels=base.unit_levels.at[:4].set(jnp.array(levels)))
    before, action = near_enemy(env, base, env.objective_enemy)
    battle, _ = enter(env, before, action)
    assert int(battle.army_battle_level_strength) == expected/5.
    won, reward = finish(env, battle)
    _, plain = finish(env, battle.replace(army_victory_rewarded=won.army_victory_rewarded))
    assert float(reward-plain) == pytest.approx(expected)
    assert bool(won.army_victory_rewarded[env.objective_enemy])
