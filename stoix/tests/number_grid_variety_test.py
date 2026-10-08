"""Large-unit footprints and single-target healing from the Python reference (CUDA)."""
import copy

import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, MAP, SHOOT, CONTINUE, DEFEND, HEAL, GUARD
from stoix.envs.number_grid_combat import UNITS, HEALER


@pytest.fixture(scope='module')
def env(current_game):
    return current_game[0]


def battle(env, enemy):
    state, _ = env.reset(jax.random.PRNGKey(42))
    return env._begin_battle(state.replace(enemy=jnp.int32(enemy)))


def stack(states):
    return jax.tree.map(lambda *values: jnp.stack(values), *states)


def test_48x48_map_with_41_squads_and_reference_profiles(env):
    assert env.size == 48 and env.num_opponents == 41
    assert len(set(map(tuple, MAP['opponent_positions']))) == 41
    assert MAP['enemy_rosters'][24:27] == [
        ['orc','orc',None,None,None,None],
        ['goblin','orc','goblin',None,None,None],
        ['goblin','goblin','goblin',None,'goblin_archer',None]]
    assert sum('titan' in row for row in MAP['enemy_rosters'][:24]) == 6
    assert sum('acolyte' in row for row in MAP['enemy_rosters'][:24]) == 8
    assert all(0 < r < 47 and 0 < c < 47 for r, c in MAP['opponent_positions'])
    assert MAP['agent_position'] not in MAP['opponent_positions']
    assert sum(any(key in ('orc', 'goblin', 'goblin_archer') for key in row)
               for row in MAP['enemy_rosters']) == 3
    assert all(any((r >= 24) == south and (c >= 24) == east
                   for r, c in MAP['opponent_positions'])
               for south in (False, True) for east in (False, True))
    fields = ('max_hp','damage','accuracy','armor','initiative','exp_kill','exp_required','size')
    expected = dict(titan=(250,60,80,0,50,120,475,2), acolyte=(50,20,100,0,10,20,80,1),
                    orc=(200,55,80,0,40,90,700,1), goblin=(50,15,80,0,30,5,50,1),
                    goblin_archer=(40,15,80,0,50,10,75,1))
    for key, values in expected.items():
        assert tuple(UNITS[key][field] for field in fields) == values
        assert UNITS[key]['level'] == 1
    for roster, count in zip(MAP['enemy_rosters'], MAP['enemy_units']):
        assert sum(key is not None for key in roster) == count
        for slot, key in enumerate(roster):
            if key and UNITS[key].get('size', 1) == 2:
                assert slot < 3 and roster[slot+3] is None
    initial, ts = jax.jit(env.reset)(jax.random.PRNGKey(1))
    assert ts.observation.shape == (497,) and env.num_actions == 80
    chex.assert_trees_all_equal(ts.observation[-12:], jnp.array([.5]*5+[0]*7))
    state = battle(env, 21)
    obs = jax.jit(env.observation)(state)
    chex.assert_trees_all_equal(obs[-6:], jnp.array([1,.5,1,0,.5,0]))
    traits_start = 112+5*env.num_opponents
    assert obs[traits_start+10*4] == 1  # role HEALER normalized by 4
    assert env.unit_traits(state)[10, 0] == HEALER
    assert jnp.all(jnp.isfinite(obs)) and initial.hp.shape == (12,)


def test_large_unit_formation_rejects_overlap_and_rear_anchors():
    for roster in ([None,None,None,'titan',None,None],
                   ['titan',None,None,'archer',None,None],
                   [None,'titan',None,None,'acolyte',None]):
        config = copy.deepcopy(MAP)
        config['enemy_rosters'][0] = roster
        config['enemy_units'][0] = sum(key is not None for key in roster)
        with pytest.raises(ValueError, match='Large units'):
            NumberGrid(map_config=config)
    config = copy.deepcopy(MAP)
    config['hero_roster'] = ['titan','titan','titan',None,None,None]
    config['hero_units'] = 3
    allies = NumberGrid(map_config=config)
    chex.assert_trees_all_equal(allies.hero_full, jnp.array([250]*3+[0]*3))
    for size in (0, 3, True, 1.5):
        with pytest.raises(ValueError, match='size'):
            NumberGrid(map_config={**MAP, 'hero_combat_stats':[dict(size=size)]+[{}]*4})


def test_titan_has_one_health_pool_action_target_and_kill_reward(env):
    state = battle(env, 21).replace(actor=jnp.int32(3))
    mask = jax.jit(env.action_mask)(state)
    chex.assert_trees_all_equal(mask[SHOOT:SHOOT+6], jnp.array([1,1,1,0,1,0], bool))
    attack = jax.jit(env._battle_step)
    result, _ = attack(state, jnp.int32(SHOOT), state.battle_key, jnp.zeros(36))
    chex.assert_trees_all_equal(result.hp[6:], jnp.array([235,85,235,0,35,0]))
    assert result.last_damage == 60  # four fighters, not six occupied cells
    assert result.actor not in (9,11)
    dying = state.replace(hp=state.hp.at[6].set(15))
    killed, _ = attack(dying, jnp.int32(SHOOT), state.battle_key, jnp.zeros(36))
    assert killed.battle_xp[1] == 120 and killed.hp[9] == 0
    again, _ = attack(killed.replace(actor=jnp.int32(4)), jnp.int32(SHOOT+2), state.battle_key, jnp.zeros(36))
    assert again.battle_xp[1] == 120
    titan = state.replace(actor=jnp.int32(6))
    acted, _ = attack(titan, jnp.int32(CONTINUE), state.battle_key, jnp.zeros(36))
    assert acted.turn_phase[6] == 2 and acted.actor not in (6,9,11)
    assert jnp.count_nonzero((acted.hp > 0) & (acted.turn_phase < 2)) == 8
    # Pending retreat occupies the front; completed retreat frees access to rear allies.
    lone = battle(env, 5).replace(actor=jnp.int32(0))
    fleeing = lone.replace(retreating=lone.retreating.at[7].set(True))
    escaped = fleeing.replace(escaped=fleeing.escaped.at[7].set(True))
    masks = jax.jit(jax.vmap(env.action_mask))(stack([fleeing, escaped]))[:, SHOOT:SHOOT+6]
    chex.assert_trees_all_equal(masks, jnp.array([[0,1,0,0,0,0],[0,0,0,1,0,0]], bool))


def test_enemy_heals_lowest_absolute_hp_self_caps_or_defends(env):
    state = battle(env, 5).replace(actor=jnp.int32(11))  # Titan 7, archer 9, acolyte 11
    cases = [state.replace(hp=state.hp.at[7].set(100).at[9].set(10).at[11].set(40)),
             state.replace(hp=state.hp.at[9].set(44)),
             state.replace(hp=state.hp.at[11].set(30)),
             state.replace(hp=state.hp.at[9].set(0)),
             state.replace(hp=state.hp.at[9].set(10), escaped=state.escaped.at[9].set(True)),
             state.replace(hp=state.hp.at[7].set(50).at[9].set(44))]
    states = stack(cases)
    result, _ = jax.jit(jax.vmap(env._battle_step, in_axes=(0,None,0,None)))(
        states, jnp.int32(CONTINUE), states.battle_key, jnp.full(36,.999))
    chex.assert_trees_all_equal(result.last_event, jnp.array([HEAL,HEAL,HEAL,GUARD,GUARD,HEAL]))
    chex.assert_trees_all_equal(result.last_damage, jnp.array([20,1,20,0,0,1]))
    chex.assert_trees_all_equal(result.last_target, jnp.array([9,9,11,-1,-1,9]))
    chex.assert_trees_all_equal(result.hp[:, :6], states.hp[:, :6])
    assert result.hp[3,9] == 0 and result.hp[4,9] == 10
    assert jnp.all(result.battle_xp == 0) and jnp.all(result.wards_used == 0)
    assert jnp.all(result.turn_phase[:,11] == 2)
    # Equal HP uses the existing random tie keys, independently of accuracy draws.
    tied = state.replace(hp=state.hp.at[9].set(20).at[11].set(20))
    choose = jax.jit(env._enemy_action)
    assert choose(tied, jnp.array([0,0,0,.1,0,.9])) == SHOOT+3
    assert choose(tied, jnp.array([0,0,0,.9,0,.1])) == SHOOT+5


def test_player_healing_mask_step_and_protections_agree():
    config = copy.deepcopy(MAP)
    config['hero_roster'][4] = 'acolyte'
    config['hero_combat_stats'] = [dict(armor=90, immunities=['life'], protections=['life']),
                                   {}, {}, {}, dict(accuracy=0)]
    env = NumberGrid(map_config=config)
    state = battle(env, 0).replace(actor=jnp.int32(4))
    state = state.replace(hp=state.hp.at[0].set(110).at[1].set(150).at[2].set(0).at[3].set(20).at[4].set(40),
                          escaped=state.escaped.at[3].set(True), defended=state.defended.at[0].set(True))
    mask = jax.jit(env.action_mask)(state)
    chex.assert_trees_all_equal(mask[SHOOT:SHOOT+6], jnp.array([1,1,0,0,1,0],bool))
    states = stack([state]*6)
    healed, _ = jax.jit(jax.vmap(env.step))(states, SHOOT+jnp.arange(6,dtype=jnp.int32))
    assert healed.hp[0,0] == 120 and healed.last_damage[0] == 10
    assert healed.hp[4,4] == 50 and healed.last_damage[4] == 10
    assert healed.last_event[1] == HEAL and healed.last_damage[1] == 0  # full health is a legal target
    assert jnp.all(healed.wards_used == 0) and jnp.all(healed.last_immune == 0)
    chex.assert_trees_all_equal(healed.hp[:,6:], states.hp[:,6:])
    for invalid in (2,3,5):
        chex.assert_trees_all_equal(healed.hp[invalid], state.hp)
        chex.assert_trees_all_equal(healed.battle_key[invalid], state.battle_key)
    assert jnp.all(healed.turn_phase[jnp.array([0,1,4]),4] == 2)


def test_new_units_level_growth_and_enemy_healer_cap(env):
    keys = ['titan','orc','goblin','goblin_archer','acolyte']
    ids = jnp.array([env.progression.ids[key] for key in keys])
    stats = jax.jit(env.progression.stats)(ids, jnp.full(5,2,jnp.int32))
    chex.assert_trees_all_equal(stats, jnp.array([[275,66,81,0,50],[220,61,81,0,40],
        [55,17,81,0,30],[45,17,81,0,50],[55,22,100,0,10]],jnp.float32))
    state = battle(env, 5)
    # Enemies have no capital: acolyte cannot change form and caps at 79/80 XP.
    state = state.replace(hp=state.hp.at[:6].set(0))
    result = jax.jit(env.progression.finish)(state, state.hp, state.escaped,
        jnp.bool_(False), jnp.bool_(True), jnp.bool_(False), jnp.array([10000,0],jnp.int32))
    ids, levels, xp = result[:3]
    assert ids[11] == state.unit_ids[11] and levels[11] == 1 and xp[11] == 79
    assert levels[7] == 2 and xp[7] == 0  # terminal Titan grows numerically
    assert result[4][7] == 275
