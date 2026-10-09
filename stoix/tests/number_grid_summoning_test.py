"""CUDA regressions for catalog coverage, zero-round copies and linked summons."""
from copy import deepcopy
import json
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, MAP, SHOOT, DEFEND, WAIT, CONTINUE
from stoix.envs.number_grid_combat import UNITS
from stoix.envs.number_grid_summoning import COPY_ALLY, COPIED, SUMMONED
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state, enemy_roster_state


@pytest.fixture(scope='module')
def summon_game():
    config = deepcopy(MAP)
    config['enemy_rosters'][0] = ['elementalist','sage','occultist','occultmaster','lyf','laclaan']
    config['enemy_units'][0] = 6
    env = NumberGrid(map_config=config)
    state,_ = compiled_method(env,'reset')(jax.random.PRNGKey(42))
    return env,state


def battle(game, heroes, enemies):
    env,initial = game
    state = hero_roster_state(env,initial,heroes)
    state = env._begin_battle(state.replace(enemy=jnp.int32(0)))
    state = enemy_roster_state(env,state,enemies)
    return state.replace(native_ids=state.unit_ids,native_levels=state.unit_levels,native_xp=state.unit_xp)


def act(env,state,action):
    return compiled_method(env,'_battle_step')(state,jnp.int32(action),state.battle_key,jnp.zeros(env.random_size))[0]


def test_complete_catalog_is_traceable_and_map_does_not_add_new_types():
    manifest = json.loads((Path(__file__).parents[2]/'docs/validation/unit-catalog.json').read_text())
    assert len(UNITS) == manifest['unit_count'] == 356
    assert len({r['game_id'] for r in UNITS.values()}) == 356
    assert len({r['unit_type'] for r in UNITS.values()}) == 52
    assert not manifest['differences']
    for row in UNITS.values():
        raw = row['game_data']
        assert row['max_hp'] == int(raw['HIT_POINT'])
        assert row['armor'] == int(raw['ARMOR'])
        assert row['exp_kill'] == int(raw['XP_KILLED'])
        assert row['size'] == (1 if raw['SIZE_SMALL'] == 'T' else 2)
        assert row['neutral'] == (row['faction'] == 'neutral')
        assert not row['upgrade_unavailable_reason']
        assert all(key in UNITS for key in row['upgrades']+row['summon_pool'])
    excluded = {'Doppelganger','Summoner','Occultmaster','Lyf','Laclaan'}
    assert all(UNITS[key]['unit_type'] not in excluded for roster in MAP['enemy_rosters'] for key in roster if key)
    # Reference unit_dynamic_stats.py moves these secondary heals into the
    # runtime primary amount. Aleman's secondary amount grows but shatter
    # deliberately does not use that amount as its armor-reduction strength.
    assert [UNITS[k]['growth']['early'][1] for k in ('sundancer','sylfid','deva_roshi')] == [6,7,5]
    assert UNITS['aleman']['growth']['secondary_early'] == [30,1]


def test_zero_round_copy_mask_stats_hp_and_normal_turn(summon_game):
    env,_ = summon_game
    state = battle(summon_game,['doppelganger','duke',None,None,None,None],['squire',None,None,None,None,None])
    state = state.replace(actor=jnp.int32(0),hp=state.hp.at[0].set(60).at[1].set(100),
                          preparation=jnp.array([True]+[False]*11),round=jnp.int32(0))
    mask = compiled_method(env,'action_mask')(state)
    assert mask[SHOOT] and mask[COPY_ALLY+1] and mask[DEFEND]
    assert not mask[COPY_ALLY] and not mask[WAIT]
    assert not mask[SHOOT+1]
    invalid,_ = compiled_method(env,'step')(state,jnp.int32(COPY_ALLY))
    chex.assert_trees_all_equal(invalid.hp,state.hp)
    result = act(env,state,COPY_ALLY+1)
    assert result.copied[0] and result.last_event == COPIED and result.round == 1
    assert result.hp[0] == 50 and env.max_hp(result)[0] == 150
    assert result.unit_ids[0] == result.unit_ids[1] and result.native_ids[0] == env.progression.ids['doppelganger']
    assert result.turn_phase[0] == 0 and not result.defended[0]
    assert env.unit_experience(result)[0,1] == UNITS['doppelganger']['exp_kill']
    assert not env.action_mask(result.replace(actor=jnp.int32(0)))[COPY_ALLY+1]
    limited = act(env,state.replace(step_count=jnp.int32(env.max_steps-1)),COPY_ALLY+1)
    assert limited.done and not limited.copied[0]
    assert limited.unit_ids[0] == state.unit_ids[0] and limited.hp[0] == 40


def test_copy_exclusions_and_persistent_stats(summon_game):
    env,_ = summon_game
    state = battle(summon_game,['doppelganger','doppelganger','duke',None,None,None],['titan','squire',None,None,None,None])
    state = state.replace(actor=jnp.int32(0),round=jnp.int32(0),
        primary_override=state.primary_override.at[2].set(200),
        initiative_override=state.initiative_override.at[2].set(1),
        armor_shreds=state.armor_shreds.at[2].set(2))
    mask = env.action_mask(state)
    assert not mask[SHOOT] and not mask[COPY_ALLY+1] and mask[COPY_ALLY+2]
    disguised = state.replace(imp=state.imp.at[1].set(True))
    assert not env.action_mask(disguised)[COPY_ALLY+1]
    result = act(env,state,COPY_ALLY+2)
    chex.assert_trees_all_equal(result.copy_stats[0],env.progression.stats(state.unit_ids,state.unit_levels)[2])
    chex.assert_trees_all_equal(result.hp[1:],state.hp[1:])
    assert result.primary_override[0] == -1 and result.initiative_override[0] == -1
    wolf = state.replace(unit_ids=state.unit_ids.at[2].set(env.progression.ids['wolf_lord']),
        unit_levels=state.unit_levels.at[2].set(4),hp=state.hp.at[2].set(275),
        fenrir=state.fenrir.at[2].set(True))
    first = act(env,wolf,COPY_ALLY+2)
    assert env.unit_sizes(first)[0] == 1 and first.copy_stats[0,0] == 275
    # A copied Fenrir remains a small Doppelganger and can itself be copied.
    assert env.action_mask(first.replace(actor=jnp.int32(1)))[COPY_ALLY]


def test_single_summon_corpse_empty_slot_owner_death_and_no_xp(summon_game):
    env,_ = summon_game
    state = battle(summon_game,['squire',None,None,'elementalist',None,None],['squire',None,None,None,None,None])
    state = state.replace(actor=jnp.int32(3),round=jnp.int32(1),preparation=jnp.zeros(12,bool),hp=state.hp.at[0].set(0),
        poison_source=state.poison_source.at[6].set(0),poison_turns=state.poison_turns.at[6].set(4),
        poison_damage=state.poison_damage.at[6].set(7))
    mask = env.action_mask(state)
    assert mask[SHOOT] and mask[SHOOT+1] and not mask[SHOOT+3]
    result = act(env,state,SHOOT)
    assert result.last_event == SUMMONED and result.summon_owner[0] == 3
    assert result.hp[0] == 100 and result.turn_phase[0] == 2
    assert result.native_ids[0] == env.progression.ids['squire']
    assert not result.defended[3] and env.unit_experience(result)[0,1] == 0
    assert result.poison_source[6] == -1 and result.poison_turns[6] > 0 and result.poison_damage[6] == 7
    dead = compiled_method(env,'_linked_summon_hp')(result,result.hp.at[3].set(0))
    assert dead[0] == 0
    restored,hp = compiled_method(env,'_restore_roster')(result,result.hp,jnp.bool_(True))
    assert hp[0] == 0 and restored.unit_ids[0] == env.progression.ids['squire']
    assert jnp.all(restored.summon_owner == -1)


@pytest.mark.parametrize('caster',['occultmaster','lyf','laclaan'])
def test_mass_summons_respect_large_pairs_and_wait_until_next_round(summon_game,caster):
    env,_ = summon_game
    state = battle(summon_game,[None,'squire',None,None,caster,None],['squire',None,None,None,None,None])
    state = state.replace(actor=jnp.int32(4),round=jnp.int32(1),preparation=jnp.zeros(12,bool))
    result = act(env,state,SHOOT)
    assert result.last_event == SUMMONED
    assert result.summon_owner[0] == result.summon_owner[2] == 4
    assert env.unit_sizes(result)[0] == env.unit_sizes(result)[2] == 2
    assert result.hp[3] == result.hp[5] == 0
    assert not jnp.any(env._summon_targets(result.replace(actor=jnp.int32(4))))
    assert result.turn_phase[0] == result.turn_phase[2] == (0 if caster == 'laclaan' else 2)


def test_copied_summoner_keeps_form_identity_and_reverts_before_xp(summon_game):
    env,_ = summon_game
    state = battle(summon_game,['doppelganger',None,None,'elementalist',None,None],['squire',None,None,None,None,None])
    state = state.replace(actor=jnp.int32(0),round=jnp.int32(0),preparation=jnp.array([True]+[False]*11))
    copied = act(env,state,COPY_ALLY+3)
    summoned = act(env,copied.replace(actor=jnp.int32(0)),SHOOT+1)
    assert summoned.summon_owner[1] == 0 and summoned.hp[1] == 100
    dying = compiled_method(env,'_linked_summon_hp')(summoned,summoned.hp.at[0].set(0))
    assert dying[1] == 0
    restored,hp = compiled_method(env,'_restore_roster')(summoned,summoned.hp.at[0].set(50),jnp.bool_(True))
    assert restored.unit_ids[0] == env.progression.ids['doppelganger'] and hp[0] == 63
    assert not jnp.any(restored.copied) and hp[1] == 0


def test_copy_kill_rewards_native_xp_and_vmap_matches_scalar(summon_game):
    env,_ = summon_game
    state = battle(summon_game,['doppelganger',None,None,'duke',None,None],['squire',None,None,None,None,None])
    state = state.replace(actor=jnp.int32(0),round=jnp.int32(0),preparation=jnp.array([True]+[False]*11))
    copied = act(env,state,COPY_ALLY+3)
    # Killing a 60-XP Duke copy pays the Doppelganger's 120 XP.
    victim = copied.replace(actor=jnp.int32(6),hp=copied.hp.at[0].set(1).at[3].set(0),round=jnp.int32(1))
    result = act(env,victim,CONTINUE)
    assert result.battle_xp[0] == UNITS['doppelganger']['exp_kill']
    assert result.unit_ids[0] == env.progression.ids['doppelganger'] and result.hp[0] == 0
    states = jax.tree.map(lambda a,b:jnp.stack((a,b)),state,victim)
    actions = jnp.array([COPY_ALLY+3,CONTINUE],jnp.int32)
    results,_ = compiled_method(env,'_battle_step',batched=True)(states,actions,states.battle_key,jnp.zeros((2,env.random_size)))
    chex.assert_trees_all_equal(jax.tree.map(lambda x:x[0],results),copied)
    chex.assert_trees_all_equal(jax.tree.map(lambda x:x[1],results),result)


def test_linked_queue_skips_summon_when_its_owner_dies_from_poison(summon_game):
    env,_ = summon_game
    state = battle(summon_game,[None,'duke',None,'elementalist',None,None],['squire',None,None,None,None,None])
    state = state.replace(actor=jnp.int32(3),round=jnp.int32(1),preparation=jnp.zeros(12,bool))
    spawned = act(env,state,SHOOT)
    waiting = spawned.replace(actor=jnp.int32(6),hp=spawned.hp.at[3].set(1),
        turn_phase=jnp.array([0,2,2,0,2,2,0,2,2,2,2,2],jnp.int32),
        priority=jnp.array([80.,50,0,90,0,0,100,0,0,0,0,0]),
        activation_done=jnp.zeros(12,bool),poison_turns=spawned.poison_turns.at[3].set(1),
        poison_damage=spawned.poison_damage.at[3].set(1),
        paralyzed=spawned.paralyzed.at[6].set(True))
    following = act(env,waiting,CONTINUE)
    assert following.hp[3] == following.hp[0] == 0
    assert not following.done and following.round == 2
    assert following.hp[following.actor] > 0 and following.turn_phase[following.actor] == 0


def test_dark_laclaan_is_copyable_but_capital_guard_is_not(summon_game):
    env,_ = summon_game
    guard = next(k for k,r in UNITS.items() if r['game_data']['UNIT_CAT'] == '8' and r['size'] == 1)
    state = battle(summon_game,['doppelganger',None,None,None,None,None],['laclaan',guard,None,None,None,None])
    state = state.replace(actor=jnp.int32(0),round=jnp.int32(0),preparation=jnp.array([True]+[False]*11))
    mask = env.action_mask(state)
    assert mask[SHOOT] and not mask[SHOOT+1]
    copied = act(env,state,SHOOT)
    assert copied.copied[0] and env._actor_summons(copied.replace(actor=jnp.int32(0))) == 2
