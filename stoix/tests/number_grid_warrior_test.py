"""Melee geometry, per-role stats, policy masks and the human-mode contract."""
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stoix.envs.number_grid import (
    MAP, NumberGrid, wrap_wall_action_mask, SHOOT, DEFEND, WAIT, RETREAT,
    CONTINUE, HIT, MISS, VICTORY, WITHDRAW,
)
from stoix.networks.base import FeedForwardActor
from stoix.networks.heads import CategoricalHead
from stoix.networks.torso import MLPTorso


# These regressions retain the saved v10 combat contract. Version 2 has its own tests.
MAP = {**MAP, 'combat_rules_version': 1, 'battle_observation_version': 3}


def battle(**overrides):
    env = NumberGrid(map_config={**MAP, **overrides})
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = env._begin_battle(state.replace(enemy=jnp.int32(6)))
    state = state.replace(actor=jnp.int32(env.warrior_slot), hp=state.hp.at[6:].set(35),
                          priority=jnp.linspace(2., 1., 12))
    return env, state


def expected_targets(slot, enemies, allies):
    """Independent set-based rule oracle, including the rear-row blocker."""
    if slot >= 3 and set(allies) & {0, 1, 2}:
        return set()
    neighbors = ({0, 1}, {0, 1, 2}, {1, 2})[slot % 3]
    for row in ({0, 1, 2}, {3, 4, 5}):
        candidates = set(enemies) & row
        adjacent = {i for i in candidates if i % 3 in neighbors}
        if adjacent:
            return adjacent
        if candidates:
            return candidates
    return set()


@pytest.mark.parametrize('slot', range(6))
def test_every_formation_and_all_death_escape_retreat_combinations(slot):
    env, state = battle(hero_warrior_slot=slot, hero_mage_slot=-1)
    # 0 dead, 1 active, 2 preparing retreat (still blocks), 3 escaped.
    enemy_status = np.array(list(itertools.product(range(4), repeat=6)))
    own_status = np.array(list(itertools.product(range(4), repeat=3)))
    # Cross every enemy configuration with every allied-front occupancy pattern.
    # HP/retreat/escape are checked independently, not collapsed by the oracle.
    def masks(es, own):
        hp = state.hp.at[6:].set(jnp.where(es == 0, 0, 35))
        hp = hp.at[:3].set(jnp.where(own == 0, 0, env.hero_full[:3]))
        hp = hp.at[slot].set(env.warrior_hp)
        escaped = state.escaped.at[6:].set(es == 3).at[:3].set(own == 3).at[slot].set(False)
        retreating = state.retreating.at[6:].set(es == 2).at[:3].set(own == 2).at[slot].set(False)
        return env.action_mask(state.replace(hp=hp, escaped=escaped, retreating=retreating))[SHOOT:DEFEND]
    actual = np.asarray(jax.jit(jax.vmap(jax.vmap(masks, in_axes=(None, 0)), in_axes=(0, None)))(
        jnp.asarray(enemy_status), jnp.asarray(own_status)))
    # Front warriors ignore allied blockers, so expected only differs for rear slots.
    for enemy_index, es in enumerate(enemy_status):
        enemies = set(np.flatnonzero((es == 1) | (es == 2)))
        for own_index, own in enumerate(own_status):
            allies = set(np.flatnonzero((own == 1) | (own == 2))) | {slot}
            targets = expected_targets(slot, enemies, allies)
            assert set(np.flatnonzero(actual[enemy_index, own_index])) == targets


@pytest.mark.parametrize('slot', range(6))
def test_step_rejects_exactly_the_masked_attacks_without_spending_rng(slot):
    env, state = battle(hero_warrior_slot=slot, hero_mage_slot=-1, warrior_accuracy=1.)
    # Includes every occupied/empty arrangement and both blocked/unblocked rear.
    occupancy = (np.arange(64)[:, None] & (1 << np.arange(6))) != 0
    def check(occupied, blocked):
        hp = state.hp.at[6:].set(jnp.where(occupied, 35, 0))
        if slot >= 3:
            hp = hp.at[:3].set(jnp.where(blocked, 45, 0))
        start = state.replace(hp=hp)
        mask = env.action_mask(start)[SHOOT:DEFEND]
        results, _ = jax.vmap(env.step, in_axes=(None, 0))(start, jnp.arange(SHOOT, DEFEND))
        changed = jnp.any(results.hp != hp, axis=-1)
        rng_changed = jnp.any(results.battle_key != start.battle_key, axis=-1)
        return mask, changed, rng_changed, results.player_turns - start.player_turns
    for blocked in (False, True):
        masks, changed, rng_changed, turns = jax.jit(jax.vmap(check, in_axes=(0, None)))(
            jnp.asarray(occupancy), blocked)
        np.testing.assert_array_equal(changed, masks)
        np.testing.assert_array_equal(rng_changed, masks)
        np.testing.assert_array_equal(turns, masks.astype(int))


@pytest.mark.parametrize('accuracy,guarded,damage,event', [(1.,False,25,HIT), (1.,True,13,HIT), (0.,False,0,MISS)])
def test_sword_hits_one_target_or_misses_and_spends_turn(accuracy, guarded, damage, event):
    env, state = battle(warrior_accuracy=accuracy)
    state = state.replace(defended=state.defended.at[7].set(guarded))
    result, _ = jax.jit(env.step)(state, jnp.int32(SHOOT+1))
    expected = np.asarray(state.hp).copy()
    expected[7] -= damage
    np.testing.assert_array_equal(result.hp, expected)
    assert result.last_damage == damage and result.last_event == event and result.last_target == 7
    assert result.turn_phase[1] == 2 and result.player_turns == state.player_turns+1


def test_killing_last_front_enemy_opens_rear_and_archers_mage_keep_all_targets():
    env, state = battle(warrior_accuracy=1.)
    state = state.replace(hp=state.hp.at[6:9].set(jnp.array([0,25,0])))
    result, _ = jax.jit(env.step)(state, jnp.int32(SHOOT+1))
    result = result.replace(actor=jnp.int32(1))
    np.testing.assert_array_equal(env.action_mask(result)[SHOOT:DEFEND], [False]*3+[True]*3)
    for actor in (0, 5):
        np.testing.assert_array_equal(env.action_mask(state.replace(actor=jnp.int32(actor)))[SHOOT:DEFEND],
                                      [False,True,False,True,True,True])


def test_rear_blocked_warrior_can_defend_wait_retreat():
    env, state = battle(hero_warrior_slot=4)
    np.testing.assert_array_equal(np.flatnonzero(env.action_mask(state)), [DEFEND,WAIT,RETREAT])
    waiting, _ = jax.jit(env.step)(state, jnp.int32(WAIT))
    assert waiting.turn_phase[4] == 1
    assert not env.action_mask(waiting.replace(actor=jnp.int32(4)))[WAIT]
    retreating, _ = env.step(state, jnp.int32(RETREAT))
    retreating = retreating.replace(actor=jnp.int32(4))
    np.testing.assert_array_equal(np.flatnonzero(env.action_mask(retreating)), [CONTINUE])


def test_individual_health_normalization_and_full_recovery():
    env, state = battle(warrior_accuracy=1.)
    assert env.max_hp(state)[1] == 100 and np.sum(env.hero_full) == 325
    state = state.replace(hp=state.hp.at[1].set(50).at[6].set(7))
    unit_hp = env.observation(state)[124:172].reshape(12,4)[:,0]
    assert unit_hp[1] == .5 and unit_hp[0] == 1 and float(unit_hp[6]) == pytest.approx(.2)
    victory = state.replace(hp=state.hp.at[0].set(0).at[6:].set(0).at[6].set(25))
    won, _ = jax.jit(env.step)(victory, jnp.int32(SHOOT))
    assert won.last_event == VICTORY
    np.testing.assert_array_equal(won.hp, env.restored_hp)
    # The last remaining hero completes retreat; all fallen comrades return.
    fleeing = state.replace(hp=state.hp.at[:6].set(0).at[1].set(1),
                            retreating=state.retreating.at[1].set(True))
    escaped, _ = env.step(fleeing, jnp.int32(CONTINUE))
    assert escaped.last_event == WITHDRAW and not escaped.done
    np.testing.assert_array_equal(escaped.hp, env.restored_hp)


def test_initiative_50_is_scaled_on_battle_start_and_each_round():
    env, state = battle()
    draws = jnp.full(13, .5)
    begun = jax.jit(env._begin_battle)(state, state.battle_key, draws)
    assert begun.priority[1] == 1.25 and begun.priority[0] == 1.5
    last = state.replace(turn_phase=jnp.full(12,2).at[1].set(0))
    result, _ = jax.jit(env._battle_step)(last, jnp.int32(DEFEND), last.battle_key, draws)
    assert result.round == last.round+1 and result.priority[1] == 1.25
    # Stronger random priority can still let the warrior act before an archer.
    draws = jnp.zeros(13).at[2].set(.999)
    assert env._begin_battle(state, state.battle_key, draws).actor == 1


def test_masked_warrior_policy_sample_argmax_and_ppo_gradient():
    env, state = battle(hero_warrior_slot=0)
    wrapped = wrap_wall_action_mask(env)
    _, ts = wrapped.step(state, jnp.int32(-1))  # unchanged battle, wrapped observation
    obs = jax.tree.map(lambda x: jnp.repeat(x[None], 64, axis=0), ts.observation)
    policy = FeedForwardActor(torso=MLPTorso(layer_sizes=[32]), action_head=CategoricalHead(18))
    params = policy.init(jax.random.PRNGKey(0), obs)
    dist = policy.apply(params, obs)
    probabilities = np.asarray(dist.probs_parameter())
    mask = np.asarray(obs['action_mask'])
    assert np.all(probabilities[~mask] == 0)
    actions = dist.sample(64, seed=jax.random.PRNGKey(1))
    assert np.all(np.take_along_axis(mask[None], np.asarray(actions)[...,None], axis=-1))
    assert np.all(np.take_along_axis(mask, np.asarray(dist.mode())[:,None], axis=-1))
    old_log = jax.lax.stop_gradient(dist.log_prob(actions[0]))
    def loss(p):
        updated = policy.apply(p, obs)
        ratio = jnp.exp(updated.log_prob(actions[0])-old_log)
        return -jnp.mean(jnp.minimum(ratio,jnp.clip(ratio,.8,1.2))+.01*updated.entropy())
    value, gradients = jax.jit(jax.value_and_grad(loss))(params)
    assert np.isfinite(value) and all(np.isfinite(x).all() for x in jax.tree.leaves(gradients))


def test_human_service_uses_identical_mask_and_rejects_unreachable_target():
    from serve_number_grid import GameService
    service = GameService()
    session = service.create(42)['session']
    env = service.env
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = env._begin_battle(state.replace(enemy=jnp.int32(11)))
    state = state.replace(actor=jnp.int32(env.warrior_slot))
    service.sessions[session] = (env.construction.faction, state, 0.)
    snapshot = service.snapshot(env, state, 0.)
    np.testing.assert_array_equal(snapshot['action_mask'], env.action_mask(state))
    with pytest.raises(ValueError, match='недоступно'):
        service.act(session, SHOOT+3)
    assert service.sessions[session][1] is state
    direct, ts = env.step(state, jnp.int32(SHOOT))
    first = service.act(session, SHOOT)['events'][0]
    assert first == {**service.snapshot(env, direct, float(ts.reward)), 'reward': float(ts.reward)}


@pytest.mark.parametrize('overrides', [
    {'hero_warrior_slot':5}, {'hero_warrior_slot':True}, {'hero_warrior_slot':6},
    {'hero_warrior_slot':-2}, {'warrior_hp':0}, {'warrior_damage':0},
    {'warrior_accuracy':float('nan')}, {'warrior_accuracy':1.1},
    {'warrior_initiative':0}, {'battle_observation_version':2},
])
def test_bad_warrior_configuration_is_rejected(overrides):
    with pytest.raises(ValueError, match='[Ww]arrior'):
        NumberGrid(map_config={**MAP, **overrides})


def test_saved_mage_map_has_old_stats_observation_and_unrestricted_attacks():
    path = Path(__file__).resolve().parents[2] / 'maps/number_grid-24x24-v8-mage-24-squads.json'
    env = NumberGrid(map_config=json.loads(path.read_text()))
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = env._begin_battle(state.replace(enemy=jnp.int32(6)))
    state = state.replace(actor=jnp.int32(1), hp=state.hp.at[6:].set(35))
    assert env.warrior_slot == -1 and env.observation_version == 2
    np.testing.assert_array_equal(env.hero_full, [45]*6)
    assert env.observation(state)[124+6*4] == jnp.float32(35/45)
    assert np.all(env.action_mask(state)[SHOOT:DEFEND])
