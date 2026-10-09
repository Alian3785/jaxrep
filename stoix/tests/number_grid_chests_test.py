"""Automatic, one-shot map loot and shared potion inventory on CUDA."""
from collections import OrderedDict
from stoix.tests.number_grid_fixtures import compiled_method
import copy
import threading

import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import MAP, REST, DEFEND, DIRECTIONS
from stoix.envs.number_grid_chests import ChestRules, CHEST_LOOT
from stoix.envs.number_grid_potions import POTION_START


def test_chest_configuration_observation_and_invalid_loot(current_game):
    env, state, _, _ = current_game
    assert env.observation_version == 28 and env.observation_size == 1031 and env.num_actions == 87
    assert state.chest_alive.tolist() == [True]*5 and state.potions.tolist() == [5,5,5,10]
    positions = [tuple(c['position']) for c in MAP['chests']]
    assert len(set(positions)) == 5
    assert not set(positions) & set(map(tuple, MAP['opponent_positions']))
    assert tuple(MAP['agent_position']) not in positions
    assert all(0 < x < 47 for p in positions for x in p)
    start = 220+5*env.num_opponents
    encoded = env.observation(state)[start:start+35].reshape(5,7)
    chex.assert_trees_all_close(encoded[:,:2], env.chest_rules.positions/47)
    chex.assert_trees_all_equal(encoded[:,2:6], env.chest_rules.loot.astype(jnp.float32))
    chex.assert_trees_all_equal(encoded[:,6], jnp.ones(5))
    collected = state.replace(chest_alive=jnp.zeros(5,bool),
        potions=state.potions+jnp.sum(env.chest_rules.loot,axis=0))
    assert collected.potions.tolist() == [6,6,6,12]
    chex.assert_trees_all_equal(env.observation(collected)[start:start+35].reshape(5,7)[:,6],jnp.zeros(5))
    columns = jnp.nonzero(env.observation(state.replace(potions=jnp.zeros(4,jnp.int32)))
                          != env.observation(state),size=4)[0]
    chex.assert_trees_all_close(env.observation(collected)[columns],jnp.full(4,1.2))
    invalid = [None, [None], [dict(position=[0,4],potions=[1,0,0,0])],
               [dict(position=MAP['agent_position'],potions=[1,0,0,0])],
               [dict(position=MAP['opponent_positions'][0],potions=[1,0,0,0])],
               [MAP['chests'][0],MAP['chests'][0]],
               [dict(position=[5,4],potions=[0,0,0,0])],
               [dict(position=[5,4],potions=[-1,0,0,0])],
               [dict(position=[5,4],potions=[True,0,0,0])],
               [dict(position=[5,4],potions=[2**31-1,0,0,0])]]
    for value in invalid:
        with pytest.raises(ValueError,match='[Cc]hest'):
            ChestRules({**MAP,'chests':value})


def test_all_eight_adjacent_tiles_once_and_every_loot_type_is_usable(current_game):
    env, initial, advance, _ = current_game
    chest = MAP['chests'][4]['position']  # all eight approach tiles are empty
    for direction, delta in enumerate(DIRECTIONS):
        state = initial.replace(position=jnp.array([chest[i]+2*delta[i] for i in range(2)]))
        action = jnp.int32((direction+4)%8)
        picked, ts = advance(state,action)
        assert picked.last_event == CHEST_LOOT and not picked.chest_alive[4]
        assert picked.potions.tolist() == [5,5,5,11]
        assert picked.movement_points == 18 and picked.map_steps == 1 and picked.day == 1
        assert float(ts.reward) == pytest.approx(env.exploration_bonus-env.step_cost)
        chex.assert_trees_all_equal(picked.hp,state.hp)
        chex.assert_trees_all_equal(picked.battle_key,state.battle_key)
        repeated, _ = advance(picked,action)  # step on the now-cleared chest tile
        chex.assert_trees_all_equal(repeated.potions,picked.potions)
        assert not repeated.chest_alive[4] and not jnp.any(repeated.last_loot)
    for index, chest in enumerate(MAP['chests']):
        row,col = chest['position']
        kind = chest['potions'].index(1)
        before = initial.replace(position=jnp.array([row,col-2]),potions=jnp.zeros(4,jnp.int32),
            hp=initial.hp.at[0].set(0 if kind == 3 else 1))
        picked, _ = advance(before,jnp.int32(2))
        assert picked.last_event == CHEST_LOOT and not picked.chest_alive[index]
        assert picked.potions[kind] == 1
        potion_action = POTION_START+kind*6
        assert env.action_mask(picked)[potion_action]
        used, _ = advance(picked,jnp.int32(potion_action))
        assert used.hp[0] == (1 if kind == 3 else min(120,1+(50,100,200)[kind]))
        assert not jnp.any(used.potions)
        assert used.movement_points == picked.movement_points and used.day == picked.day


def test_nonmovement_does_not_collect_and_multiple_chests_batch_safely(current_game):
    env,initial,advance,_ = current_game
    beside = initial.replace(position=jnp.array([25,11]),hp=initial.hp.at[0].set(1))
    for state,action in ((beside,REST),(beside,POTION_START),(beside,-1),
                         (beside.replace(movement_points=jnp.int32(0)),2),
                         (beside.replace(done=jnp.bool_(True)),2)):
        after,_=advance(state,jnp.int32(action))
        chex.assert_trees_all_equal(after.chest_alive,state.chest_alive)
        assert not jnp.any(after.last_loot)
    attack,_=advance(beside,jnp.int32(1))  # [24,12] is an Orc stack; attacker stays put
    assert attack.in_battle and attack.chest_alive[2]
    chex.assert_trees_all_equal(attack.potions,beside.potions)
    defended,_=advance(attack.replace(actor=jnp.int32(0)),jnp.int32(DEFEND))
    chex.assert_trees_all_equal(defended.chest_alive,beside.chest_alive)
    rules=ChestRules({**copy.deepcopy(MAP),'chests':[
        dict(position=[5,4],potions=[2,0,0,1]),dict(position=[5,5],potions=[0,1,1,0])]})
    one=initial.replace(position=jnp.array([4,4]),chest_alive=jnp.ones(2,bool))
    states=jax.tree.map(lambda x:jnp.stack((x,x)),one)
    picked=jax.jit(jax.vmap(rules.collect))(states,jnp.array([True,False]))
    chex.assert_trees_all_equal(picked.potions,jnp.array([[7,6,6,11],[5,5,5,10]]))
    chex.assert_trees_all_equal(picked.chest_alive,jnp.array([[False,False],[True,True]]))
    chex.assert_tree_all_finite(picked)


def test_human_service_collects_into_the_same_potion_inventory(current_game):
    from serve_number_grid import GameService
    env,initial,advance,_=current_game
    service=GameService.__new__(GameService)
    service.lock=threading.Lock()
    service.environments={env.construction.faction:(env,compiled_method(env,'reset'),advance)}
    state=initial.replace(position=jnp.array([4,2]),hp=initial.hp.at[0].set(20))
    service.sessions=OrderedDict({'loot':(env.construction.faction,state,0.)})
    picked=service.act('loot',2)['snapshot']
    assert picked['state']['last_event'] == CHEST_LOOT
    assert picked['state']['potions'] == [6,5,5,10]
    assert not picked['state']['chest_alive'][0]
    healed=service.act('loot',POTION_START)['snapshot']
    assert healed['state']['hp'][0] == 70 and healed['state']['potions'] == [5,5,5,10]
