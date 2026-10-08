"""Original recovery potions, finite inventory and map-only actions on CUDA."""
from collections import OrderedDict
import threading

import chex
import jax
import jax.numpy as jnp
import pytest

from numbergrid_config import make_config
from stoix.envs.number_grid import NumberGrid, BattleState, MAP, REST, SHOOT
from stoix.envs.number_grid_potions import PotionRules, POTION_START, POTION_HEAL, POTION_REVIVE
from stoix.utils.make_env import make


@pytest.fixture(scope='module')
def game(current_game):
    return current_game[:3]


def test_starting_stock_observation_and_input_validation(game):
    env, state, _ = game
    assert state.potions.tolist() == [5,5,5,10]
    assert env.num_actions == 80 and env.observation_size == 532
    assert env.observation(state).shape == (532,)
    chex.assert_trees_all_equal(env.observation(state)[-55:-51], jnp.ones(4))
    used = state.replace(potions=jnp.array([4,3,0,9]))
    chex.assert_trees_all_close(env.observation(used)[-55:-51], jnp.array([.8,.6,0.,.9]))
    assert [p['amount'] for p in env.potion_rules.metadata()] == [50,100,200,1]
    for invalid in ([], [5,5,10], [5,5,5,-1], [5,True,5,10], [5,5,5,2**31]):
        with pytest.raises(ValueError, match='initial_potions'):
            PotionRules(invalid)


def test_healing_sizes_overheal_and_shared_inventory_without_turn_cost(game):
    env, state, advance = game
    wounded = state.replace(position=jnp.array([2,7]), movement_points=jnp.int32(0),
        built_today=jnp.bool_(True), unit_levels=state.unit_levels.at[0].set(50),
        hp=state.hp.at[0].set(10), recovery_balance=jnp.array([70,2]))
    for kind, amount in enumerate((50,100,200)):
        action = POTION_START+kind*6
        assert env.action_mask(wounded)[action]  # no temple, no gold, zero movement, near an enemy
        healed, ts = advance(wounded, jnp.int32(action))
        assert healed.hp[0] == 10+amount and healed.potions[kind] == 4
        assert healed.last_event == POTION_HEAL and healed.last_damage == amount
        assert healed.last_potion == kind and healed.last_target == 0
        assert float(ts.reward) == pytest.approx(-env.step_cost)
        for field in ('gold','movement_points','day','built_today','buildings','position','map_steps',
                      'recovery_balance','battle_key','unit_ids','unit_xp','unit_levels'):
            chex.assert_trees_all_equal(getattr(healed,field),getattr(wounded,field))
        assert not healed.in_battle
    nearly_full = state.replace(hp=state.hp.at[0].set(119))
    capped, _ = advance(nearly_full, jnp.int32(POTION_START+12))
    assert capped.hp[0] == 120 and capped.last_damage == 1 and capped.potions[2] == 4
    # A single shared stock, not five bottles per target; rest does not refill it.
    target_two = capped.replace(hp=capped.hp.at[1].set(10))
    used, _ = advance(target_two, jnp.int32(POTION_START+13))
    assert used.hp[1] == 150 and used.potions[2] == 3
    rested, _ = advance(used, jnp.int32(REST))
    chex.assert_trees_all_equal(rested.potions, used.potions)


def test_revive_then_heal_and_invalid_uses_never_consume(game):
    env, state, advance = game
    dead = state.replace(hp=state.hp.at[1].set(0), position=jnp.array([1,1]),
                         recovery_balance=jnp.array([-7,-1]))
    revived, ts = advance(dead, jnp.int32(POTION_START+19))
    assert revived.hp[1] == 1 and revived.potions.tolist() == [5,5,5,9]
    assert revived.last_event == POTION_REVIVE and revived.last_damage == 1
    assert float(ts.reward) == pytest.approx(-env.step_cost)
    chex.assert_trees_all_equal(revived.recovery_balance, dead.recovery_balance)
    healed, _ = advance(revived, jnp.int32(POTION_START+1))
    assert healed.hp[1] == 51 and healed.potions.tolist() == [4,5,5,9]
    invalid = [(state,56), (state,74), (dead,57), (state,61), (state,79),
               (dead.replace(potions=dead.potions.at[3].set(0)),75),
               (dead.replace(position=state.position, buildings=env.capital.temple_bit, gold=jnp.int32(1000),
                             potions=dead.potions.at[3].set(0)),75),
               (healed.replace(potions=healed.potions.at[0].set(0)),57),
               (healed.replace(position=state.position, buildings=env.capital.temple_bit, gold=jnp.int32(1000),
                               potions=healed.potions.at[0].set(0)),57),
               (dead.replace(done=jnp.bool_(True)),75),
               (env._begin_battle(dead.replace(enemy=jnp.int32(0))),75), (dead,80), (dead,-1)]
    for before, action in invalid:
        if 0 <= action < env.num_actions:
            assert not env.action_mask(before)[action]
        after, _ = advance(before,jnp.int32(action))
        for field in ('hp','potions','gold','movement_points','day','buildings','battle_key','recovery_balance'):
            chex.assert_trees_all_equal(getattr(after,field),getattr(before,field))
    reset, _ = env.reset(jax.random.PRNGKey(43))
    assert reset.potions.tolist() == [5,5,5,10]


def test_every_type_can_target_all_six_slots_under_jit_vmap():
    env = NumberGrid(map_config={**MAP,'hero_units':6,
        'hero_roster':['possessed','duke','possessed','cultist','cultist','possessed']})
    state, _ = env.reset(jax.random.PRNGKey(42))
    actions = jnp.arange(56,80,dtype=jnp.int32)
    states = jax.tree.map(lambda x:jnp.broadcast_to(x,(24,)+x.shape),state)
    hp = jnp.where(jnp.arange(24)[:,None] < 18, 1, 0)
    states = states.replace(hp=states.hp.at[:,:6].set(jnp.broadcast_to(hp,(24,6))))
    masks = jax.jit(jax.vmap(env.action_mask))(states)
    assert jnp.all(masks[jnp.arange(24),actions])
    following, _ = jax.jit(jax.vmap(env.step))(states,actions)
    slots, kinds = jnp.arange(24)%6, jnp.arange(24)//6
    maximum = env.max_hp(state)[:6]
    expected = jnp.where(kinds==3,1,jnp.minimum(maximum[slots],1+jnp.array([50,100,200,1])[kinds]))
    chex.assert_trees_all_equal(following.hp[jnp.arange(24),slots],expected)
    assert jnp.all(jnp.sum(states.potions-following.potions,axis=1)==1)
    chex.assert_tree_all_finite(following)


def test_large_unit_uses_only_its_anchor_and_battle_keeps_stock(game):
    env = NumberGrid(map_config={**MAP,
        'hero_roster':['titan','duke','possessed',None,'cultist','cultist']})
    state, _ = env.reset(jax.random.PRNGKey(42))
    dead = state.replace(hp=state.hp.at[0].set(0))
    assert env.action_mask(dead)[74] and not env.action_mask(dead)[77]
    revived, _ = jax.jit(env.step)(dead,jnp.int32(74))
    assert revived.hp[0] == 1 and revived.hp[3] == 0 and revived.potions[3] == 9
    battle = env._begin_battle(revived.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(0))
    battle = battle.replace(hp=battle.hp.at[6].set(1))
    won, _ = jax.jit(env._battle_step)(battle,jnp.int32(SHOOT),battle.battle_key,jnp.zeros(36))
    assert not won.in_battle
    chex.assert_trees_all_equal(won.potions,revived.potions)


def test_ppo_autoreset_restores_inventory_and_keeps_terminal_stock():
    config = make_config()
    config.env.kwargs.max_steps = 1
    training, _ = make(config)
    state, _ = training.reset(jax.random.split(jax.random.PRNGKey(42),2))
    def injure(live):
        if isinstance(live,BattleState):
            return live.replace(hp=live.hp.at[:,0].set(10),potions=live.potions.at[:,0].set(2),
                                position=live.position.at[1].set(jnp.array([4,2])))
        return live.replace(base_env_state=injure(live.base_env_state))
    state = injure(state)
    following, ts = jax.jit(training.step)(state,jnp.array([POTION_START,2],jnp.int32))
    assert jnp.all(ts.truncated())
    chex.assert_trees_all_equal(following.potions,jnp.array([[5,5,5,10]]*2))
    chex.assert_trees_all_close(ts.extras['next_obs']['observation'][:,-55:-51],
                              jnp.array([[.2,1.,1.,1.],[.6,1.,1.,1.]]))
    chex.assert_trees_all_equal(ts.observation['observation'][:,-55:-51],jnp.ones((2,4)))
    chex.assert_trees_all_equal(following.chest_alive,jnp.ones((2,5),bool))
    chest_flag=220+5*len(MAP['opponent_positions'])+6
    chex.assert_trees_all_equal(ts.extras['next_obs']['observation'][:,chest_flag],jnp.array([1.,0.]))
    chex.assert_trees_all_equal(ts.observation['observation'][:,chest_flag],jnp.ones(2))


def test_human_service_uses_same_inventory_and_quotes(game):
    from serve_number_grid import GameService
    env, state, advance = game
    service = GameService.__new__(GameService)
    service.lock = threading.Lock()
    service.environments = {env.construction.faction:(env,jax.jit(env.reset),advance)}
    state = state.replace(hp=state.hp.at[0].set(20).at[1].set(0))
    service.sessions = OrderedDict({'potions':(env.construction.faction,state,0.)})
    snap = service.snapshot(env,state,0.)
    assert snap['potion_quotes'][0][0] == 50 and snap['potion_quotes'][2][0] == 100
    result = service.act('potions',56)['snapshot']
    assert result['state']['hp'][0] == 70 and result['state']['potions'] == [4,5,5,10]
    result = service.act('potions',75)['snapshot']
    assert result['state']['hp'][1] == 1 and result['state']['potions'] == [4,5,5,9]
