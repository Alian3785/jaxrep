"""Transformation and state-contract checks for the current combat environment."""
from functools import lru_cache

from stoix.tests.number_grid_fixtures import compiled_method
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from stoa import AddActionMaskWrapper
from stoix.networks.base import FeedForwardActor
from stoix.networks.heads import CategoricalHead
from stoix.networks.torso import MLPTorso

from stoix.tests.number_grid_fixtures import current_environment
from stoix.envs.number_grid import SHOOT, WAIT, DEFEND, CONTINUE, REST


def scenarios(env):
    world, _ = env.reset(jax.random.PRNGKey(42))
    battle = env._begin_battle(world.replace(enemy=jnp.int32(11)))
    hero = battle.replace(actor=jnp.int32(0))
    states = (
        world, hero, hero, hero,
        battle.replace(actor=jnp.int32(6)),
        hero.replace(retreating=hero.retreating.at[0].set(True)),
        hero.replace(turn_phase=hero.turn_phase.at[0].set(1)),
        hero.replace(done=jnp.bool_(True)),
    )
    actions = jnp.asarray([2, SHOOT, WAIT, DEFEND, CONTINUE, CONTINUE, WAIT, SHOOT], jnp.int32)
    return states, actions


def assert_same_result(expected, actual):
    chex.assert_trees_all_equal_shapes_and_dtypes(expected, actual)
    # XLA can fuse float32 normalization; discrete state must still be exact.
    chex.assert_trees_all_close(expected, actual, atol=1e-6, rtol=1e-6)
    chex.assert_tree_all_finite(actual)
    for wanted, got in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        if not jnp.issubdtype(wanted.dtype, jnp.inexact):
            chex.assert_trees_all_equal(wanted, got)


@lru_cache(maxsize=1)
def reference_batch():
    # Compare compiled scalar steps with the compiled vmapped production path.
    # Reuse the scalar oracle rather than compiling each individual action.
    env = current_environment()
    states, actions = scenarios(env)
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    scalar_step = compiled_method(env,'step')
    expected = jax.tree.map(lambda *xs: jnp.stack(xs),
                            *(scalar_step(state, action) for state, action in zip(states, actions)))
    return env, states, actions, batched, expected


class CombatTransformsTest(chex.TestCase):
    def test_batched_step_matches_individual_steps(self):
        # Map movement, hero actions, enemy AI, escape, invalid and terminal steps.
        env, states, actions, batched, expected = reference_batch()
        actual = compiled_method(env,'step',batched=True)(batched,actions)
        assert_same_result(expected, actual)
        next_states, ts = actual
        chex.assert_shape(next_states.hp, (8, 12))
        chex.assert_shape(ts.observation, (8, env.observation_size))
        chex.assert_type([next_states.hp, next_states.defended, ts.observation],
                         [jnp.int32, jnp.bool_, jnp.float32])

    @chex.variants(with_jit=True, without_jit=False)
    def test_reset_and_mask_preserve_batch_contract(self):
        env = current_environment()
        keys = jax.random.split(jax.random.PRNGKey(7), 4)
        states, ts = self.variant(jax.vmap(env.reset))(keys)
        masks = self.variant(jax.vmap(env.action_mask))(states)
        chex.assert_shape(masks, (4, env.num_actions))
        chex.assert_type(masks, jnp.bool_)
        chex.assert_tree_all_finite((states, ts))
        for index, key in enumerate(keys):
            expected = env.reset(key)
            actual = jax.tree.map(lambda x, index=index: x[index], (states, ts))
            assert_same_result(expected, actual)


def test_cached_autoreset_changes_rng_and_preserves_final_observation(training_autoreset):
    env, eval_env, advance = training_autoreset
    assert isinstance(eval_env, AddActionMaskWrapper)
    state, ts = env.reset(jax.random.split(jax.random.PRNGKey(2),2))
    original = np.asarray(state.battle_key)
    state, ts = advance(state,jnp.array([7,3]))
    assert np.all(ts.truncated()) and np.all(state.step_count == 0)
    assert not np.array_equal(original,state.battle_key)
    assert np.all(ts.extras['next_obs']['observation'][:,3] == 1)
    assert np.all(ts.observation['observation'][:,3] == 0)
    np.testing.assert_array_equal(ts.observation['action_mask'][:,8:REST],False)
    assert np.all(ts.observation['action_mask'][:, REST])
    again, _ = advance(state,jnp.array([7,3]))
    assert not np.array_equal(state.battle_key,again.battle_key)


def test_masked_warrior_policy_sample_argmax_and_ppo_gradient(current_game):
    env, initial, _, _ = current_game
    state = env._begin_battle(initial.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(0))
    # Wrapper/schema integration is covered by batched reset/step above.
    # The gradient check only needs the actual combat observation and mask.
    observation = {'observation':env.observation(state),'action_mask':env.action_mask(state)}
    obs = jax.tree.map(lambda x:jnp.repeat(x[None],64,axis=0),observation)
    policy = FeedForwardActor(torso=MLPTorso(layer_sizes=[32]), action_head=CategoricalHead(env.num_actions))
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


def test_human_service_uses_identical_mask_and_rejects_unreachable_target(human_service):
    service = human_service
    session = service.create(42)['session']
    env = service.env
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = env._begin_battle(state.replace(enemy=jnp.int32(11)))
    state = state.replace(actor=jnp.int32(0))
    service.sessions[session] = (env.construction.faction, state, 0.)
    snapshot = service.snapshot(env, state, 0.)
    np.testing.assert_array_equal(snapshot['action_mask'], env.action_mask(state))
    with pytest.raises(ValueError, match='недоступно'):
        service.act(session, SHOOT+3)
    assert service.sessions[session][1] is state
    direct, ts = compiled_method(env,'step')(state,jnp.int32(SHOOT))
    first = service.act(session, SHOOT)['events'][0]
    assert first == {**service.snapshot(env, direct, float(ts.reward)), 'reward': float(ts.reward)}


def test_batched_random_battle_rollouts_have_valid_actions_and_finite_state(current_game):
    env, initial, _, _ = current_game
    start = jax.vmap(lambda k,e: env._begin_battle(initial.replace(battle_key=k,enemy=e)))(
        jax.random.split(jax.random.PRNGKey(1),64),jnp.arange(64,dtype=jnp.int32)%env.num_opponents)
    def run(states):
        def step(carry, _):
            states, key = carry
            key, sub = jax.random.split(key)
            masks = jax.vmap(env.action_mask)(states)
            action = jnp.argmax(jnp.where(masks,jax.random.uniform(sub,masks.shape),-1),axis=-1)
            states, ts = jax.vmap(env.step)(states,action)
            good = jnp.all(states.done | jnp.any(masks,axis=-1)) & jnp.all(states.hp>=0) & jnp.all(jnp.isfinite(ts.observation))
            return (states,key),good
        return jax.lax.scan(step,(states,jax.random.PRNGKey(5)),None,length=150)
    (states,_),good = jax.jit(run)(start)
    assert np.all(good) and int(jnp.sum(states.battle_steps)) > 100
