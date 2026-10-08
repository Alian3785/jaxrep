"""Transformation and state-contract checks for the current combat environment."""
from functools import lru_cache

import chex
import jax
import jax.numpy as jnp

from stoix.envs.number_grid import NumberGrid, SHOOT, WAIT, DEFEND, CONTINUE


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
    # Reuse the identical scalar oracle across eager/JIT variants; each still
    # executes its own batched transformation on the GPU.
    env = NumberGrid()
    states, actions = scenarios(env)
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    expected = jax.tree.map(lambda *xs: jnp.stack(xs),
                            *(env.step(state, action) for state, action in zip(states, actions)))
    return env, states, actions, batched, expected


class CombatTransformsTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_batched_step_matches_individual_steps(self):
        # Map movement, hero actions, enemy AI, escape, invalid and terminal steps.
        env, states, actions, batched, expected = reference_batch()
        actual = self.variant(jax.vmap(env.step))(batched, actions)
        assert_same_result(expected, actual)
        next_states, ts = actual
        chex.assert_shape(next_states.hp, (8, 12))
        chex.assert_shape(ts.observation, (8, env.observation_size))
        chex.assert_type([next_states.hp, next_states.defended, ts.observation],
                         [jnp.int32, jnp.bool_, jnp.float32])

    @chex.variants(with_jit=True, without_jit=True)
    def test_reset_and_mask_preserve_batch_contract(self):
        env = NumberGrid()
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


def test_step_does_not_retrace_when_only_state_values_change():
    env = NumberGrid()
    states, actions = scenarios(env)
    chex.clear_trace_counter()

    @jax.jit
    @chex.assert_max_traces(n=1)
    def advance(state, action):
        return env.step(state, action)

    for state, action in zip(states, actions):
        chex.assert_tree_all_finite(advance(state, action))
