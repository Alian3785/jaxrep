"""Gold accrues on successful map movement, independently of combat turns."""
import chex
import jax
import jax.numpy as jnp
import pytest

from numbergrid_config import make_config
from stoix.envs.number_grid import NumberGrid, SHOOT, DEFEND, CONTINUE
from stoix.utils.make_env import make


@pytest.fixture(scope='module')
def env():
    return NumberGrid()


def test_income_at_20_40_and_60_moves_and_reset(env):
    state, initial = env.reset(jax.random.PRNGKey(42))
    assert state.gold == state.map_steps == 0
    chex.assert_trees_all_equal(initial.observation[-2:], jnp.zeros(2))
    actions = jnp.tile(jnp.array([2, 6], jnp.int32), 30)

    @jax.jit
    def rollout(start):
        def step(carry, action):
            following, ts = env.step(carry, action)
            return following, (following.gold, following.map_steps, ts.observation[-2:])
        return jax.lax.scan(step, start, actions)

    final, (gold, moves, economy) = rollout(state)
    expected_moves = jnp.arange(1, 61)
    chex.assert_trees_all_equal(moves, expected_moves)
    chex.assert_trees_all_equal(gold, expected_moves // 20 * 100)
    chex.assert_trees_all_close(economy[:, 0], gold / 1000.)
    chex.assert_trees_all_close(economy[:, 1], expected_moves % 20 / 20.)
    assert final.gold == 300 and final.map_steps == 60 and not final.in_battle
    fresh, _ = env.reset(jax.random.PRNGKey(43))
    assert fresh.gold == fresh.map_steps == 0


def test_invalid_moves_terminal_state_and_battle_actions_do_not_pay(env):
    world, _ = env.reset(jax.random.PRNGKey(42))
    world = world.replace(map_steps=jnp.int32(19))
    battle = env._begin_battle(world.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(0))
    states = (
        world.replace(position=jnp.array([1, 1], jnp.int32)),  # wall
        world.replace(position=jnp.array([3, 7], jnp.int32)),  # occupied cell
        world, world, world.replace(done=jnp.bool_(True)),
        battle, battle, battle.replace(actor=jnp.int32(6)),
    )
    actions = jnp.array([0, 2, -1, 18, 2, SHOOT, DEFEND, CONTINUE], jnp.int32)
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    following, _ = jax.jit(jax.vmap(env.step))(batched, actions)
    chex.assert_trees_all_equal(following.gold, jnp.zeros(8, jnp.int32))
    chex.assert_trees_all_equal(following.map_steps, jnp.full(8, 19, jnp.int32))


def test_twentieth_move_pays_before_battle_and_recovery_preserves_gold(env):
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = state.replace(position=jnp.array([2, 6], jnp.int32), map_steps=jnp.int32(19))
    state, _ = jax.jit(env.step)(state, jnp.int32(2))
    assert state.in_battle and state.gold == 100 and state.map_steps == 20
    state = state.replace(actor=jnp.int32(0))
    won, _ = jax.jit(env._battle_step)(state, jnp.int32(SHOOT), state.battle_key, jnp.zeros(36))
    assert not won.in_battle and won.gold == 100 and won.map_steps == 20
    escaping = state.replace(
        hp=state.hp.at[1:6].set(0), retreating=state.retreating.at[0].set(True))
    escaped, _ = jax.jit(env._battle_step)(
        escaping, jnp.int32(CONTINUE), escaping.battle_key, jnp.zeros(36))
    assert not escaped.in_battle and escaped.gold == 100 and escaped.map_steps == 20


def test_training_autoreset_resets_gold_but_keeps_terminal_observation():
    config = make_config()
    config.env.kwargs.max_steps = 20
    training, _ = make(config)
    state, _ = training.reset(jax.random.split(jax.random.PRNGKey(7), 2))
    advance = jax.jit(training.step)
    for index in range(20):
        state, ts = advance(state, jnp.full(2, 2 if index % 2 == 0 else 6, jnp.int32))
    assert jnp.all(ts.truncated())
    chex.assert_trees_all_equal(state.gold, jnp.zeros(2, jnp.int32))
    chex.assert_trees_all_equal(state.map_steps, jnp.zeros(2, jnp.int32))
    chex.assert_trees_all_close(ts.extras['next_obs']['observation'][:, -2], jnp.full(2, .1))
    chex.assert_trees_all_equal(ts.observation['observation'][:, -2:], jnp.zeros((2, 2)))
