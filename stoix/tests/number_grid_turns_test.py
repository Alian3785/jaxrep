"""Strategic turns, movement budget and rest are executed only on CUDA."""
import chex
import jax
import jax.numpy as jnp
import pytest

from numbergrid_config import make_config
from stoix.tests.number_grid_fixtures import MAP
from stoix.envs.number_grid import NumberGrid, SHOOT, DEFEND, CONTINUE, REST, RESTED
from stoix.utils.make_env import make


@pytest.fixture(scope='module')
def env():
    return NumberGrid(map_config=MAP)


def test_ten_moves_exhaust_budget_without_income_or_automatic_turn(env):
    state, initial = env.reset(jax.random.PRNGKey(42))
    assert state.gold == state.map_steps == 0 and state.day == 1
    chex.assert_trees_all_equal(initial.observation[-2:], jnp.array([0., 1.]))
    actions = jnp.tile(jnp.array([2, 6], jnp.int32), 6)
    @jax.jit
    def rollout(start):
        def step(carry, action):
            following, ts = env.step(carry, action)
            return following, (following.movement_points, following.gold, following.day,
                               ts.observation[-2:])
        return jax.lax.scan(step, start, actions)
    final, (points, gold, days, economy) = rollout(state)
    expected = jnp.maximum(20 - 2*jnp.arange(1, 13), 0)
    chex.assert_trees_all_equal(points, expected)
    chex.assert_trees_all_equal(gold, jnp.zeros(12, jnp.int32))
    chex.assert_trees_all_equal(days, jnp.ones(12, jnp.int32))
    chex.assert_trees_all_close(economy[:, 1], expected / 20.)
    assert final.map_steps == 10 and not final.in_battle
    chex.assert_trees_all_equal(final.position, state.position)
    mask = env.action_mask(final)
    assert not jnp.any(mask[:8]) and mask[REST]


def test_rest_penalty_is_linear_for_every_remaining_budget_and_free_at_zero(env):
    state, _ = env.reset(jax.random.PRNGKey(42))
    points = jnp.arange(0, 21, 2, dtype=jnp.int32)
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (11,)+x.shape), state)
    states = states.replace(movement_points=points, built_today=jnp.ones(11, jnp.bool_))
    assert jnp.all(jax.jit(jax.vmap(env.action_mask))(states)[:, REST])
    following, ts = jax.jit(jax.vmap(env.step))(states, jnp.full(11, REST, jnp.int32))
    chex.assert_trees_all_close(ts.reward, -.001*points)
    assert float(ts.reward[0]) == 0.
    chex.assert_trees_all_close(ts.extras['rest_penalty'], .001*points)
    assert jnp.all(ts.extras['turn_ended']) and not jnp.any(ts.extras['building_constructed'])
    chex.assert_trees_all_equal(following.movement_points, jnp.full(11, 20, jnp.int32))
    chex.assert_trees_all_equal(following.gold, jnp.full(11, 100, jnp.int32))
    chex.assert_trees_all_equal(following.day, jnp.full(11, 2, jnp.int32))
    chex.assert_trees_all_equal(following.last_event, jnp.full(11, RESTED, jnp.int32))
    assert not jnp.any(following.built_today)
    for field in ('position', 'hp', 'alive', 'map_steps', 'battle_key', 'buildings', 'blocked_buildings'):
        chex.assert_trees_all_equal(getattr(following, field), getattr(states, field))


def test_every_direction_costs_two_and_insufficient_points_rejects_the_move(env):
    state, _ = env.reset(jax.random.PRNGKey(42))
    states = jax.tree.map(lambda x: jnp.broadcast_to(x, (24,)+x.shape), state)
    # All eight neighboring cells of the initial position are free.
    points = jnp.repeat(jnp.array([2, 1, 0], jnp.int32), 8)
    states = states.replace(movement_points=points)
    actions = jnp.tile(jnp.arange(8, dtype=jnp.int32), 3)
    masks = jax.jit(jax.vmap(env.action_mask))(states)
    chex.assert_trees_all_equal(masks[jnp.arange(24), actions], points >= 2)
    following, _ = jax.jit(jax.vmap(env.step))(states, actions)
    chex.assert_trees_all_equal(following.position[:8], state.position + env.directions)
    chex.assert_trees_all_equal(following.position[8:], states.position[8:])
    chex.assert_trees_all_equal(following.movement_points, jnp.where(points >= 2, points-2, points))


def test_invalid_actions_battle_and_terminal_states_cannot_advance_the_day(env):
    world, _ = env.reset(jax.random.PRNGKey(42))
    battle = env._begin_battle(world.replace(enemy=jnp.int32(11))).replace(actor=jnp.int32(0))
    states = (
        world.replace(position=jnp.array([1, 1], jnp.int32)),
        world.replace(position=jnp.array([3, 7], jnp.int32), movement_points=jnp.int32(0)),
        world, world, world.replace(done=jnp.bool_(True)),
        battle, battle, battle.replace(actor=jnp.int32(6)), battle,
    )
    actions = jnp.array([0, 2, -1, 44, REST, SHOOT, DEFEND, CONTINUE, REST], jnp.int32)
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    following, ts = jax.jit(jax.vmap(env.step))(batched, actions)
    for field in ('gold', 'map_steps', 'day', 'movement_points', 'built_today'):
        chex.assert_trees_all_equal(getattr(following, field), getattr(batched, field))
    assert not jnp.any(ts.extras['turn_ended'])
    assert not env.action_mask(battle)[REST] and not env.action_mask(states[4])[REST]
    chex.assert_trees_all_equal(following.hp[-1], battle.hp)
    chex.assert_trees_all_equal(following.battle_key[-1], battle.battle_key)


def test_last_points_can_attack_and_recovery_does_not_restore_movement(env):
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = state.replace(position=jnp.array([3, 7], jnp.int32), movement_points=jnp.int32(2))
    state, _ = jax.jit(env.step)(state, jnp.int32(2))
    assert state.in_battle and state.gold == 0 and state.movement_points == 0
    state = state.replace(actor=jnp.int32(0))
    won, _ = jax.jit(env._battle_step)(state, jnp.int32(SHOOT), state.battle_key, jnp.zeros(36))
    escaping = state.replace(hp=state.hp.at[1:6].set(0), retreating=state.retreating.at[0].set(True))
    escaped, _ = jax.jit(env._battle_step)(escaping, jnp.int32(CONTINUE), escaping.battle_key, jnp.zeros(36))
    for recovered in (won, escaped):
        assert not recovered.in_battle and recovered.day == 1 and recovered.map_steps == 0
        assert recovered.movement_points == recovered.gold == 0
        assert not jnp.any(env.action_mask(recovered)[:8]) and env.action_mask(recovered)[REST]


def test_repeated_rest_and_exhaustion_cycles_pay_once_per_rest(env):
    state, _ = env.reset(jax.random.PRNGKey(42))
    actions = jnp.concatenate((jnp.array([REST, REST], jnp.int32),
                               jnp.tile(jnp.concatenate((jnp.tile(jnp.array([2, 6], jnp.int32), 5),
                                                         jnp.array([REST], jnp.int32))), 2)))
    def step(carry, action):
        following, ts = env.step(carry, action)
        return following, ts.reward
    final, rewards = jax.jit(lambda start: jax.lax.scan(step, start, actions))(state)
    assert final.day == 5 and final.gold == 400 and final.map_steps == 20
    assert final.movement_points == 20
    chex.assert_trees_all_close(rewards[:2], jnp.full(2, -.02))
    assert rewards[12] == rewards[23] == 0


def test_training_autoreset_rest_keeps_terminal_income_and_new_turn_observation():
    config = make_config()
    config.env.kwargs.max_steps = 11
    training, _ = make(config)
    state, _ = training.reset(jax.random.split(jax.random.PRNGKey(7), 2))
    advance = jax.jit(training.step)
    for index in range(10):
        state, _ = advance(state, jnp.full(2, 2 if index % 2 == 0 else 6, jnp.int32))
    state, ts = advance(state, jnp.full(2, REST, jnp.int32))
    assert jnp.all(ts.truncated())
    chex.assert_trees_all_equal(state.gold, jnp.zeros(2, jnp.int32))
    chex.assert_trees_all_equal(state.day, jnp.ones(2, jnp.int32))
    chex.assert_trees_all_equal(state.movement_points, jnp.full(2, 20, jnp.int32))
    chex.assert_trees_all_equal(ts.reward, jnp.zeros(2))
    chex.assert_trees_all_close(ts.extras['next_obs']['observation'][:, -14:-12], jnp.array([[.1, 1.], [.1, 1.]]))
    chex.assert_trees_all_equal(ts.observation['observation'][:, -14:-12], jnp.array([[0., 1.], [0., 1.]]))
    assert jnp.all(ts.observation['action_mask'][:, REST])


def test_bad_rest_penalty_settings_are_rejected():
    for penalty in (-.1, float('nan'), float('inf'), True, '0.001'):
        with pytest.raises(ValueError, match='rest_penalty_per_point'):
            NumberGrid(map_config={**MAP, 'rest_penalty_per_point': penalty})
