"""Verify game rules and the GPU-compatible reset/step contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf
from stoa import StepType

from stoix.environments.number_grid import LOST, RUNNING, TIMED_OUT, WON, NumberGrid
from stoix.utils.make_env import make_number_grid_env


@pytest.fixture
def env():
    return NumberGrid()


def initial(env):
    return env.reset(jax.random.PRNGKey(42))[0]


def test_fixed_layout_and_observation(env):
    a, ta = env.reset(jax.random.PRNGKey(0))
    b, tb = env.reset(jax.random.PRNGKey(987))
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        np.testing.assert_array_equal(x, y)
    assert a.walls.shape == (16, 16)
    assert int(a.walls.sum()) == 60
    assert int(a.agent_strength) == 1
    np.testing.assert_array_equal(a.opponent_strengths, [0, 1, 2])
    assert ta.observation.shape == (289,)
    np.testing.assert_array_equal(ta.observation, tb.observation)
    assert bool(env.observation_space().contains(ta.observation))
    assert int(ta.step_type) == int(StepType.FIRST)


@pytest.mark.parametrize("action, expected", [(0, [7, 7]), (1, [7, 8]), (2, [8, 8]),
                         (3, [9, 8]), (4, [9, 7]), (5, [9, 6]), (6, [8, 6]), (7, [7, 6])])
def test_eight_directions(env, action, expected):
    state = initial(env).replace(agent_position=jnp.array([8, 7], jnp.int32))
    next_state, timestep = jax.jit(env.step)(state, jnp.int32(action))
    np.testing.assert_array_equal(next_state.agent_position, expected)
    assert float(timestep.reward) == 0
    assert int(next_state.step_count) == 1


def test_walls_and_occupied_cells_block_motion(env):
    state = initial(env)
    moved, timestep = env.step(state, jnp.int32(5))  # southwest into perimeter
    np.testing.assert_array_equal(moved.agent_position, state.agent_position)
    assert float(timestep.reward) == 0
    state = state.replace(agent_position=jnp.array([8, 7]),
                          opponent_positions=jnp.array([[8, 8], [3, 12], [2, 3]]))
    assert not bool(env.legal_actions(state)[2])
    moved, _ = env.step(state, jnp.int32(2))
    np.testing.assert_array_equal(moved.agent_position, [8, 7])


def test_growth_sequence_and_win_rewards(env):
    state = initial(env).replace(agent_position=jnp.array([8, 7]),
                          opponent_positions=jnp.array([[8, 9], [3, 12], [2, 3]]))
    state, first = env.step(state, jnp.int32(2))
    assert float(first.reward) == 1 and int(state.agent_strength) == 2
    np.testing.assert_array_equal(state.opponent_alive, [False, True, True])
    state = state.replace(agent_position=jnp.array([5, 11]))
    state, second = env.step(state, jnp.int32(0))  # (4,11) diagonally adjacent to 1
    assert float(second.reward) == 1 and int(state.agent_strength) == 3
    state = state.replace(agent_position=jnp.array([4, 5]))
    state, final = env.step(state, jnp.int32(7))  # (3,4) diagonally adjacent to 2
    assert float(final.reward) == 4 and int(state.agent_strength) == 4
    assert int(state.outcome) == WON and not bool(state.opponent_alive.any())
    assert float(first.reward + second.reward + final.reward) == 6
    assert bool(final.terminated()) and float(final.discount) == 0
    again, repeated = env.step(state, jnp.int32(2))
    assert float(repeated.reward) == 0 and int(again.step_count) == int(state.step_count)


@pytest.mark.parametrize("opponent_strength", [1, 2])
def test_equal_or_stronger_opponent_causes_loss(env, opponent_strength):
    state = initial(env).replace(agent_position=jnp.array([8, 7]),
        opponent_positions=jnp.array([[8, 9], [3, 12], [2, 3]]),
        opponent_strengths=jnp.array([opponent_strength, 1, 2]))
    state, timestep = env.step(state, jnp.int32(2))
    assert int(state.outcome) == LOST and float(timestep.reward) == -1
    assert int(state.agent_strength) == 1 and bool(state.opponent_alive.all())


def test_loss_has_priority_over_weaker_neighbor(env):
    state = initial(env).replace(agent_position=jnp.array([8, 7]),
        opponent_positions=jnp.array([[7, 8], [9, 8], [2, 3]]))
    state, timestep = env.step(state, jnp.int32(2))
    assert int(state.outcome) == LOST and float(timestep.reward) == -1
    assert int(state.agent_strength) == 1 and bool(state.opponent_alive.all())
    assert int(timestep.extras["opponents_defeated"]) == 0


def test_all_weaker_adjacent_opponents_removed(env):
    state = initial(env).replace(agent_position=jnp.array([8, 7]),
        agent_strength=jnp.int32(3), opponent_positions=jnp.array([[7, 8], [9, 8], [2, 3]]))
    state, timestep = env.step(state, jnp.int32(2))
    assert float(timestep.reward) == 2 and int(state.agent_strength) == 5
    np.testing.assert_array_equal(state.opponent_alive, [False, False, True])


def test_diagonal_contact_is_adjacent(env):
    state = initial(env).replace(agent_position=jnp.array([8, 7]),
        opponent_positions=jnp.array([[7, 9], [3, 12], [2, 3]]))
    state, timestep = env.step(state, jnp.int32(2))
    assert float(timestep.reward) == 1 and int(state.agent_strength) == 2


def test_timeout_at_exactly_2000_is_truncation(env):
    state = initial(env).replace(step_count=jnp.int32(1998))
    state, first = env.step(state, jnp.int32(5))
    assert int(state.outcome) == RUNNING and not bool(first.last())
    state, final = env.step(state, jnp.int32(5))
    assert int(state.outcome) == TIMED_OUT and int(state.step_count) == 2000
    assert bool(final.truncated()) and float(final.discount) == 1 and float(final.reward) == 0


def test_loss_on_step_2000_has_priority_over_timeout(env):
    state = initial(env).replace(step_count=jnp.int32(1999), agent_position=jnp.array([8, 7]),
        opponent_positions=jnp.array([[3, 3], [8, 9], [2, 3]]))
    state, final = env.step(state, jnp.int32(2))
    assert int(state.outcome) == LOST and bool(final.terminated()) and float(final.reward) == -1


def test_jit_vmap_scan_run_on_gpu_and_keep_fixed_shapes(env):
    assert all(device.platform == "gpu" for device in jax.devices())
    keys = jax.random.split(jax.random.PRNGKey(3), 32)
    states, _ = jax.jit(jax.vmap(env.reset))(keys)

    @jax.jit
    def rollout(states):
        def one(current, _):
            return jax.vmap(env.step)(current, jnp.full(32, 5, dtype=jnp.int32))
        return jax.lax.scan(one, states, None, length=25)

    states, timesteps = rollout(states)
    jax.block_until_ready((states, timesteps))
    assert timesteps.observation.shape == (25, 32, 289)
    assert all(array.device.platform == "gpu" for array in jax.tree.leaves((states, timesteps)))


def test_stoix_autoreset_retains_terminal_observation_and_metrics(env):
    config = OmegaConf.create({"arch": {"num_envs": 2}, "env": {"kwargs": {"max_steps": 1}}})
    wrapped, _ = make_number_grid_env("NumberGrid-16x16-v0", config)
    states, _ = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
    states, timesteps = jax.jit(wrapped.step)(states, jnp.array([0, 0], jnp.int32))
    assert bool(jnp.all(timesteps.truncated()))
    assert bool(jnp.all(timesteps.extras["episode_metrics"]["timed_out_episode"]))
    np.testing.assert_array_equal(timesteps.extras["episode_metrics"]["episode_length"], [1, 1])
    # After auto-reset the clock observation is 0; actual final observation has clock 1.
    np.testing.assert_array_equal(timesteps.observation[:, -9], [0, 0])
    np.testing.assert_array_equal(timesteps.extras["next_obs"][:, -9], [1, 1])
