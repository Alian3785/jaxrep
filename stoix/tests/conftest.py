"""Share the immutable current map and compiled GPU transitions between mechanics suites."""
import jax
import pytest

from stoix.envs.number_grid import NumberGrid


@pytest.fixture(scope='session')
def current_game():
    env = NumberGrid()
    state, _ = env.reset(jax.random.PRNGKey(42))
    return env, state, jax.jit(env.step), jax.jit(env._battle_step)
