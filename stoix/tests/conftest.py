"""Share the immutable current map and compiled GPU transitions between mechanics suites."""
from stoix.tests.number_grid_fixtures import compiled_method
import jax
import pytest

from stoix.tests.number_grid_fixtures import current_environment


@pytest.fixture(scope='session')
def current_game():
    env = current_environment()
    state, _ = env.reset(jax.random.PRNGKey(42))
    return env, state, compiled_method(env,'step'), compiled_method(env,'_battle_step')


@pytest.fixture(scope='session')
def human_service(current_game):
    from collections import OrderedDict
    import threading
    from serve_number_grid import GameService
    env, _, step, _ = current_game
    service = GameService.__new__(GameService)
    service.env = env
    service.environments = {env.construction.faction:(env,compiled_method(env,'reset'),step)}
    service.sessions = OrderedDict()
    service.lock = threading.Lock()
    return service


@pytest.fixture(scope='session')
def training_autoreset():
    from numbergrid_config import make_config
    from stoix.utils.make_env import make
    config = make_config()
    config.env.kwargs.max_steps = 1
    env, eval_env = make(config)
    return env, eval_env, compiled_method(env,'step')
