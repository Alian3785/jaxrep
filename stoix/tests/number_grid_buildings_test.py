"""Capital rules checked on CUDA, including every building of every faction."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from numbergrid_config import make_config
from stoix.tests.number_grid_fixtures import MAP
from stoix.envs.number_grid import NumberGrid, BUILD, ACTIONS, BUILD_START, REST
from stoix.envs.number_grid_buildings import FACTIONS, BuildingRules
from stoix.utils.make_env import make


def ancestors(rows, index):
    by_name = {row['name']: i for i, row in enumerate(rows)}
    result = set()
    pending = list(rows[index]['requires'])
    while pending:
        current = by_name[pending.pop()]
        if current not in result:
            result.add(current)
            pending.extend(rows[current]['requires'])
    return result


@pytest.fixture(scope='module', params=FACTIONS)
def env(request):
    return NumberGrid(map_config={**MAP, 'faction': request.param})


def test_every_faction_building_cost_requirements_and_exclusions(env):
    rows = env.construction.rows
    assert len(rows) == {'empire': 25, 'mountain_clans': 24, 'undead_hordes': 25,
                         'legions': 25, 'elves': 23}[env.construction.faction]
    initial, ts = env.reset(jax.random.PRNGKey(42))
    assert initial.gold == initial.buildings == 0 and not initial.built_today
    assert ts.observation.shape == (271,) and env.action_space().num_values == ACTIONS == 44
    # Supply precisely the prerequisites and exact gold for each of all 122
    # buildings. This covers each shared action slot, including service buildings.
    states = [initial.replace(buildings=jnp.uint32(sum(1 << j for j in ancestors(rows, i))),
                              gold=jnp.int32(row['gold']), movement_points=jnp.int32(0))
              for i, row in enumerate(rows)]
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    actions = jnp.arange(len(rows), dtype=jnp.int32)+BUILD_START
    masks = jax.jit(jax.vmap(env.action_mask))(batched)
    assert jnp.all(masks[jnp.arange(len(rows)), actions])
    following, ts = jax.jit(jax.vmap(env.step))(batched, actions)
    chex.assert_trees_all_equal(following.gold, jnp.zeros(len(rows), jnp.int32))
    chex.assert_trees_all_equal(following.map_steps, batched.map_steps)
    chex.assert_trees_all_equal(following.movement_points, batched.movement_points)
    chex.assert_trees_all_equal(following.day, batched.day)
    chex.assert_trees_all_equal(following.hp, batched.hp)
    chex.assert_trees_all_equal(following.position, batched.position)
    chex.assert_trees_all_equal(following.battle_key, batched.battle_key)
    chex.assert_trees_all_equal(following.last_event, jnp.full(len(rows), BUILD, jnp.int32))
    chex.assert_trees_all_close(ts.reward, jnp.full(len(rows), .05-env.step_cost))
    assert jnp.all(following.built_today)
    assert jnp.all(ts.extras['building_constructed'])
    for i, row in enumerate(rows):
        assert int(following.buildings[i]) == int(states[i].buildings) | (1 << i)
        excluded = {j for j, other in enumerate(rows) if other['name'] in row['blocks']}
        excluded |= {j for j in range(len(rows)) if ancestors(rows, j) & excluded}
        assert int(following.blocked_buildings[i]) == sum(1 << j for j in excluded)
    chex.assert_tree_all_finite(ts.observation)


def test_build_mask_and_step_reject_every_restriction_and_unused_slot(env):
    state, _ = env.reset(jax.random.PRNGKey(42))
    states, actions = [], []
    for i, row in enumerate(env.construction.rows):
        ready = state.replace(buildings=jnp.uint32(sum(1 << j for j in ancestors(env.construction.rows, i))),
                              gold=jnp.int32(row['gold']))
        rejected = [ready.replace(gold=ready.gold-1), ready.replace(built_today=jnp.bool_(True)),
                    ready.replace(buildings=ready.buildings | jnp.uint32(1 << i)),
                    ready.replace(blocked_buildings=jnp.uint32(1 << i)),
                    ready.replace(in_battle=jnp.bool_(True)), ready.replace(done=jnp.bool_(True))]
        if row['requires']:
            rejected.append(ready.replace(buildings=jnp.uint32(0)))
        states.extend(rejected)
        actions.extend([BUILD_START+i] * len(rejected))
    for i in range(len(env.construction.rows), 25):
        states.append(state.replace(gold=jnp.int32(100_000)))
        actions.append(BUILD_START+i)
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    actions = jnp.asarray(actions, jnp.int32)
    mask = jax.jit(jax.vmap(env.action_mask))(batched)
    assert not jnp.any(mask[jnp.arange(len(actions)), actions])
    following, ts = jax.jit(jax.vmap(env.step))(batched, actions)
    for field in ('gold', 'buildings', 'blocked_buildings', 'built_today', 'map_steps', 'movement_points', 'day', 'hp', 'battle_key'):
        chex.assert_trees_all_equal(getattr(following, field), getattr(batched, field))
    assert jnp.all(ts.reward <= 0)
    assert not jnp.any(ts.extras['building_constructed'])


def test_one_build_per_turn_until_explicit_rest_and_branch_lock():
    env = NumberGrid(map_config=MAP)
    state, _ = env.reset(jax.random.PRNGKey(7))
    state = state.replace(movement_points=jnp.int32(2), gold=jnp.int32(2000))
    step = jax.jit(env.step)
    built, _ = step(state, jnp.int32(BUILD_START+4))
    assert built.movement_points == 2 and built.gold == 1800 and built.built_today
    assert not jnp.any(env.action_mask(built)[BUILD_START:REST]) and env.action_mask(built)[REST]
    exhausted, _ = step(built, jnp.int32(2))
    assert exhausted.movement_points == 0 and exhausted.built_today and exhausted.day == 1
    assert exhausted.gold == 1800
    next_day, ts = step(exhausted, jnp.int32(REST))
    assert next_day.map_steps == 1 and next_day.gold == 1900 and not next_day.built_today
    assert next_day.day == 2 and ts.reward == 0
    for index in (5, 8, 11):
        assert next_day.blocked_buildings & (1 << index)
    assert not env.action_mask(next_day)[BUILD_START+5]
    shrine, _ = step(next_day, jnp.int32(BUILD_START+7))
    assert shrine.gold == 1400 and shrine.movement_points == 20 and shrine.built_today
    assert shrine.buildings == (1 << 4) | (1 << 7)
    exhausted, _ = jax.jit(lambda s: jax.lax.scan(
        lambda s, a: (step(s, a)[0], None), s,
        jnp.tile(jnp.array([6, 2], jnp.int32), 5)))(shrine)
    assert exhausted.movement_points == 0 and exhausted.gold == 1400 and exhausted.built_today
    tomorrow, _ = step(exhausted, jnp.int32(REST))
    assert tomorrow.gold == 1500 and not tomorrow.built_today and tomorrow.day == 3
    assert tomorrow.buildings == shrine.buildings
    assert tomorrow.blocked_buildings == shrine.blocked_buildings


def test_scenario_locks_close_descendants_and_bad_settings_fail():
    env = NumberGrid(map_config={**MAP, 'blocked_buildings': ['Нечестивый портал'],
                                'max_building_level': 3})
    state, _ = env.reset(jax.random.PRNGKey(42))
    state = state.replace(gold=jnp.int32(100_000))
    mask = jax.jit(env.action_mask)(state)[BUILD_START:REST]
    for i in (0, 1, 2, 3, 9, 10, 11, 12, 17, 18, 19, 20, 21):
        assert not mask[i] and state.blocked_buildings & (1 << i)
    assert mask[4] and mask[22] and mask[23] and mask[24]
    for change in ({'faction': 'unknown'}, {'blocked_buildings': ['unknown']},
                   {'blocked_buildings': 'Храм'}, {'max_building_level': 0},
                   {'building_reward': float('nan')}, {'building_reward': -.1}):
        with pytest.raises(ValueError):
            BuildingRules({**MAP, **change})


def test_capital_survives_battle_recovery_and_resets_on_new_episode():
    env = NumberGrid(map_config=MAP)
    start, _ = env.reset(jax.random.PRNGKey(42))
    developed = start.replace(buildings=jnp.uint32(1 << 4), blocked_buildings=jnp.uint32(1 << 5),
                              built_today=jnp.bool_(True), gold=jnp.int32(100))
    battle = env._begin_battle(developed.replace(enemy=jnp.int32(0))).replace(actor=jnp.int32(0))
    won, _ = jax.jit(env._battle_step)(battle, jnp.int32(8), battle.battle_key, jnp.zeros(36))
    assert not won.in_battle
    escaping = battle.replace(hp=battle.hp.at[1:6].set(0), retreating=battle.retreating.at[0].set(True))
    escaped, _ = jax.jit(env._battle_step)(escaping, jnp.int32(17), escaping.battle_key, jnp.zeros(36))
    assert not escaped.in_battle
    for recovered in (won, escaped):
        for field in ('buildings', 'blocked_buildings', 'built_today', 'gold'):
            chex.assert_trees_all_equal(getattr(recovered, field), getattr(developed, field))
    reset, _ = env.reset(jax.random.PRNGKey(43))
    assert reset.buildings == reset.blocked_buildings == reset.gold == 0 and not reset.built_today


def test_training_autoreset_retains_final_build_in_terminal_observation():
    config = make_config()
    config.env.kwargs.max_steps = 3
    training, _ = make(config)
    state, _ = training.reset(jax.random.split(jax.random.PRNGKey(7), 2))
    advance = jax.jit(training.step)
    for _ in range(2):
        state, _ = advance(state, jnp.full(2, REST, jnp.int32))
    state, ts = advance(state, jnp.full(2, BUILD_START, jnp.int32))
    assert jnp.all(ts.truncated())
    chex.assert_trees_all_equal(state.buildings, jnp.zeros(2, jnp.uint32))
    chex.assert_trees_all_equal(state.built_today, jnp.zeros(2, jnp.bool_))
    final_obs = ts.extras['next_obs']['observation']
    np.testing.assert_array_equal(final_obs[:, -27], [1, 1])
    np.testing.assert_array_equal(final_obs[:, -28], [1, 1])
    np.testing.assert_array_equal(ts.observation['observation'][:, -28:-2], np.zeros((2, 26)))


def test_human_sessions_build_with_shared_actions_and_isolate_factions():
    from serve_number_grid import GameService
    service = GameService()
    sessions = []
    for faction, first_building in [('legions', 'Нечестивый портал'), ('elves', 'Пещера кентавров')]:
        game = service.create(42, faction)
        token = game['session']
        sessions.append(token)
        assert game['construction']['buildings'][0]['name'] == first_building
        assert game['map']['faction'] == faction
        assert len(game['snapshot']['action_mask']) == 44
        stored_faction, state, total = service.sessions[token]
        assert stored_faction == faction and state.gold == state.buildings == 0
        service.sessions[token] = (faction, state.replace(gold=jnp.int32(200)), total)
        built = service.act(token, BUILD_START)['snapshot']
        assert built['state']['gold'] == 0 and built['state']['buildings'] == 1
        assert built['building_status'][0] == 1
        assert built['total_reward'] == pytest.approx(.05-service.env.step_cost)
        rested = service.act(token, REST)
        assert rested['snapshot']['state']['day'] == 2
        assert rested['snapshot']['state']['gold'] == 100
        assert not rested['snapshot']['state']['built_today']
        assert rested['snapshot']['rest_penalty'] == pytest.approx(.02)
        assert rested['events'][0]['reward'] == pytest.approx(-.02)
        with pytest.raises(ValueError, match='недоступно'):
            service.act(token, BUILD_START)
    assert service.sessions[sessions[0]][0] == 'legions'
    assert service.sessions[sessions[1]][0] == 'elves'
    assert service.create(43)['snapshot']['state']['buildings'] == 0
