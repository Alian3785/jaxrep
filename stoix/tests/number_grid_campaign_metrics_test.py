"""Reference campaign telemetry: successful events, completion boundaries and windows."""
import jax
import jax.numpy as jnp
import pytest

from numbergrid_campaign import CampaignMetrics
from stoix.envs.number_grid import BUILD_START, DEFEND, REST, CONTINUE
from stoix.envs.number_grid_metrics import CAMPAIGN_FIELDS, summarize_campaign, campaign_events
from stoix.tests.number_grid_fixtures import compiled_method, hero_roster_state, enemy_roster_state

pytestmark = pytest.mark.core


def counters(timestep):
    return dict(zip(CAMPAIGN_FIELDS, map(float, timestep.extras['campaign_events'])))


def test_executed_build_invalid_repeat_and_timeout(current_game):
    env, initial, step, _ = current_game
    state = initial.replace(gold=jnp.int32(10000))
    mask = compiled_method(env, 'action_mask')(state)
    slot = int(jnp.flatnonzero(mask[BUILD_START:REST])[0])
    built, ts = step(state, jnp.int32(BUILD_START + slot))
    assert counters(ts)['buildings_built'] == 1
    _, rejected = step(built, jnp.int32(BUILD_START + slot))
    assert sum(counters(rejected).values()) == 0
    _, timeout = step(built.replace(step_count=jnp.int32(env.max_steps - 1)), jnp.int32(REST))
    assert counters(timeout)['timeouts'] == 1
    assert counters(timeout)['victories'] == counters(timeout)['defeats'] == 0


def battle(current_game):
    env, initial, step, _ = current_game
    state = compiled_method(env, '_begin_battle')(initial.replace(enemy=jnp.int32(11)))
    state = hero_roster_state(env, state, ['archer', None, None, None, None, None])
    state = enemy_roster_state(env, state, ['archer', None, None, None, None, None])
    return env, initial, step, state.replace(actor=jnp.int32(0), round=jnp.int32(3),
        turn_phase=jnp.full(12, 2, jnp.int32).at[0].set(0).at[6].set(0),
        priority=jnp.arange(12, dtype=jnp.float32), activation_done=jnp.zeros(12, bool))


def test_internal_dot_victory_counts_once(current_game):
    _, _, step, state = battle(current_game)
    state = state.replace(poison_mode=state.poison_mode.at[6].set(1),
                          poison_damage=state.poison_damage.at[6].set(10000))
    after, ts = step(state, jnp.int32(DEFEND))
    values = counters(ts)
    assert not after.in_battle and not after.lost
    assert values['battle_victories'] == values['enemy_squads_defeated'] == 1
    assert values['victories'] == values['battle_defeats'] == 0
    _, rested = step(after, jnp.int32(REST))
    assert counters(rested)['battle_victories'] == 0


def test_summon_defeat_is_not_campaign_defeat(current_game):
    _, initial, step, state = battle(current_game)
    state = state.replace(actor=jnp.int32(6), summon_battle=jnp.bool_(True),
        spell_party=jnp.stack((initial.unit_ids[:6], initial.unit_levels[:6], initial.unit_xp[:6], initial.hp[:6])),
        poison_mode=state.poison_mode.at[0].set(1), poison_damage=state.poison_damage.at[0].set(10000))
    after, ts = step(state, jnp.int32(CONTINUE))
    assert not after.in_battle and not after.lost
    assert counters(ts)['battle_defeats'] == 1
    assert counters(ts)['defeats'] == counters(ts)['battle_retreats'] == 0


def test_world_spell_survives_automatic_battle_and_stale_fields_are_ignored(current_game):
    env, initial, _, _ = current_game
    index = next(i for i, row in enumerate(env.casting.rows) if row['cast']['kind'] == 'summon_battle')
    world = initial.replace(last_cast_spell=jnp.int32(index))
    final = initial.replace(last_cast_spell=jnp.int32(-1))
    measure = jax.jit(lambda a, b, c, valid: campaign_events(env, a, b, c,
        jnp.int32(env.casting.start + index), valid, jnp.zeros(4, jnp.int32)))
    values = dict(zip(CAMPAIGN_FIELDS, map(float, measure(initial, world, final, jnp.bool_(True)))))
    assert values['summoned_units'] == values['spell_casts'] == 1
    values = measure(world, world, world, jnp.bool_(False))
    assert not jnp.any(values)


def test_gpu_reduction_keeps_every_total_and_latest_100_completions():
    n, fields = 130, len(CAMPAIGN_FIELDS)
    counts = jnp.zeros((n, 2, fields)).at[:, 0, CAMPAIGN_FIELDS.index('hired_units')].set(jnp.arange(n))
    counts = counts.at[:, 0, CAMPAIGN_FIELDS.index('victories')].set(1)
    terminal = jnp.zeros((n, 2), bool).at[:, 0].set(True)
    events = jnp.zeros_like(counts).at[:, 0].set(counts[:, 0])
    summary = jax.jit(summarize_campaign)({'is_terminal_step': terminal,
        'campaign_events': events, 'campaign_episode': counts,
        'episode_return': jnp.full(terminal.shape, 7.), 'episode_length': jnp.full(terminal.shape, 100)})
    tracker = CampaignMetrics()
    tracker.add(jax.device_get(summary))
    data = tracker.metrics(reset_window=True)
    assert data['campaign/episodes_total'] == 130
    assert data['campaign/hired_units_total'] == sum(range(130))
    assert data['campaign/hire_episodes_rate'] == pytest.approx(129 / 130)
    assert data['campaign/window/hired_units_mean'] == pytest.approx(sum(range(30, 130)) / 100)
    assert data['campaign/window/enemy_squads_defeated_sample_count'] == 100
    assert data['campaign/window/victory_rate'] == 1
    assert data['campaign/window/episode_return_mean'] == 7
    empty = tracker.metrics()
    assert empty['campaign/window/episodes'] == 0
    assert 'campaign/window/victory_rate' not in empty
    assert empty['campaign/window/hired_units_mean'] == data['campaign/window/hired_units_mean']


@pytest.mark.parametrize('training_autoreset', ['contract'], indirect=True)
def test_episode_counters_reset_with_environment(training_autoreset):
    env, _, step = training_autoreset
    state, _ = compiled_method(env, 'reset')(jax.random.split(jax.random.PRNGKey(42), 2))
    for _ in range(2):
        state, ts = step(state, jnp.full(2, REST, jnp.int32))
        assert jnp.all(ts.extras['episode_metrics']['is_terminal_step'])
        assert jnp.all(ts.extras['episode_metrics']['campaign_episode'][:, CAMPAIGN_FIELDS.index('timeouts')] == 1)
        assert not jnp.any(state.campaign_counts)
