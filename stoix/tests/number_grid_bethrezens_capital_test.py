"""User-requested S101 capital challenge; extended CUDA tests, opt-in only.

Sources: docs/BETHREZENS_RETREAT.md, pinned S101FT0000 / S101KC0000,
and user-requested reward revision on 2026-10-15 (3 / 10 / 10 / 10 / -100).
Run only on explicit request through scripts/run_gpu.sh; not part of core.
"""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import CONTINUE, DEFEND, SHOOT, NumberGrid, load_map
from stoix.envs import number_grid_reference_rewards as rewards
from stoix.tests.number_grid_fixtures import compiled_method


@pytest.fixture(scope='module')
def capital():
    assert jax.default_backend() == 'gpu', 'Use scripts/run_gpu.sh with CUDA'
    env = NumberGrid(map_config=load_map('bethrezens_retreat_capital'))
    initial, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    return env, initial


def test_goal_requires_both_capital_defenders_and_ignores_old_exit(capital):
    env, initial = capital
    assert env.capital_objective and not env.position_objective
    assert env.map_config['reference_enemy_ids'][0] == 'S101FT0000'
    assert env.map_config['reference_enemy_ids'][7] == 'S101KC0000'
    assert jnp.where(env.goal_enemies)[0].tolist() == [0, 7]
    assert env.capital_objective_position.tolist() == [10, 11]
    assert bool(env.progression.capital_guards[env.progression.ids['g000uu3004']])
    won = compiled_method(env, '_campaign_won')
    assert not bool(won(initial.replace(position=jnp.array([2, 30], jnp.int32))))
    for enemy in (0, 7):
        assert not bool(won(initial.replace(alive=initial.alive.at[enemy].set(False))))
    cleared = initial.replace(alive=initial.alive.at[0].set(False).at[7].set(False))
    assert bool(won(cleared))  # no need to defeat the other 111 armies
    assert not bool(won(cleared.replace(lost=jnp.bool_(True))))
    assert not env.territory.objective_cities


def test_capital_keeps_intro_and_ambush_but_removes_exit_event(capital):
    env, initial = capital
    assert initial.position.tolist() == [23, 37]
    ids = [e['id'] for e in env.script_events.events]
    assert 'S101EV0012' not in ids
    assert {eid for eid, fired in zip(ids, initial.script_events_fired) if bool(fired)} == {
        'S101EV0009', 'S101EV0013', 'S101EV0014'}
    ambush = jax.jit(lambda s: env.script_events.process(env, s, locations=True))
    after = ambush(initial.replace(position=jnp.array([3, 22], jnp.int32)))
    assert bool(after.script_events_fired[ids.index('S101EV0008')])
    assert int(jnp.sum(after.alive)) == 108
    assert int(jnp.sum(initial.script_orders == 4)) == 7
    obs = compiled_method(env, 'observation')(initial)
    assert obs.shape == (env.observation_size,)
    assert jnp.allclose(obs[-7:-3], jnp.array([10, 11, -13, -26])/(env.size-1))
    assert obs[-3:].tolist() == [0., 0., 0.]


def test_goal_pays_102_once_without_exit_reward(capital):
    env, initial = capital
    exit_state = initial.replace(position=jnp.array([2, 30], jnp.int32))
    assert float(rewards.finish(env, initial, exit_state)[1]) == 0.
    won = initial.replace(won=jnp.bool_(True), done=jnp.bool_(True),
                          alive=initial.alive.at[0].set(False).at[7].set(False))
    paid, bonus = rewards.finish(env, initial, won)
    assert float(bonus) == pytest.approx(102.)
    assert float(rewards.finish(env, paid, paid)[1]) == 0.
    _, repeated = compiled_method(env, 'step')(paid, jnp.int32(CONTINUE))
    assert float(repeated.reward) == 0.


@pytest.fixture(scope='module')
def guardian_battle(capital):
    env, initial = capital
    return compiled_method(env, '_begin_battle')(
        initial.replace(enemy=jnp.int32(0), position=env.capital_objective_position),
        jax.random.PRNGKey(43), jnp.full(env.random_size, .5, jnp.float32))


def test_round_bonus_counts_full_rounds_and_excludes_other_battles(capital, guardian_battle):
    env, _ = capital
    state = guardian_battle.replace(round=jnp.int32(1))
    bonus = compiled_method(env, '_capital_round_bonus')
    assert float(bonus(state, state.hp, state.escaped, jnp.int32(1))) == 0.
    assert float(bonus(state, state.hp, state.escaped, jnp.int32(2))) == 3.
    assert float(bonus(state, state.hp, state.escaped, jnp.int32(3))) == 6.
    for excluded in (
        state.replace(enemy=jnp.int32(7)),  # visiting Death Knight, no Ashgan
        state.replace(enemy=jnp.int32(8)),
        state.replace(summon_battle=jnp.bool_(True)),
        state.replace(in_battle=jnp.bool_(False)),
        state.replace(round=jnp.int32(0)),
        state.replace(post_victory=jnp.int32(1)),
        state.replace(round=jnp.int32(env.max_rounds+1)),
        state.replace(hp=state.hp.at[7].set(0)),  # Ashgan already dead
    ):
        assert float(bonus(excluded, excluded.hp, excluded.escaped, excluded.round+1)) == 0.
    assert float(bonus(state, state.hp.at[:6].set(0), state.escaped, jnp.int32(2))) == 0.
    assert float(bonus(state, state.hp, state.escaped.at[:6].set(True), jnp.int32(2))) == 0.
    summons_only = state.replace(summon_owner=state.summon_owner.at[:6].set(4))
    assert float(bonus(summons_only, state.hp, state.escaped, jnp.int32(2))) == 0.


def test_actual_last_activation_pays_once_and_next_activation_does_not(capital, guardian_battle):
    env, _ = capital
    # The genuine hero takes the last activation of round 1, with both sides alive.
    state = guardian_battle.replace(actor=jnp.int32(4), round=jnp.int32(1),
        turn_phase=jnp.full(12, 2, jnp.int32).at[4].set(0),
        activation_done=jnp.ones(12, bool))
    battle = compiled_method(env, '_battle_step_regular')
    draws = jnp.full(env.random_size, .99, jnp.float32)
    after, reward = battle(state, jnp.int32(DEFEND), jax.random.PRNGKey(44), draws)
    assert int(after.round) == 2 and bool(after.in_battle)
    assert float(reward) == pytest.approx(3.0025)
    # Ashgan's following activation is inside round 2, not another completed round.
    _, reward = battle(after, jnp.int32(CONTINUE), jax.random.PRNGKey(45), draws)
    assert float(reward) == pytest.approx(.0025)


def test_first_guardian_contact_pays_once_at_actual_world_entry(capital):
    env, initial = capital
    # Attack north into the capital; a neighbouring tile alone earns nothing.
    before = initial.replace(position=env.capital_objective_position+jnp.array([1, 0]))
    world = compiled_method(env, '_world_step')
    key, draws = jax.random.PRNGKey(46), jnp.full(env.random_size, .5, jnp.float32)
    after, reward = world(before, jnp.int32(0), key, draws)
    already_paid = before.replace(capital_rewarded=before.capital_rewarded.at[0].set(True))
    _, reference = world(already_paid, jnp.int32(0), key, draws)
    assert bool(after.in_battle) and int(after.enemy) == 0
    assert after.capital_rewarded.tolist() == [True, False, False]
    assert float(reward-reference) == pytest.approx(10.)
    progress = compiled_method(env, '_capital_progress_bonus')
    assert float(progress(before, before.hp, jnp.bool_(True))[1]) == 0.
    assert float(progress(after, after.hp, jnp.bool_(True))[1]) == 0.


def test_hp_milestones_use_full_guardian_hp_and_exact_thresholds(capital, guardian_battle):
    env, _ = capital
    bonus = compiled_method(env, '_capital_progress_bonus')
    state = guardian_battle.replace(round=jnp.int32(1))
    assert int(state.hp[7]) == 900
    # Damage to other members of the garrison does not count as Ashgan damage.
    assert float(bonus(state, state.hp.at[6].set(0))[1]) == 0.
    for remaining, expected, flags in (
        (601, 0., [False, False, False]),
        (600, 10., [False, True, False]),
        (301, 10., [False, True, False]),
        (300, 20., [False, True, True]),
        (0, 20., [False, True, True]),  # a lethal hit can cross both thresholds
    ):
        after, amount = bonus(state, state.hp.at[7].set(remaining))
        assert float(amount) == expected
        assert after.capital_rewarded.tolist() == flags
    # A new battle may start wounded: the denominator remains the full 900 HP.
    wounded = state.replace(hp=state.hp.at[7].set(650),
                            enemy_initial_hp=state.enemy_initial_hp.at[1].set(650))
    assert float(bonus(wounded, wounded.hp.at[7].set(600))[1]) == 10.


def test_hp_milestones_persist_through_healing_and_reentry_but_reset(capital, guardian_battle):
    env, _ = capital
    bonus = compiled_method(env, '_capital_progress_bonus')
    state = guardian_battle.replace(round=jnp.int32(1))
    paid, amount = bonus(state, state.hp.at[7].set(600))
    assert float(amount) == 10.
    paid = paid.replace(hp=paid.hp.at[7].set(600))
    _, healing_reward = bonus(paid, paid.hp.at[7].set(900))
    assert float(healing_reward) == 0.
    reentered = compiled_method(env, '_begin_battle')(
        paid.replace(hp=paid.hp.at[7].set(900)), jax.random.PRNGKey(47),
        jnp.full(env.random_size, .5, jnp.float32))
    assert reentered.capital_rewarded.tolist() == [False, True, False]
    assert float(bonus(reentered, reentered.hp.at[7].set(600))[1]) == 0.
    second, amount = bonus(reentered, reentered.hp.at[7].set(300))
    assert float(amount) == 10.
    assert float(bonus(second, second.hp.at[7].set(0))[1]) == 0.
    obs = compiled_method(env, 'observation')(second)
    assert obs[-3:].tolist() == [0., 1., 1.]
    fresh, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(48))
    assert fresh.capital_rewarded.tolist() == [False, False, False]


def test_progress_excludes_other_battles_and_noncombat_phases(capital, guardian_battle):
    env, _ = capital
    bonus = compiled_method(env, '_capital_progress_bonus')
    state = guardian_battle.replace(round=jnp.int32(1))
    for excluded in (
        state.replace(enemy=jnp.int32(7)),  # other capital defender, without Ashgan
        state.replace(enemy=jnp.int32(8)),
        state.replace(enemy=jnp.int32(-1)),
        state.replace(summon_battle=jnp.bool_(True)),
        state.replace(in_battle=jnp.bool_(False)),
        state.replace(done=jnp.bool_(True)),
        state.replace(post_victory=jnp.int32(1)),
        state.replace(unit_ids=state.unit_ids.at[7].set(env.progression.ids['duke'])),
        state.replace(native_ids=state.native_ids.at[7].set(env.progression.ids['duke'])),
    ):
        after, amount = bonus(excluded, excluded.hp.at[7].set(0), jnp.bool_(True))
        assert float(amount) == 0.
        assert not bool(jnp.any(after.capital_rewarded))
    # Preparation counts for contact, but changing HP during it is not an attack.
    preparing = state.replace(round=jnp.int32(0))
    assert float(bonus(preparing, preparing.hp, jnp.bool_(True))[1]) == 10.
    assert float(bonus(preparing, preparing.hp.at[7].set(0))[1]) == 0.


def test_real_attack_pays_second_threshold_in_battle_reward(capital, guardian_battle):
    env, _ = capital
    state = guardian_battle.replace(actor=jnp.int32(4), round=jnp.int32(1),
        hp=guardian_battle.hp.at[7].set(301),
        capital_rewarded=jnp.array([True, True, False]))
    battle = compiled_method(env, '_battle_step_regular')
    key, draws = jax.random.PRNGKey(49), jnp.zeros(env.random_size, jnp.float32)
    after, reward = battle(state, jnp.int32(SHOOT+1), key, draws)
    _, already_paid = battle(state.replace(capital_rewarded=jnp.ones(3, bool)),
                             jnp.int32(SHOOT+1), key, draws)
    assert int(after.hp[7]) <= 300
    assert after.capital_rewarded.tolist() == [True, True, True]
    assert float(reward-already_paid) == pytest.approx(10.)


def test_timeout_penalty_is_100_once_and_keeps_other_terminal_rewards(capital):
    env, before = capital
    timeout = before.replace(done=jnp.bool_(True))
    paid, amount = rewards.finish(env, before, timeout)
    assert float(amount) == -100.
    assert float(rewards.finish(env, paid, paid)[1]) == 0.
    assert float(rewards.finish(env, before, timeout.replace(lost=jnp.bool_(True)))[1]) == -6.
    partial = timeout.replace(reference_defeated=timeout.reference_defeated.at[8].set(True))
    assert float(rewards.finish(env, before, partial)[1]) == pytest.approx(-100.+5./env.reference_enemy_count)
    _, repeated = compiled_method(env, 'step')(paid, jnp.int32(CONTINUE))
    assert float(repeated.reward) == 0.
