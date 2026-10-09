"""Reporting regressions included in the project's GPU-only test session."""
from argparse import Namespace
from unittest.mock import Mock

import pytest

from numbergrid_tracking import NumberGridTracking


def tracker():
    return NumberGridTracking(Namespace(wandb_mode='disabled'))


def row(episodes, wins, steps=250_000, **extra):
    return {
        'episodes': float(episodes), 'episode_wins': float(wins),
        'episode_success_rate': wins / max(episodes, 1),
        'training_steps': steps, 'training_seconds': 1.0,
        'mean_episode_length': 123.0, 'mean_episode_return': 4.0,
        'all_finite': True, 'actor_loss': -0.1, **extra,
    }


def test_win_rate_counts_episodes_instead_of_averaging_windows():
    tracking = tracker()
    tracking.training_metrics(row(2, 2), 250_000, 1.0, 1.1)
    metrics = tracking.training_metrics(row(8, 0, 500_000), 250_000, 1.0, 2.2)
    assert metrics['episodes/completed_total'] == 10
    assert metrics['episodes/wins_total'] == 2
    assert metrics['episodes/win_rate'] == 0
    assert metrics['episodes/win_rate_total'] == pytest.approx(0.2)
    assert metrics['episodes/length_mean'] == 123


def test_empty_window_does_not_draw_false_zero_episode_means():
    tracking = tracker()
    metrics = tracking.training_metrics(row(0, 0), 250_000, 1.0, 1.1)
    assert metrics['episodes/completed'] == 0
    assert 'episodes/length_mean' not in metrics
    assert 'episodes/return_mean' not in metrics
    assert 'episodes/win_rate' not in metrics
    assert 'episodes/win_rate_total' not in metrics


def test_battle_wins_are_not_counted_as_whole_map_wins():
    tracking = tracker()
    metrics = tracking.training_metrics(
        row(10, 1, battle_victory=42, battle_transition=125_000), 250_000, 1.0, 1.1)
    assert metrics['episodes/wins'] == 1
    assert metrics['battles/wins'] == 42
    assert metrics['battles/transition_fraction'] == 0.5
    assert metrics['ppo/actor_loss'] == -0.1
    assert 'ppo/battle_victory' not in metrics


def test_failed_training_finishes_wandb_as_failed():
    tracking = tracker()
    run = Mock()
    tracking.run = run
    with pytest.raises(ValueError, match='training failed'):
        with tracking:
            raise ValueError('training failed')
    run.finish.assert_called_once_with(exit_code=1)


def test_disabled_tracking_preserves_metadata_without_creating_wandb_run(tmp_path):
    with tracker() as tracking:
        tracking.start(tmp_path, {}, {}, {})
        tracking.log({'training_steps': 0})
        assert tracking.run is None
    import json
    metadata = json.loads((tmp_path / 'wandb.json').read_text())
    assert metadata['mode'] == 'disabled'
    assert metadata['status'] == 'finished'


def test_million_step_summary_includes_all_episodes_with_weighted_means():
    tracking = tracker()
    tracking.run = Mock()
    for index, (episodes, wins, length) in enumerate(((1, 1, 10), (3, 0, 30), (0, 0, 0), (6, 2, 50))):
        summary = tracking.log_training(
            row(episodes, wins, (index + 1) * 250_000, mean_episode_length=length,
                mean_episode_return=length / 10, battle_victory=10, battle_transition=100_000, building_constructed=index+1,
                turn_ended=2*(index+1), rest_penalty=.01*(index+1)),
            250_000, 0.5, (index + 1) * 0.6)
        if index < 3:
            assert summary is None
            tracking.run.log.assert_not_called()
    tracking.run.log.assert_called_once()
    logged = tracking.run.log.call_args.args[0]
    assert logged['training_steps'] == 1_000_000
    assert logged['episodes/completed'] == 10
    assert logged['episodes/wins'] == 3
    assert logged['episodes/win_rate'] == pytest.approx(0.3)
    assert logged['episodes/length_mean'] == pytest.approx(40)
    assert logged['episodes/return_mean'] == pytest.approx(4)
    assert logged['battles/wins'] == 40
    assert logged['construction/built'] == logged['construction/built_total'] == 10
    assert logged['turns/rests'] == logged['turns/rests_total'] == 20
    assert logged['turns/rest_penalty'] == logged['turns/rest_penalty_total'] == pytest.approx(.1)
    assert 'ppo/turn_ended' not in logged and 'ppo/rest_penalty' not in logged
    assert 'ppo/building_constructed' not in logged
    assert logged['battles/transition_fraction'] == pytest.approx(0.4)
    assert logged['performance/steps_per_second'] == 500_000


def test_final_partial_window_is_logged_once_and_counts_are_preserved():
    tracking = tracker()
    tracking.run = Mock()
    for index in range(5):
        tracking.log_training(row(2, 1, (index + 1) * 250_000, turn_ended=3, rest_penalty=.015), 250_000, 0.5,
                              (index + 1) * 0.6, final=index == 4)
    calls = [call.args[0] for call in tracking.run.log.call_args_list]
    assert [call['training_steps'] for call in calls] == [1_000_000, 1_250_000]
    assert calls[0]['episodes/completed'] == 8
    assert calls[1]['episodes/completed'] == 2
    assert calls[1]['episodes/completed_total'] == 10
    assert calls[1]['episodes/wins_total'] == 5
    assert calls[1]['turns/rests'] == 3 and calls[1]['turns/rests_total'] == 15
    assert calls[1]['turns/rest_penalty'] == pytest.approx(.015)
    assert calls[1]['turns/rest_penalty_total'] == pytest.approx(.075)
    assert not tracking.pending


def test_smoke_run_shorter_than_interval_still_has_training_chart_point():
    tracking = tracker()
    tracking.run = Mock()
    tracking.log_training(row(4, 1), 250_000, 0.5, 0.6, final=True)
    tracking.run.log.assert_called_once()
    assert tracking.run.log.call_args.args[0]['episodes/wins'] == 1


@pytest.fixture
def comparison_runs(tmp_path):
    """Create complete run metadata without starting a learner."""
    import copy
    import json
    game_map = {'name': 'before', 'size': 48, 'enemy_rosters': [['squire']]}
    config = {'arch': {'seed': 42, 'total_timesteps': 20_000_000, 'num_envs': 250},
              'env': {'kwargs': {'map_config': game_map}},
              'system': {'epochs': 4, 'num_minibatches': 10},
              'network': {'layers': [128, 128]}}
    result = dict(seed=42, training_steps=20_000_000, device='test GPU', backend='gpu',
                  jax='0.8', python='3.12', cuda_runtime='13', num_envs=250, rollout_length=100,
                  mean_steps_per_second=200_000, weights_changed=True, checkpoint_roundtrip_verified=True)
    def write(name, *, map_config=None, config_change=None, result_change=None, packages=None):
        directory = tmp_path/name
        directory.mkdir(exist_ok=True)
        settings = copy.deepcopy(config)
        if map_config is not None:
            settings['env']['kwargs']['map_config'] = map_config
        if config_change:
            config_change(settings)
        for filename, data in [('results.json', {**result, **(result_change or {})}),
                               ('config.json', settings), ('map.json', settings['env']['kwargs']['map_config'])]:
            (directory/filename).write_text(json.dumps(data), encoding='utf-8')
        (directory/'packages.txt').write_text(packages or 'jax==0.8\noptax==0.2\nnvidia_cuda_runtime==13\n', encoding='utf-8')
        return directory
    return write, game_map


def test_comparison_accepts_normalized_packages_and_only_declared_map_change(comparison_runs):
    """Expected unit replacements remain comparable without hiding other configuration edits."""
    from scripts.run_unit_action_stage import map_changes, validate_comparison
    write, original = comparison_runs
    changed = {**original, 'name': 'after', 'enemy_rosters': [['witch']]}
    before = write('before')
    after = write('after', map_config=changed, packages=' OPTAX==0.2 \nNVIDIA-CUDA-Runtime==13\nJAX==0.8\n')
    declaration = map_changes(original, changed)
    validate_comparison(before, after, list(reversed(declaration)))
    with pytest.raises(ValueError, match='map changes'):
        validate_comparison(before, after)
    write('after', map_config={**changed, 'size': 49})
    with pytest.raises(ValueError, match='map changes'):
        validate_comparison(before, after, declaration)


def test_comparison_rejects_ppo_seed_budget_dependency_and_metadata_drift(comparison_runs):
    """Settings omitted by the former six-field check must invalidate speed comparisons."""
    import json
    from scripts.run_unit_action_stage import validate_comparison
    write, _ = comparison_runs
    before = write('before')
    for section, key, value in [('system', 'epochs', 2), ('system', 'num_minibatches', 5),
                                ('network', 'layers', [64, 64]), ('arch', 'seed', 43),
                                ('arch', 'total_timesteps', 5_000_000)]:
        after = write('after', config_change=lambda c, s=section, k=key, v=value: c[s].update({k: v}))
        with pytest.raises(ValueError, match='config.json'):
            validate_comparison(before, after)
    after = write('after', packages='jax==0.8\noptax==0.3\nnvidia_cuda_runtime==13\n')
    with pytest.raises(ValueError, match='packages.txt'):
        validate_comparison(before, after)
    after = write('after', result_change={'seed': 43})
    with pytest.raises(ValueError, match='seed'):
        validate_comparison(before, after)
    after = write('after')
    (after/'map.json').write_text(json.dumps({'name': 'unrelated'}))
    with pytest.raises(ValueError, match='disagree'):
        validate_comparison(before, after)


@pytest.mark.parametrize('mode', ['--baseline', '--baseline-run'])
def test_both_stage_modes_validate_saved_artifacts_before_reporting_regression(comparison_runs, monkeypatch, mode):
    """Both CLI paths reject incompatible runs before writing a regression percentage."""
    import sys
    from scripts import run_unit_action_stage as stage
    write, _ = comparison_runs
    before = write('before-42' if mode == '--baseline' else 'before')
    write('after-bench-42' if mode == '--baseline' else 'after-20m',
          config_change=lambda c: c['system'].update(epochs=2))
    monkeypatch.setattr(sys, 'argv', ['stage', 'after', '--root', str(before.parent), mode, 'before'])
    monkeypatch.setattr(stage.subprocess, 'run', lambda *a, **k: Namespace(returncode=0))
    with pytest.raises(ValueError, match='config.json'):
        stage.main()
    assert not (before.parent/'after-comparison.json').exists()
