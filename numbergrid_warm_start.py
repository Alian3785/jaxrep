"""Validated weights-only continuation; optimizer and episodes start afresh."""
import hashlib
import json
from pathlib import Path


def prepare_warm_start(checkpoint, total, game_map, config, root):
    """Validate a checkpoint and continue its LR schedule without restoring optimizer state."""
    path = Path(checkpoint).resolve()
    parent = path.parent.parent
    records = json.loads((parent / 'checkpoints.json').read_text())
    record = next((r for r in records if (parent / r['path']).resolve() == path), None)
    if record is None:
        raise ValueError('Warm-start checkpoint is absent from its saved manifest')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('Warm-start checkpoint hash mismatch')
    initial = record['training_steps']
    if not 0 < initial < total:
        raise ValueError('Total budget must exceed the checkpoint training steps')
    if json.loads((parent / 'map.json').read_text()) != game_map:
        raise ValueError('Warm-start map differs from the saved training map')
    hashes = json.loads((parent / 'source_hashes.json').read_text())
    # Only orchestration/logging changed to support this continuation. All
    # recorded environment, learner and configuration sources must still match.
    orchestration = {'benchmark_number_grid.py', 'numbergrid_tracking.py', 'numbergrid_warm_start.py'}
    changed = [name for name, digest in hashes.items()
               if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    if any(name not in orchestration for name in changed):
        raise ValueError(f'Warm-start source incompatibility: {changed}')
    previous = json.loads((parent / 'config.json').read_text())
    rate_names = ('actor_lr', 'critic_lr')
    parent_warm_start = previous.get('warm_start') or {}
    base_rates = parent_warm_start.get('base_learning_rates',
        {name: previous['system'][name] for name in rate_names})
    if {name: config['system'][name] for name in rate_names} != base_rates:
        raise ValueError('Warm-start base learning rates differ')
    # A child run stores decayed session rates in system. Compare the original
    # rates separately so another continuation still validates its configuration.
    system = {key: value for key, value in config['system'].items() if key not in rate_names}
    previous_system = {key: value for key, value in previous['system'].items() if key not in rate_names}
    if system != previous_system:
        raise ValueError('Warm-start system configuration differs')
    if config['network'] != previous['network']:
        raise ValueError('Warm-start network configuration differs')
    for name in ('num_envs', 'update_batch_size', 'num_updates_per_eval'):
        if config['arch'][name] != previous['arch'][name]:
            raise ValueError(f'Warm-start configuration differs: {name}')
    parent_initial = parent_warm_start.get('initial_training_steps', 0)
    fraction = (initial - parent_initial) / previous['arch']['total_timesteps']
    if not 0 <= fraction < 1:
        raise ValueError('Weights-only continuation requires an unfinished parent schedule')
    rates = {name: previous['system'][name] * (
        1 - fraction if previous['system']['decay_learning_rates'] else 1)
        for name in rate_names}
    return data, dict(
        checkpoint=str(path), parent_results=str(parent), checkpoint_sha256=record['sha256'],
        initial_training_steps=initial, target_training_steps=total,
        session_training_steps=total-initial, optimizer_restored=False,
        environment_state_restored=False, source_changes=changed,
        base_learning_rates=base_rates, learning_rates=rates,
        learning_rate_schedule='Continue from parent next-update rate, decay over remaining budget')
