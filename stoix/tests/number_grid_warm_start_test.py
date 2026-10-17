"""Opt-in continuation regressions; run only through scripts/run_gpu.sh on request."""
from copy import deepcopy
import hashlib
import json

import jax
import pytest

from numbergrid_warm_start import prepare_warm_start


@pytest.fixture(scope='module', autouse=True)
def require_cuda():
    """Keep even metadata-only checks inside the project's CUDA test session."""
    assert jax.default_backend() == 'gpu', 'Use scripts/run_gpu.sh with CUDA'


def write_checkpoint(directory, config, steps):
    """Write a minimal manifest and payload without constructing a learner."""
    checkpoint = directory / 'checkpoints' / f'{steps:09d}.msgpack'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint payload for metadata validation')
    documents = {
        'checkpoints.json': [dict(path=str(checkpoint.relative_to(directory)), training_steps=steps,
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())],
        'config.json': config, 'map.json': {'name': 'same map'}, 'source_hashes.json': {},
    }
    for name, value in documents.items():
        (directory / name).write_text(json.dumps(value), encoding='utf-8')
    return checkpoint


@pytest.fixture(params=[True, False])
def continued_checkpoint(tmp_path, request):
    """Represent an interrupted child run with decayed or constant session rates."""
    config = dict(system=dict(actor_lr=.001, critic_lr=.002, decay_learning_rates=request.param,
                              epochs=4), network=dict(layers=[128, 128]),
                  arch=dict(num_envs=250, update_batch_size=1, num_updates_per_eval=10,
                            total_timesteps=1000))
    first = write_checkpoint(tmp_path / 'first', config, 400)
    _, metadata = prepare_warm_start(first, 2000, {'name': 'same map'}, config, tmp_path)
    child = deepcopy(config)
    child['system'].update(metadata['learning_rates'])
    child['arch']['total_timesteps'] = 1600
    child['warm_start'] = metadata
    checkpoint = write_checkpoint(tmp_path / 'child', child, 1200)
    return config, checkpoint, request.param


def test_repeated_continuation_preserves_base_rates_and_advances_schedule(tmp_path, continued_checkpoint):
    """A second warm-start uses elapsed child steps without rejecting its reduced rates."""
    config, checkpoint, decay = continued_checkpoint
    original = deepcopy(config)
    _, metadata = prepare_warm_start(checkpoint, 3000, {'name': 'same map'}, config, tmp_path)
    assert metadata['base_learning_rates'] == {'actor_lr': .001, 'critic_lr': .002}
    assert metadata['learning_rates'] == pytest.approx(
        {'actor_lr': .0003, 'critic_lr': .0006} if decay else {'actor_lr': .001, 'critic_lr': .002})
    assert metadata['initial_training_steps'] == 1200
    assert metadata['session_training_steps'] == 1800
    assert config == original


@pytest.mark.parametrize('decay', [True, False])
def test_finished_run_continues_from_final_weights_at_cumulative_rate(tmp_path, decay):
    """User request 2026-10-17: a finished run's params.msgpack, verified through results.json."""
    config = dict(system=dict(actor_lr=.001, critic_lr=.002, decay_learning_rates=decay, epochs=4),
                  network=dict(layers=[128, 128]),
                  arch=dict(num_envs=250, update_batch_size=1, num_updates_per_eval=10, total_timesteps=1000))
    run = tmp_path / 'finished'
    run.mkdir()
    params = run / 'params.msgpack'
    params.write_bytes(b'final weights')
    documents = {'results.json': dict(training_steps=1000,
                                      checkpoint_sha256=hashlib.sha256(b'final weights').hexdigest()),
                 'config.json': config, 'map.json': {'name': 'same map'}, 'source_hashes.json': {}}
    for name, value in documents.items():
        (run / name).write_text(json.dumps(value), encoding='utf-8')
    data, metadata = prepare_warm_start(params, 3000, {'name': 'same map'}, config, tmp_path)
    assert data == b'final weights' and metadata['initial_training_steps'] == 1000
    assert metadata['session_training_steps'] == 2000
    # One linear schedule over 3000 steps would be at 2/3 of the base rate after 1000.
    assert metadata['learning_rates'] == pytest.approx(
        {'actor_lr': .001*2/3, 'critic_lr': .002*2/3} if decay else {'actor_lr': .001, 'critic_lr': .002})
    params.write_bytes(b'tampered weights')
    with pytest.raises(ValueError, match='hash mismatch'):
        prepare_warm_start(params, 3000, {'name': 'same map'}, config, tmp_path)


@pytest.mark.parametrize('section,key,value,error', [
    ('system', 'actor_lr', .002, 'base learning rates'),
    ('system', 'critic_lr', .004, 'base learning rates'),
    ('system', 'epochs', 8, 'system configuration'),
    ('network', 'layers', [64, 64], 'network configuration'),
])
def test_chain_still_rejects_configuration_changes(tmp_path, continued_checkpoint, section, key, value, error):
    """Excluding session rates must not weaken the original configuration checks."""
    config, checkpoint, _ = continued_checkpoint
    changed = deepcopy(config)
    changed[section][key] = value
    with pytest.raises(ValueError, match=error):
        prepare_warm_start(checkpoint, 3000, {'name': 'same map'}, changed, tmp_path)
