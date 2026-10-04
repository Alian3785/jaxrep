"""Measured local run of the unchanged Stoix Anakin feedforward PPO learner."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('JAX_PLATFORMS', 'cuda')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf
from stoix.systems.ppo.anakin.ff_ppo import learner_setup
from stoix.utils.make_env import make
from stoix.envs.number_grid import MAP, NumberGrid, wrap_wall_action_mask
from numbergrid_config import ROOT, make_config


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def unreplicate(tree):
    return jax.tree.map(lambda x: x[0, 0], tree)


def evaluate(actor, params, episodes=1024, seed=1042, greedy=True, map_config=None):
    env = wrap_wall_action_mask(NumberGrid(map_config=map_config))
    @jax.jit
    def run(p):
        states, ts = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(seed), episodes))
        def condition(carry):
            return ~jnp.all(carry[0].done)
        def advance(carry):
            states, ts, returns, key = carry
            key, subkey = jax.random.split(key)
            dist = actor.apply(p, ts.observation)
            actions = dist.mode() if greedy else dist.sample(seed=subkey)
            states, ts = jax.vmap(env.step)(states, actions)
            return states, ts, returns + ts.reward, key
        states, ts, returns, _ = jax.lax.while_loop(
            condition, advance, (states, ts, jnp.zeros(episodes), jax.random.PRNGKey(seed + 1)))
        return states.won, states.step_count, returns
    won, lengths, returns = jax.device_get(run(params))
    return dict(episodes=episodes, successes=int(won.sum()), success_rate=float(won.mean()),
                mean_return=float(returns.mean()), mean_length=float(lengths.mean()),
                min_length=int(lengths.min()), max_length=int(lengths.max()),
                action_selection='argmax' if greedy else 'sample', seed=seed)


def train(args):
    started = time.perf_counter()
    if len(jax.devices()) != 1 or jax.devices()[0].platform != 'gpu':
        raise RuntimeError('This measured profile requires exactly one CUDA GPU.')
    total = 250_000 if args.smoke else args.total_timesteps
    game_map = json.loads(Path(args.map).read_text(encoding='utf-8')) if args.map else MAP
    config = make_config(total, args.seed, map_config=game_map)
    run_name = f"number_grid-{game_map['size']}x{game_map['size']}" + ('-step-cost' if game_map.get('step_cost', 0) else '') + ('-smoke' if args.smoke else '')
    if game_map.get('exploration_bonus', 0):
        run_name += '-exploration'
    if game_map.get('mask_walls', False):
        run_name += '-wall-mask'
    output = Path(args.output) if args.output else ROOT / 'results' / run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output already contains files; choose another --output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'config.json', OmegaConf.to_container(config, resolve=True))
    save_json(output / 'map.json', game_map)
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (output / 'packages.txt').write_text('\n'.join(packages) + '\n')
    def status(phase, **kwargs):
        record = dict(phase=phase, **kwargs)
        save_json(output / 'status.json', record)
        print(json.dumps(record), flush=True)
    status('initializing', device=str(jax.devices()[0]), total_timesteps=total)
    env, _ = make(config)
    keys = tuple(jax.random.split(jax.random.PRNGKey(args.seed), 3))
    learner, actor, state = learner_setup(env, keys, config)
    initial_params = jax.device_get(unreplicate(state.params))
    initial = evaluate(actor, unreplicate(state.params.actor_params), episodes=64, map_config=game_map)
    status('compiling', initial_evaluation=initial)
    def chunk(s):
        result = learner(s)
        losses = {k: jnp.mean(v) for k, v in result.train_metrics.items()}
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in result.train_metrics.values()]))
        episodes = result.episode_metrics
        mask = episodes['is_terminal_step']
        count = jnp.sum(mask)
        losses.update(
            all_finite=finite, episodes=count,
            mean_episode_return=jnp.sum(episodes['episode_return'] * mask) / jnp.maximum(count, 1),
            mean_episode_length=jnp.sum(episodes['episode_length'] * mask) / jnp.maximum(count, 1),
            # Exploration can increase return without a win: use the actual flag.
            episode_success_rate=jnp.sum(episodes['episode_success'] & mask) / jnp.maximum(count, 1),
        )
        return result.learner_state, losses
    jax.block_until_ready(state)
    begin_compile = time.perf_counter()
    compiled = jax.jit(chunk).lower(state).compile()
    compile_seconds = time.perf_counter() - begin_compile
    step_size = config.arch.num_updates_per_eval * config.arch.num_envs * config.system.rollout_length
    rows = []
    training_seconds = 0.0
    for i in range(total // step_size):
        begin = time.perf_counter()
        state, metrics = compiled(state)
        jax.block_until_ready((state, metrics))
        elapsed = time.perf_counter() - begin
        training_seconds += elapsed
        row = {k: float(v) for k, v in jax.device_get(metrics).items()}
        if not row['all_finite']:
            raise FloatingPointError('Non-finite learner metric')
        row.update(training_steps=(i + 1) * step_size, training_seconds=training_seconds)
        rows.append(row)
        status('training', training_steps=row['training_steps'], total_timesteps=total,
               mean_steps_per_second=row['training_steps']/training_seconds,
               mean_episode_return=row['mean_episode_return'])
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status('evaluating', training_seconds=training_seconds)
    params = unreplicate(state.params)
    for leaf in jax.tree.leaves(params):
        if not np.isfinite(np.asarray(leaf)).all():
            raise FloatingPointError('Non-finite parameter')
    changed = any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(initial_params), jax.tree.leaves(params)))
    if not changed:
        raise AssertionError('PPO did not update weights')
    expected_updates = config.arch.num_updates * config.system.epochs * config.system.num_minibatches
    actual_actor_updates = int(np.asarray(state.opt_states.actor_opt_state[1][0].count).reshape(-1)[0])
    actual_critic_updates = int(np.asarray(state.opt_states.critic_opt_state[1][0].count).reshape(-1)[0])
    if actual_actor_updates != expected_updates or actual_critic_updates != expected_updates:
        raise AssertionError('Unexpected optimizer update count')
    checkpoint = serialization.to_bytes(params)
    (output / 'params.msgpack').write_bytes(checkpoint)
    (output / 'learner_state.msgpack').write_bytes(serialization.to_bytes(state))
    restored = serialization.from_bytes(params, (output / 'params.msgpack').read_bytes())
    if not all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(restored))):
        raise AssertionError('Checkpoint roundtrip failed')
    final = evaluate(actor, restored.actor_params, episodes=1024, map_config=game_map)
    sampled = evaluate(actor, restored.actor_params, episodes=1024, greedy=False, map_config=game_map)
    source = ROOT / 'stoix/systems/ppo/anakin/ff_ppo.py'
    result = dict(
        environment=game_map['name'], algorithm='Stoix Anakin feedforward PPO', recurrent=False,
        device=jax.devices()[0].device_kind, backend=jax.default_backend(), jax=jax.__version__,
        cuda_runtime=importlib.metadata.version('nvidia-cuda-runtime'), python=platform.python_version(),
        training_steps=total, ppo_updates=config.arch.num_updates,
        actor_optimizer_updates=actual_actor_updates, critic_optimizer_updates=actual_critic_updates,
        seed=args.seed, training_seconds=training_seconds,
        mean_steps_per_second=total/training_seconds, ppo_compile_seconds=compile_seconds,
        initial_evaluation=initial, final_evaluation=final, sampled_evaluation=sampled,
        parameter_count=sum(x.size for x in jax.tree.leaves(params)),
        weights_changed=changed, checkpoint_roundtrip_verified=True,
        checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
        learner_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        total_runner_seconds=time.perf_counter()-started,
        timing='Sum of synchronized learner calls, including rollout and PPO updates; excludes compilation, evaluation, host logging and saving.',
        reconstruction='Fixed map and observation encoding recreated from README; remote weights and results were not recovered.',
    )
    save_json(output / 'results.json', result)
    status('complete', **result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true', help='250,000 transitions, separate output')
    p.add_argument('--total-timesteps', type=int, default=5_000_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output')
    p.add_argument('--map', help='Optional saved map JSON for reproducible comparisons')
    train(p.parse_args())
