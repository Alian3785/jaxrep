"""Benchmark Stoix PPO on full Craftax, including rollout and gradient updates."""

import time

PROCESS_START = time.perf_counter()

import argparse
import csv
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.systems.ppo.anakin.ff_ppo import learner_setup
from stoix.utils.make_env import make
from stoix.utils.total_timestep_checker import check_total_timesteps


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = ROOT / ("smoke-craftax-results" if args.smoke else "results/craftax-5m")
    output.mkdir(parents=True, exist_ok=True)
    overrides = [
        "env=craftax/symbolic",
        "arch.total_num_envs=1000",
        "arch.total_timesteps=5000000",
        "arch.num_evaluation=10",
        "arch.num_eval_episodes=64",
        "arch.absolute_metric=False",
        "system.rollout_length=50",
        "env.reset_ratio=20",
    ]
    if args.smoke:
        overrides += [
            "arch.total_num_envs=8",
            "arch.total_timesteps=64",
            "arch.num_evaluation=1",
            "arch.num_eval_episodes=4",
            "system.rollout_length=4",
            "system.epochs=2",
            "system.num_minibatches=2",
        ]
    with initialize_config_dir(
        version_base=None, config_dir=str(ROOT / "stoix/configs/default/anakin")
    ):
        config = compose(config_name="default_ff_ppo", overrides=overrides)
    OmegaConf.set_struct(config, False)
    devices = jax.devices()
    assert all(device.platform == "gpu" for device in devices), devices
    config.num_devices = len(devices)
    config = check_total_timesteps(config)
    transitions_per_chunk = (
        config.arch.total_num_envs
        * config.system.rollout_length
        * config.arch.num_updates_per_eval
    )
    expected_steps = transitions_per_chunk * config.arch.num_evaluation
    assert expected_steps == config.arch.total_timesteps
    assert (
        config.arch.num_envs * config.system.rollout_length
    ) % config.system.num_minibatches == 0
    print(f"GPU: {devices}; exact transition budget: {expected_steps:,}", flush=True)

    setup_start = time.perf_counter()
    env, eval_env = make(config)
    key, eval_key, actor_key, critic_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), 4
    )
    learn, actor, learner_state = learner_setup(env, (key, actor_key, critic_key), config)
    jax.block_until_ready(learner_state)
    evaluator, _, (eval_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=eval_key,
        eval_act_fn=get_distribution_act_fn(config, actor.apply),
        params=learner_state.params.actor_params,
        config=config,
    )
    jax.block_until_ready((eval_params, eval_keys))
    setup_seconds = time.perf_counter() - setup_start
    OmegaConf.save(config, output / "config.yaml", resolve=True)

    print("Compiling Stoix PPO learner...", flush=True)
    compile_start = time.perf_counter()
    compiled_learn = learn.lower(learner_state).compile()
    training_compile_seconds = time.perf_counter() - compile_start
    print(f"PPO compile: {training_compile_seconds:.3f} s", flush=True)

    print("Compiling evaluator...", flush=True)
    compile_start = time.perf_counter()
    compiled_eval = evaluator.lower(eval_params, eval_keys, None).compile()
    evaluation_compile_seconds = time.perf_counter() - compile_start
    print(f"Evaluation compile: {evaluation_compile_seconds:.3f} s", flush=True)

    rows = []
    training_seconds = 0.0
    evaluation_seconds = 0.0
    training_episodes = 0
    training_successes = 0
    loop_start = time.perf_counter()
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = None
        for chunk in range(config.arch.num_evaluation):
            start = time.perf_counter()
            result = compiled_learn(learner_state)
            jax.block_until_ready(result)
            chunk_seconds = time.perf_counter() - start
            training_seconds += chunk_seconds
            learner_state = result.learner_state

            episode_metrics = jax.device_get(result.episode_metrics)
            final = episode_metrics["is_terminal_step"].astype(bool)
            returns = episode_metrics["episode_return"][final]
            completed = int(np.sum(final))
            successes = int(np.sum(returns > 0.0))
            training_episodes += completed
            training_successes += successes
            losses = {
                name: float(np.mean(value))
                for name, value in jax.device_get(result.train_metrics).items()
            }
            assert all(np.isfinite(value) for value in losses.values()), losses

            eval_key, *keys = jax.random.split(eval_key, len(devices) + 1)
            eval_keys = jnp.stack(keys).reshape(len(devices), -1)
            eval_params = jax.tree.map(lambda x: x[:, 0], learner_state.params.actor_params)
            jax.block_until_ready((eval_params, eval_keys))
            start = time.perf_counter()
            evaluation = compiled_eval(eval_params, eval_keys, None)
            jax.block_until_ready(evaluation)
            eval_seconds = time.perf_counter() - start
            evaluation_seconds += eval_seconds
            metrics = jax.device_get(evaluation.episode_metrics)
            row = {
                "timesteps": (chunk + 1) * transitions_per_chunk,
                "chunk_training_seconds": chunk_seconds,
                "cumulative_training_seconds": training_seconds,
                "training_steps_per_second": transitions_per_chunk / chunk_seconds,
                "episodes_completed": completed,
                "positive_return_training_episodes": successes,
                "training_mean_return": float(np.mean(returns)) if completed else None,
                "evaluation_seconds": eval_seconds,
                "evaluation_mean_return": float(np.mean(metrics["episode_return"])),
                "evaluation_mean_length": float(np.mean(metrics["episode_length"])),
                **losses,
            }
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            print(
                f"{row['timesteps']:,}/{expected_steps:,}: "
                f"{row['training_steps_per_second']:,.0f} steps/s, "
                f"eval return {row['evaluation_mean_return']:.4f}, "
                f"eval length {row['evaluation_mean_length']:.1f}",
                flush=True,
            )
    training_loop_wall_seconds = time.perf_counter() - loop_start

    checkpoint_start = time.perf_counter()
    # Save actor, critic, optimizer states and RNG. Environment batches are omitted.
    checkpoint = jax.device_get(
        jax.tree.map(
            lambda x: x[0, 0],
            {"params": learner_state.params, "opt_states": learner_state.opt_states,
             "key": learner_state.key},
        )
    )
    checkpoint_path = output / "model.msgpack"
    checkpoint_path.write_bytes(serialization.to_bytes(checkpoint))
    restored = serialization.from_bytes(checkpoint, checkpoint_path.read_bytes())
    for original, actual in zip(jax.tree.leaves(checkpoint), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(original, actual)
        assert np.all(np.isfinite(actual))
    optimizer_steps = int(checkpoint["opt_states"].actor_opt_state[1][0].count)
    critic_optimizer_steps = int(checkpoint["opt_states"].critic_opt_state[1][0].count)
    expected_optimizer_steps = (
        config.arch.num_updates * config.system.epochs * config.system.num_minibatches
    )
    assert optimizer_steps == expected_optimizer_steps, (optimizer_steps, expected_optimizer_steps)
    assert critic_optimizer_steps == expected_optimizer_steps
    assert rows[-1]["timesteps"] == expected_steps
    checkpoint_seconds = time.perf_counter() - checkpoint_start

    summary = {
        "environment": config.env.scenario.name,
        "algorithm": "Stoix Anakin feedforward PPO",
        "device": str(devices[0]),
        "gpu_name": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip(),
        "stoix_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {name: version(name) for name in
            ("stoix", "stoa-env", "craftax", "gymnax", "jax", "jaxlib", "flax", "optax", "distrax",
             "nvidia-cuda-runtime", "nvidia-cudnn-cu13")},
        "seed": config.arch.seed,
        "total_training_timesteps": expected_steps,
        "parallel_environments": config.arch.total_num_envs,
        "rollout_length": config.system.rollout_length,
        "ppo_updates": config.arch.num_updates,
        "ppo_epochs": config.system.epochs,
        "num_minibatches": config.system.num_minibatches,
        "optimizer_steps_actor": optimizer_steps,
        "optimizer_steps_critic": critic_optimizer_steps,
        "use_optimistic_reset": config.env.use_optimistic_reset,
        "reset_ratio": min(config.env.reset_ratio, config.arch.num_envs),
        "observation_shape": list(env.observation_space().shape),
        "actor_parameters": sum(x.size for x in jax.tree.leaves(checkpoint["params"].actor_params)),
        "critic_parameters": sum(x.size for x in jax.tree.leaves(checkpoint["params"].critic_params)),
        "setup_seconds": setup_seconds,
        "training_compile_seconds": training_compile_seconds,
        "evaluation_compile_seconds": evaluation_compile_seconds,
        "training_seconds": training_seconds,
        "training_steps_per_second": expected_steps / training_seconds,
        "training_seconds_including_compile": training_seconds + training_compile_seconds,
        "training_steps_per_second_including_compile":
            expected_steps / (training_seconds + training_compile_seconds),
        "training_loop_wall_seconds": training_loop_wall_seconds,
        "evaluation_seconds": evaluation_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "process_wall_seconds": time.perf_counter() - PROCESS_START,
        "training_episodes_completed": training_episodes,
        "positive_return_training_episodes": training_successes,
        "evaluation_episodes_per_chunk": config.arch.num_eval_episodes,
        "final_evaluation_mean_return": rows[-1]["evaluation_mean_return"],
        "final_evaluation_mean_length": rows[-1]["evaluation_mean_length"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_roundtrip_verified": True,
        "timing_method": "perf_counter; block_until_ready on full learner output; "
            "exact AOT compilation; rollout + actor/critic gradient updates included; "
            "evaluation, logging, checkpointing excluded from training_seconds",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
