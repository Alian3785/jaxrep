"""GPU evaluation and recording of actual NumberGrid policy trajectories."""

import jax
import jax.numpy as jnp

from stoix.environments.number_grid import RUNNING


def make_evaluator(env, actor_apply, episodes, mode="policy"):
    if mode not in ("policy", "greedy", "random"):
        raise ValueError(mode)

    @jax.jit
    def evaluate(params, key):
        key, reset_key = jax.random.split(key)
        states, timesteps = jax.vmap(env.reset)(jax.random.split(reset_key, episodes))
        returns = jnp.zeros(episodes, dtype=jnp.float32)

        def condition(carry):
            return jnp.any(carry[0].outcome == RUNNING)

        def advance(carry):
            states, observations, returns, key = carry
            key, action_key = jax.random.split(key)
            if mode == "random":
                actions = jax.random.randint(action_key, (episodes,), 0, 8)
            else:
                policy = actor_apply(params, observations)
                actions = policy.mode() if mode == "greedy" else policy.sample(seed=action_key)
            states, timesteps = jax.vmap(env.step)(states, actions)
            return states, timesteps.observation, returns + timesteps.reward, key

        states, _, returns, _ = jax.lax.while_loop(
            condition, advance, (states, timesteps.observation, returns, key)
        )
        return {
            "episode_return": returns,
            "episode_length": states.step_count,
            "outcome": states.outcome,
            "opponents_defeated": 3 - jnp.sum(states.opponent_alive.astype(jnp.int32), axis=-1),
        }

    return evaluate


def make_recorder(env, actor_apply, greedy=False):
    @jax.jit
    def record(params, key):
        key, reset_key = jax.random.split(key)
        state, timestep = env.reset(reset_key)

        def advance(carry, _):
            state, observation, key = carry
            key, action_key = jax.random.split(key)
            policy = actor_apply(params, observation[None, :])
            action = policy.mode()[0] if greedy else policy.sample(seed=action_key)[0]
            state, timestep = env.step(state, action)
            frame = {
                "position": state.agent_position,
                "strength": state.agent_strength,
                "alive": state.opponent_alive,
                "outcome": state.outcome,
                "step": state.step_count,
                "reward": timestep.reward,
                "action": action,
            }
            return (state, timestep.observation, key), frame

        _, frames = jax.lax.scan(advance, (state, timestep.observation, key), None, length=env.max_steps)
        return frames

    return record
