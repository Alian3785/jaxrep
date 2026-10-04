"""A fully observable, pure-JAX 16x16 grid with strength-based encounters."""

from typing import Optional

import jax
import jax.numpy as jnp
from flax import struct
from stoa import Environment, StepType, TimeStep
from stoa.core_wrappers.wrapper import Wrapper
from stoa.spaces import ArraySpace, BoundedArraySpace, DictSpace, DiscreteSpace


# Clockwise from north. Coordinates are (row, column), zero based.
MOVES = jnp.array(
    [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)],
    dtype=jnp.int32,
)
ACTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
RUNNING, WON, LOST, TIMED_OUT = 0, 1, 2, 3


@struct.dataclass
class NumberGridState:
    walls: jax.Array
    agent_position: jax.Array
    agent_strength: jax.Array
    opponent_positions: jax.Array
    opponent_strengths: jax.Array
    opponent_alive: jax.Array
    step_count: jax.Array
    outcome: jax.Array


class NumberGrid(Environment):
    """Eight movement actions; encounters are resolved on adjacent cells.

    A stronger/equal adjacent opponent causes immediate defeat, before any
    rewards or strength changes. Otherwise every weaker adjacent opponent is
    removed simultaneously. Reset is explicit; Stoix adds its auto-reset wrapper.
    """

    def __init__(
        self,
        size: int = 16,
        max_steps: int = 2000,
        adjacent_diagonals: bool = True,
        additive_victory_reward: bool = True,
    ):
        super().__init__()
        if size != 16:
            raise ValueError("This task uses a fixed 16x16 map.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.size = size
        self.max_steps = max_steps
        self.adjacent_diagonals = adjacent_diagonals
        self.additive_victory_reward = additive_victory_reward
        self.maximum_return = 6.0 if additive_victory_reward else 5.0
        self.observation_size = size * size + 2 + 1 + 3 * 7 + 1 + 8

    def reset(self, rng_key: jax.Array, env_params: Optional[object] = None):
        del rng_key, env_params  # Fixed layout, independent of the RNG.
        rows, columns = jnp.indices((self.size, self.size))
        walls = (rows == 0) | (columns == 0) | (rows == self.size - 1) | (columns == self.size - 1)
        state = NumberGridState(
            walls=walls,
            agent_position=jnp.array([14, 1], dtype=jnp.int32),
            agent_strength=jnp.array(1, dtype=jnp.int32),
            opponent_positions=jnp.array([[13, 6], [7, 12], [2, 3]], dtype=jnp.int32),
            opponent_strengths=jnp.array([0, 1, 2], dtype=jnp.int32),
            opponent_alive=jnp.ones(3, dtype=jnp.bool_),
            step_count=jnp.array(0, dtype=jnp.int32),
            outcome=jnp.array(RUNNING, dtype=jnp.int32),
        )
        return state, self._timestep(state, jnp.float32(0), jnp.int32(0), first=True)

    def legal_actions(self, state: NumberGridState):
        targets = state.agent_position[None, :] + MOVES
        clipped = jnp.clip(targets, 0, self.size - 1)
        in_bounds = jnp.all((targets >= 0) & (targets < self.size), axis=-1)
        is_wall = state.walls[clipped[:, 0], clipped[:, 1]]
        occupied = jnp.any(
            jnp.all(targets[:, None, :] == state.opponent_positions[None, :, :], axis=-1)
            & state.opponent_alive[None, :],
            axis=-1,
        )
        return in_bounds & ~is_wall & ~occupied

    def observe(self, state: NumberGridState):
        # Full state features, including the map. No hidden information.
        alive = state.opponent_alive.astype(jnp.float32)
        opponent_features = jnp.concatenate(
            [
                state.opponent_positions.astype(jnp.float32) / (self.size - 1),
                (state.opponent_positions - state.agent_position).astype(jnp.float32) / (self.size - 1),
                state.opponent_strengths[:, None].astype(jnp.float32) / 2,
                alive[:, None],
                (state.agent_strength > state.opponent_strengths)[:, None].astype(jnp.float32),
            ],
            axis=-1,
        ) * alive[:, None]
        return jnp.concatenate(
            [
                state.walls.reshape(-1).astype(jnp.float32),
                state.agent_position.astype(jnp.float32) / (self.size - 1),
                jnp.array([state.agent_strength / 4], dtype=jnp.float32),
                opponent_features.reshape(-1),
                jnp.array([state.step_count / self.max_steps], dtype=jnp.float32),
                self.legal_actions(state).astype(jnp.float32),
            ]
        )

    def _timestep(self, state, reward, defeated, first=False):
        terminated = (state.outcome == WON) | (state.outcome == LOST)
        truncated = state.outcome == TIMED_OUT
        step_type = jnp.where(
            terminated, StepType.TERMINATED,
            jnp.where(truncated, StepType.TRUNCATED, StepType.MID),
        )
        if first:
            step_type = StepType.FIRST
        return TimeStep(
            step_type=step_type,
            reward=jnp.asarray(reward, dtype=jnp.float32),
            discount=jnp.where(terminated, jnp.float32(0), jnp.float32(1)),
            observation=self.observe(state),
            extras={
                "outcome": state.outcome,
                "opponents_defeated": jnp.asarray(defeated, dtype=jnp.int32),
            },
        )

    def step(self, state: NumberGridState, action: jax.Array, env_params=None):
        del env_params

        def advance(current):
            action_index = jnp.clip(action, 0, 7)
            valid = (action >= 0) & (action < 8) & self.legal_actions(current)[action_index]
            position = jnp.where(valid, current.agent_position + MOVES[action_index], current.agent_position)
            difference = jnp.abs(current.opponent_positions - position)
            if self.adjacent_diagonals:
                adjacent = jnp.max(difference, axis=-1) == 1
            else:
                adjacent = jnp.sum(difference, axis=-1) == 1
            adjacent = adjacent & current.opponent_alive
            lost = jnp.any(adjacent & (current.agent_strength <= current.opponent_strengths))
            defeated_mask = adjacent & ~lost
            defeated = jnp.sum(defeated_mask.astype(jnp.int32))
            alive = current.opponent_alive & ~defeated_mask
            won = ~lost & ~jnp.any(alive)
            count = current.step_count + 1
            timeout = (count >= self.max_steps) & ~lost & ~won
            outcome = jnp.where(lost, LOST, jnp.where(won, WON, jnp.where(timeout, TIMED_OUT, RUNNING)))
            new_state = current.replace(
                agent_position=position,
                agent_strength=current.agent_strength + defeated,
                opponent_alive=alive,
                step_count=count,
                outcome=outcome.astype(jnp.int32),
            )
            win_reward = defeated.astype(jnp.float32) + 3.0 if self.additive_victory_reward else jnp.float32(3)
            reward = jnp.where(lost, -1.0, jnp.where(won, win_reward, defeated.astype(jnp.float32)))
            return new_state, self._timestep(new_state, reward, defeated)

        # Terminal states are absorbing. Reset/auto-reset is a separate operation.
        return jax.lax.cond(
            state.outcome == RUNNING,
            advance,
            lambda current: (current, self._timestep(current, jnp.float32(0), jnp.int32(0))),
            state,
        )

    def observation_space(self, env_params=None):
        return BoundedArraySpace((self.observation_size,), jnp.float32, -1, 1, name="number_grid")

    def action_space(self, env_params=None):
        return DiscreteSpace(8, dtype=jnp.int32, name="movement")

    def state_space(self, env_params=None):
        # Spaces describe the fields of NumberGridState's fixed-shape PyTree.
        return DictSpace({
            "walls": ArraySpace((16, 16), jnp.bool_),
            "agent_position": BoundedArraySpace((2,), jnp.int32, 0, 15),
            "agent_strength": BoundedArraySpace((), jnp.int32, 1, 4),
            "opponent_positions": BoundedArraySpace((3, 2), jnp.int32, 0, 15),
            "opponent_strengths": BoundedArraySpace((3,), jnp.int32, 0, 2),
            "opponent_alive": ArraySpace((3,), jnp.bool_),
            "step_count": BoundedArraySpace((), jnp.int32, 0, self.max_steps),
            "outcome": BoundedArraySpace((), jnp.int32, RUNNING, TIMED_OUT),
        })


class RecordNumberGridMetrics(Wrapper):
    """Keep exact win/loss/timeout counts in Stoix rollout metrics."""

    @staticmethod
    def _add(timestep):
        metrics = {
            **timestep.extras["episode_metrics"],
            "won_episode": timestep.extras["outcome"] == WON,
            "lost_episode": timestep.extras["outcome"] == LOST,
            "timed_out_episode": timestep.extras["outcome"] == TIMED_OUT,
            "opponents_defeated": timestep.extras["opponents_defeated"],
        }
        return timestep.replace(extras={**timestep.extras, "episode_metrics": metrics})

    def reset(self, rng_key, env_params=None):
        state, timestep = self._env.reset(rng_key, env_params)
        return state, self._add(timestep)

    def step(self, state, action, env_params=None):
        state, timestep = self._env.step(state, action, env_params)
        return state, self._add(timestep)
