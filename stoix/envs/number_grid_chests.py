"""One-shot potion chests; Chebyshev pickup range from the Python reference.

The current scenario collects on a successful map move, per the user request.
See docs/MAP_CHESTS.md for sources and scenario choices.
"""
import jax.numpy as jnp

CHEST_LOOT = 21


class ChestRules:
    def __init__(self, game_map):
        chests = game_map['chests']
        if not isinstance(chests, list):
            raise ValueError('chests must be a list')
        positions, loot = [], []
        occupied = {tuple(p) for p in game_map['opponent_positions']}
        occupied.add(tuple(game_map['agent_position']))
        totals = list(game_map['initial_potions'])
        for chest in chests:
            if not isinstance(chest, dict):
                raise ValueError('Each chest must contain position and potions')
            position, contents = chest.get('position'), chest.get('potions')
            if (not isinstance(position, (list, tuple)) or len(position) != 2
                    or any(type(x) is not int or not 0 < x < game_map['size']-1 for x in position)
                    or tuple(position) in occupied):
                raise ValueError('Chest positions must be distinct, inside the map and unoccupied')
            if (not isinstance(contents, (list, tuple)) or len(contents) != 4
                    or any(type(n) is not int or n < 0 or n > 2**31-1 for n in contents)
                    or not any(contents)):
                raise ValueError('Chest potions must contain four nonnegative int32 counts and some loot')
            occupied.add(tuple(position))
            positions.append(position)
            loot.append(contents)
            totals = [a+b for a, b in zip(totals, contents)]
            if any(n > 2**31-1 for n in totals):
                raise ValueError('Chest loot plus initial inventory must fit int32')
        self.count = len(chests)
        self.positions = jnp.asarray(positions, jnp.int32).reshape(-1, 2)
        self.loot = jnp.asarray(loot, jnp.int32).reshape(-1, 4)
        self.features = jnp.concatenate((self.positions/(game_map['size']-1),
                                        self.loot.astype(jnp.float32)), axis=1)

    def collect(self, state, moved):
        nearby = jnp.max(jnp.abs(self.positions-state.position), axis=1) <= 1
        collected = state.chest_alive & nearby & moved
        loot = jnp.sum(jnp.where(collected[:, None], self.loot, 0), axis=0)
        return state.replace(chest_alive=state.chest_alive & ~collected,
                             potions=state.potions+loot, last_loot=loot,
                             last_event=jnp.where(jnp.any(collected), CHEST_LOOT, state.last_event))

    def observation(self, state):
        # Two normalized coordinates, four item counts and a live/collected flag.
        return jnp.concatenate((self.features, state.chest_alive[:, None]), axis=1).reshape(-1)
