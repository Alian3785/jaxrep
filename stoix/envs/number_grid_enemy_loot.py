"""One-shot inventories of defeated stacks/forts, entirely within the JAX step.

Source: native CMidStack/CCapital/CMidVillage ITEM_ID inventories. See
docs/EMPIRE_ALLIANCE_WAR.md for automatic collection in the single-party port.
"""
import jax.numpy as jnp


class EnemyLootRules:
    def __init__(self, game_map, potions, items):
        entries = game_map.get('enemy_loot', [])
        if not isinstance(entries, list):
            raise ValueError('enemy_loot must be a list')
        count = len(game_map['opponent_positions'])
        for row in entries:
            guards = row.get('guards', [])
            if (not guards or len(set(guards)) != len(guards)
                    or any(type(i) is not int or not 0 <= i < count for i in guards)):
                raise ValueError('Enemy loot needs distinct valid guard indices')
        self.guards = jnp.array([[i in row['guards'] for i in range(count)] for row in entries], bool)
        self.potions = jnp.array([[row.get('potions', {}).get(k, 0) for k in potions.keys] for row in entries], jnp.int32)
        self.items = items
        if items is not None:
            self.contents = jnp.array([[row.get('items', {}).get(k, 0) for k in items.keys] for row in entries], jnp.int32)
            tokens = [(i, items.keys.index(k)) for i, row in enumerate(entries)
                      for k, n in row.get('items', {}).items() for _ in range(n)]
            self.sources = jnp.array([i for i, _ in tokens], jnp.int32)
            self.tokens = jnp.array([i for _, i in tokens], jnp.int32)

    def grant(self, before, after):
        # The alive -> defeated transition is the receipt: invalid actions,
        # subsequent turns and previously empty forts cannot pay a second time.
        was_guarded = jnp.any(self.guards & before.alive, axis=1)
        guarded = jnp.any(self.guards & after.alive, axis=1)
        collected = was_guarded & ~guarded
        potions = jnp.sum(jnp.where(collected[:, None], self.potions, 0), axis=0)
        if self.items is not None:
            previous_loot = after.last_item_loot
            after = self.items._collect(after, collected, self.sources, self.tokens, self.contents)
            if self.tokens.size:
                after = after.replace(last_item_loot=previous_loot + after.last_item_loot)
        return after.replace(potions=after.potions + potions, last_loot=after.last_loot + potions)
