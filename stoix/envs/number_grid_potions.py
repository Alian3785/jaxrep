"""Finite map-recovery inventory, shared by PPO and the human interface.

Effects: installed GItem.DBF and the Python reference potion catalogue.
See docs/MAP_POTIONS.md for source IDs and the user-defined starting stock.
"""
import jax.numpy as jnp

POTION_START, POTION_TYPES, POTION_TARGETS = 56, 4, 6
POTION_ACTIONS = POTION_TYPES * POTION_TARGETS
POTION_HEAL, POTION_REVIVE = 19, 20
POTIONS = (
    dict(key='healing', name='Банка исцеления', amount=50, effect='heal', game_id='G000IG0005'),
    dict(key='restoration', name='Бутыль лечения', amount=100, effect='heal', game_id='G000IG0006'),
    dict(key='ointment', name='Целебная мазь', amount=200, effect='heal', game_id='G000IG0018'),
    dict(key='life', name='Зелье воскрешения', amount=1, effect='revive', game_id='G000IG0001'),
)


class PotionRules:
    def __init__(self, initial_counts):
        if (not isinstance(initial_counts, (list, tuple)) or len(initial_counts) != POTION_TYPES
                or any(type(n) is not int or n < 0 or n > 2**31-1 for n in initial_counts)):
            raise ValueError('initial_potions must contain four nonnegative int32 counts')
        self.initial_counts = jnp.asarray(initial_counts, jnp.int32)
        self.amounts = jnp.asarray([p['amount'] for p in POTIONS], jnp.int32)

    def available(self, state, max_hp, potion=None, slot=None):
        """One predicate for the full (type, slot) mask and selected-action checks."""
        if potion is None:
            potion = jnp.arange(POTION_TYPES)[:, None]
            slot = jnp.arange(POTION_TARGETS)[None, :]
        hp, maximum = state.hp[slot], max_hp[slot]
        target = jnp.where(potion == 3, hp == 0, (hp > 0) & (hp < maximum))
        return (~state.done & ~state.in_battle & (state.potions[potion] > 0)
                & (maximum > 0) & target)

    def apply(self, state, action, max_hp):
        """Validated map use consumes exactly one item, even for a partial heal."""
        using = (action >= POTION_START) & (action < POTION_START+POTION_ACTIONS)
        index = jnp.clip(action-POTION_START, 0, POTION_ACTIONS-1)
        potion, slot = index // POTION_TARGETS, index % POTION_TARGETS
        reviving = potion == 3
        hp = jnp.where(reviving, 1, jnp.minimum(max_hp[slot], state.hp[slot]+self.amounts[potion]))
        amount = jnp.where(using, hp-state.hp[slot], 0)
        return state.replace(
            hp=state.hp.at[slot].add(amount),
            potions=state.potions.at[potion].add(-using.astype(jnp.int32)),
            last_potion=jnp.where(using, potion, -1),
            last_event=jnp.where(using, jnp.where(reviving, POTION_REVIVE, POTION_HEAL), state.last_event),
            last_target=jnp.where(using, slot, state.last_target),
            last_damage=jnp.where(using, amount, state.last_damage),
        )

    def observation(self, state):
        return state.potions / jnp.maximum(self.initial_counts, 1)

    def quotes(self, state, max_hp):
        """Display the actual HP effect on each target; legality is in the mask."""
        hp, maximum = state.hp[:6], max_hp[:6]
        healing = jnp.where(hp[None, :] > 0,
                            jnp.minimum(jnp.maximum(maximum-hp, 0)[None, :], self.amounts[:3, None]), 0)
        revive = ((hp == 0) & (maximum > 0)).astype(jnp.int32)
        return jnp.concatenate((healing, revive[None, :]))

    def metadata(self):
        counts = self.initial_counts.tolist()
        return [dict(p, initial_count=counts[i], action_start=POTION_START+i*POTION_TARGETS)
                for i, p in enumerate(POTIONS)]
