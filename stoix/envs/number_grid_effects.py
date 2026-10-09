"""Fixed-shape, source-aware combat effects; all runtime work stays in JAX."""
import jax.numpy as jnp

CAPITAL_GUARDS = {'Ашган', 'Ашкаэль', 'Видар', 'Мизраэль', 'Иллюмиэлль'}


def source_protection(eligible, source, traits, used, slots):
    """Secondary source is 1-based, or zero for an effect without a source.

    A missed/ineligible effect never spends a ward. Immunity takes precedence.
    Return valid hits, updated spent-source bits, and display slot bitsets.
    """
    bit = jnp.where(source > 0, jnp.left_shift(jnp.uint32(1), jnp.maximum(source, 1)-1), jnp.uint32(0))
    immune = eligible & ((traits[slots, 2] & bit) != 0)
    warded = eligible & ~immune & ((traits[slots, 3] & ~used[slots] & bit) != 0)
    used = used.at[slots].set(used[slots] | jnp.where(warded, bit, jnp.uint32(0)))
    bits = jnp.left_shift(jnp.uint32(1), slots.astype(jnp.uint32))
    return (eligible & ~immune & ~warded, used,
            jnp.sum(jnp.where(immune, bits, jnp.uint32(0))),
            jnp.sum(jnp.where(warded, bits, jnp.uint32(0))))


def armor_after_shreds(base, count):
    # Reference casts armour to int when the first Shatter lands.
    return jnp.where(count > 0, jnp.maximum(0., jnp.trunc(base)-15*count), base)


def advance_poison_queue(hp, escaped, phases, priority, round_number, activated,
                         turns, damage, source, immunities, extra_priority,
                         next_actor, hold, max_rounds):
    """Resolve the pending queue and, if necessary, one following round in bulk.

    Candidate ticks are independent. Apply only the prefix through the first
    survivor or the death that ends combat. Ties follow argmax's slot order.
    This avoids a divergent device while-loop across parallel PPO environments.
    """
    slots = jnp.arange(12)

    def ended(health):
        active = (health > 0) & ~escaped
        return ~jnp.any(active[:6]) | ~jnp.any(active[6:])

    def sweep(health, phase, order, done, left, amounts, owners, actor, enabled):
        active = (health > 0) & ~escaped
        pending = active & (phase < 2)
        scores = jnp.where(phase == 0, order, -order)
        first = pending & ~done
        immune = (immunities & jnp.uint32(1 << 4)) != 0
        tick = first & (left > 0) & ~immune
        losses = jnp.where(tick, jnp.minimum(health, amounts), 0)
        potential = health-losses
        survivors = pending & (potential > 0)
        survivor = jnp.argmax(jnp.where(survivors, scores, -1e9))

        def last(mask):
            minimum = jnp.min(jnp.where(mask, scores, jnp.inf))
            return jnp.max(jnp.where(mask & (scores == minimum), slots, 0))

        # A side can die before a later survivor's turn. Stop at its last
        # casualty, so no opponent receives a tick after victory is decided.
        wiped_hero = ~jnp.any((potential[:6] > 0) & ~escaped[:6])
        wiped_enemy = ~jnp.any((potential[6:] > 0) & ~escaped[6:])
        last_hero = last(active & (slots < 6))
        last_enemy = last(active & (slots >= 6))
        candidates = jnp.array([survivor, last_hero, last_enemy, last(pending)])
        usable = jnp.array([jnp.any(survivors), wiped_hero, wiped_enemy, True])
        candidate_scores = jnp.where(usable, scores[candidates], -1e9)
        top = jnp.max(candidate_scores)
        boundary = jnp.min(jnp.where(usable & (candidate_scores == top), candidates, 12))
        visited = enabled & pending & ((scores > scores[boundary]) |
                                      ((scores == scores[boundary]) & (slots <= boundary)))
        losses = jnp.where(visited, losses, 0)
        health -= losses
        remaining = jnp.where(first & immune, 0, left-tick.astype(jnp.int32))
        left = jnp.where(visited, remaining, left)
        dead = visited & (health == 0)
        phase = jnp.where(dead, 2, phase)
        done |= dead
        actor = jnp.where(enabled, boundary, actor).astype(jnp.int32)
        live_dot = (left > 0) & (health > 0) & ~escaped
        left = jnp.where(live_dot, left, 0)
        amounts = jnp.where(live_dot, amounts, 0)
        owners = jnp.where(live_dot, owners, -1)
        wrap = enabled & ~ended(health) & ~jnp.any((health > 0) & ~escaped & (phase < 2))
        return health, phase, done, left, amounts, owners, actor, losses, wrap, visited

    enabled = ~hold & ~ended(hp) & (round_number <= max_rounds)
    hp, phases, activated, turns, damage, source, next_actor, losses, wrap, visited = sweep(
        hp, phases, priority, activated, turns, damage, source, next_actor, enabled)
    round_number += wrap.astype(jnp.int32)
    phases = jnp.where(wrap, 0, phases)
    priority = jnp.where(wrap, extra_priority, priority)
    activated = jnp.where(wrap, False, activated)
    hp, phases, activated, turns, damage, source, next_actor, more, _, more_visited = sweep(
        hp, phases, priority, activated, turns, damage, source, next_actor,
        wrap & (round_number <= max_rounds))
    return hp, phases, priority, round_number, activated, turns, damage, source, next_actor, losses+more, visited | more_visited



def advance_periodic_queue(hp, escaped, phases, priority, round_number, activated,
                         turns, damage, source, immunities, extra_priority,
                         next_actor, hold, max_rounds, immunity_masks, fallback_damage):
    """Resolve the pending queue and, if necessary, one following round in bulk.

    Per-effect ticks run in source order for each candidate. Apply only the prefix through the first
    survivor or the death that ends combat. Ties follow argmax's slot order.
    This avoids a divergent device while-loop across parallel PPO environments.
    """
    slots = jnp.arange(12)

    def ended(health):
        active = (health > 0) & ~escaped
        return ~jnp.any(active[:6]) | ~jnp.any(active[6:])

    def sweep(health, phase, order, done, left, amounts, owners, actor, enabled):
        active = (health > 0) & ~escaped
        pending = active & (phase < 2)
        scores = jnp.where(phase == 0, order, -order)
        first = (pending & ~done)[:, None]
        immune = (immunities[:, None] & immunity_masks[None, :]) != 0
        tick = first & (left > 0) & ~immune
        raw = jnp.where(tick, jnp.where(amounts == 0, fallback_damage[None, :], amounts), 0)
        before = jnp.maximum(health[:, None]-(jnp.cumsum(raw, axis=1)-raw), 0)
        tick &= before > 0
        losses = jnp.minimum(raw, before)
        potential = health-jnp.sum(losses, axis=1)
        survivors = pending & (potential > 0)
        survivor = jnp.argmax(jnp.where(survivors, scores, -1e9))

        def last(mask):
            minimum = jnp.min(jnp.where(mask, scores, jnp.inf))
            return jnp.max(jnp.where(mask & (scores == minimum), slots, 0))

        # A side can die before a later survivor's turn. Stop at its last
        # casualty, so no opponent receives a tick after victory is decided.
        wiped_hero = ~jnp.any((potential[:6] > 0) & ~escaped[:6])
        wiped_enemy = ~jnp.any((potential[6:] > 0) & ~escaped[6:])
        last_hero = last(active & (slots < 6))
        last_enemy = last(active & (slots >= 6))
        candidates = jnp.array([survivor, last_hero, last_enemy, last(pending)])
        usable = jnp.array([jnp.any(survivors), wiped_hero, wiped_enemy, True])
        candidate_scores = jnp.where(usable, scores[candidates], -1e9)
        top = jnp.max(candidate_scores)
        boundary = jnp.min(jnp.where(usable & (candidate_scores == top), candidates, 12))
        visited = enabled & pending & ((scores > scores[boundary]) |
                                      ((scores == scores[boundary]) & (slots <= boundary)))
        losses = jnp.where(visited[:, None], losses, 0)
        health -= jnp.sum(losses, axis=1)
        remaining = jnp.where(first & immune & (before > 0), 0, left-tick.astype(jnp.int32))
        left = jnp.where(visited[:, None], remaining, left)
        dead = visited & (health == 0)
        phase = jnp.where(dead, 2, phase)
        done |= dead
        actor = jnp.where(enabled, boundary, actor).astype(jnp.int32)
        live_dot = (left > 0) & ((health > 0) & ~escaped)[:, None]
        left = jnp.where(live_dot, left, 0)
        amounts = jnp.where(live_dot, amounts, 0)
        owners = jnp.where(live_dot, owners, -1)
        wrap = enabled & ~ended(health) & ~jnp.any((health > 0) & ~escaped & (phase < 2))
        return health, phase, done, left, amounts, owners, actor, losses, wrap, visited

    enabled = ~hold & ~ended(hp) & (round_number <= max_rounds)
    hp, phases, activated, turns, damage, source, next_actor, losses, wrap, visited = sweep(
        hp, phases, priority, activated, turns, damage, source, next_actor, enabled)
    round_number += wrap.astype(jnp.int32)
    phases = jnp.where(wrap, 0, phases)
    priority = jnp.where(wrap, extra_priority, priority)
    activated = jnp.where(wrap, False, activated)
    hp, phases, activated, turns, damage, source, next_actor, more, _, more_visited = sweep(
        hp, phases, priority, activated, turns, damage, source, next_actor,
        wrap & (round_number <= max_rounds))
    return hp, phases, priority, round_number, activated, turns, damage, source, next_actor, losses+more, visited | more_visited

def vampiric_heal(hp, maximum, escaped, actor, pool, share_leftover):
    """Reference leech and equal capped integer shares, without a device loop."""
    maximum = maximum.astype(jnp.int32)
    self_heal = jnp.minimum(pool,jnp.maximum(maximum[actor]-hp[actor],0))
    hp = hp.at[actor].add(self_heal)
    allies = ((jnp.arange(12) < 6) == (actor < 6)) & (jnp.arange(12) != actor) & (hp > 0) & ~escaped
    needs = jnp.where(allies,jnp.maximum(maximum-hp,0),0)
    missing = jnp.where(actor < 6,needs[:6],needs[6:])
    budget = jnp.where(share_leftover,pool-self_heal,0)
    ordered = jnp.sort(missing)
    prefix = jnp.cumsum(ordered)-ordered
    count = jnp.arange(6,0,-1)
    lower = jnp.concatenate((jnp.zeros(1,jnp.int32),ordered[:-1]))
    # Every suffix is a possible set of recipients still needing HP. Select
    # the largest feasible equal integer share; at most five residual HP go
    # to the first still-wounded slots, matching the reference's loop order.
    level = jnp.max(jnp.where(budget >= prefix+count*lower,(budget-prefix)//count,0))
    given = jnp.minimum(missing,level)
    remaining = budget-jnp.sum(given)
    eligible = missing > level
    given += (eligible & (jnp.cumsum(eligible) <= remaining)).astype(jnp.int32)
    zeros = jnp.zeros(6,jnp.int32)
    return hp+jnp.where(actor < 6,jnp.concatenate((given,zeros)),jnp.concatenate((zeros,given)))
