"""GPU event counters for the reference CampaignMetricsCallback (no reward shaping)."""
import jax
import jax.numpy as jnp

from stoix.envs.number_grid_capital import HEAL_START, REVIVE_START
from stoix.envs.number_grid_casting import SUMMON

# Fixed layout keeps telemetry a single small array in the rollout/state tree.
CAMPAIGN_FIELDS = (
    'victories', 'defeats', 'timeouts',
    'battle_victories', 'battle_defeats', 'battle_retreats',
    'unit_upgrades', 'castle_heal_uses', 'castle_healed_hp',
    'chests_collected', 'buildings_built', 'ruins_cleared',
    'sold_items', 'merchant_sale_gold', 'spells_learned', 'spell_casts',
    'summoned_units', 'hired_units', 'battle_items_equipped', 'battle_items_used',
    'unit_swaps', 'enemy_squads_defeated', 'cities_captured',
)
EPISODE_WINDOW = 100


def empty_campaign_counts():
    return jnp.zeros(len(CAMPAIGN_FIELDS), jnp.float32)


def campaign_events(env, before, world, after, action, valid, battle_counts):
    """Measure executed events, retaining the map result before automatic combat.

    Map summons count as one summon, even when their entire battle finishes
    inside the same decision. Native battle summons are not map spell summons.
    """
    world_action = valid & ~before.in_battle
    terminal = ~before.done & after.done
    values = dict(zip(CAMPAIGN_FIELDS, [jnp.float32(0)] * len(CAMPAIGN_FIELDS)))
    values.update(
        victories=terminal & after.won,
        defeats=terminal & after.lost & ~after.won,
        timeouts=terminal & ~after.won & ~after.lost,
        battle_victories=battle_counts[0], battle_defeats=battle_counts[1],
        battle_retreats=battle_counts[2], unit_upgrades=battle_counts[3],
        enemy_squads_defeated=jnp.sum(before.alive & ~after.alive),
    )
    if env.basic_combat:
        values['buildings_built'] = jax.lax.population_count(world.buildings & ~before.buildings)
    if env.capital_enabled:
        healing = world_action & (action >= HEAL_START) & (action < REVIVE_START)
        healed = jnp.sum(jnp.maximum(world.hp[:6] - before.hp[:6], 0))
        values['castle_heal_uses'] = healing & (healed > 0)
        values['castle_healed_hp'] = jnp.where(healing, healed, 0)
    if env.chests_enabled:
        values['chests_collected'] = jnp.sum(before.chest_alive & ~world.chest_alive)
    if env.ruins_enabled:
        values['ruins_cleared'] = jnp.sum(~before.ruin_looted & after.ruin_looted)
    if env.sites_enabled:
        values['sold_items'] = jnp.where(world_action, world.last_sold_count, 0)
        values['merchant_sale_gold'] = jnp.where(world_action, world.last_sale_gold, 0)
    if env.spell_research_enabled:
        values['spells_learned'] = jax.lax.population_count(world.learned_spells & ~before.learned_spells)
    if env.spell_casting_enabled:
        cast = world_action & (world.last_cast_spell >= 0)
        values['spell_casts'] = cast
        values['summoned_units'] = cast & (env.casting.kinds[jnp.maximum(world.last_cast_spell, 0)] == SUMMON)
    if env.recruitment_enabled:
        values['hired_units'] = world_action & (action >= env.recruitment.start) & (action < env.recruitment.dismiss_start)
    if env.battle_items_enabled and env.battle_item_rules.count:
        rules = env.battle_item_rules
        values['battle_items_equipped'] = world_action & (action >= rules.equip_start) & (action < rules.use_start)
        values['battle_items_used'] = valid & before.in_battle & (action >= rules.use_start) & (action < rules.end)
    if env.unit_swaps_enabled:
        values['unit_swaps'] = world_action & (action >= env.unit_swaps.start) & (action < env.unit_swaps.end)
    if env.territory_enabled:
        values['cities_captured'] = jnp.sum(~before.city_owned & after.city_owned)
    return jnp.stack([jnp.asarray(values[name], jnp.float32) for name in CAMPAIGN_FIELDS])


def summarize_campaign(episodes):
    """Reduce on GPU; transfer at most 100 completed records per learner chunk.

    Totals and participation counts include *every* event/episode. Only the
    rolling-window records are bounded. Single-device rollout order is update,
    time, environment; simultaneous completions use the environment index.
    """
    ended = episodes['is_terminal_step']
    events = episodes['campaign_events']
    counts = episodes['campaign_episode']
    mask = ended[..., None]
    axes = tuple(range(ended.ndim))
    terminal = ended.reshape(-1)
    k = min(EPISODE_WINDOW, terminal.size)
    ranks = jnp.where(terminal, jnp.arange(terminal.size), -1)
    _, indices = jax.lax.top_k(ranks, k)
    records = jnp.concatenate((counts, episodes['episode_return'][..., None],
                               episodes['episode_length'][..., None]), axis=-1)
    recent = records.reshape(-1, len(CAMPAIGN_FIELDS) + 2)[indices[::-1]]
    return {
        'events': jnp.sum(events, axis=axes),
        'completed': jnp.sum(jnp.where(mask, counts, 0), axis=axes),
        'participation': jnp.sum(mask & (counts > 0), axis=axes),
        'count': jnp.sum(ended),
        'recent': recent,
    }
