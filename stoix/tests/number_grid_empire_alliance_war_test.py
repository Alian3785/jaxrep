"""Extended GPU checks for the native S143 Capital port. Not part of core.

Source: docs/maps/empire_alliance_war/source.json and EMPIRE_ALLIANCE_WAR.md.
Run only on explicit request through scripts/run_gpu.sh.
"""
from collections import Counter
import json
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, action_names, load_map
from stoix.envs.number_grid_bitsets import PackedBits
from stoix.tests.number_grid_fixtures import compiled_method

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def empire_game():
    assert all(d.platform == 'gpu' for d in jax.devices()), 'GPU is required'
    env = NumberGrid(map_config=load_map('empire_alliance_war'))
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    return env, state


def test_native_source_and_all_inventory_copies_survive_conversion():
    from scripts.build_bethrezens_retreat import item_catalog
    source = json.loads((ROOT / 'docs/maps/empire_alliance_war/source.json').read_text(encoding='utf-8'))
    game_map = load_map('empire_alliance_war')
    assert source['source']['sha256'] == 'aa771abf39f8a7919903abb14f4034b66eb5c60ff05e6c62a49572ce7d30b497'
    assert game_map['size'] == 146 and source['size'] == 144
    assert game_map['faction'] == 'empire' and not game_map['scenario_events_enabled']
    assert len(source['events']) == 117 and not game_map['scenario_events']
    assert [len(game_map[k]) for k in ('enemy_rosters', 'cities', 'ruins', 'chests', 'merchants', 'spell_shops', 'trainers', 'mercenaries')] == [258, 24, 58, 28, 6, 7, 8, 3]
    assert {a['source_id'] for a in source['armies'] if a['owner'] != 'S143PL0001'} == set(game_map['reference_enemy_ids'])
    assert game_map['reference_enemy_ids'][game_map['objective_enemy']] == 'S143FT0001'
    guardian = game_map['enemy_rosters'][game_map['objective_enemy']]
    assert 'g000uu8040' in guardian and sum(k is not None for k in guardian) == 6
    catalog = item_catalog()
    expected = Counter()
    for contents in ([a['inventory'] for a in source['armies']] + [c['inventory'] for c in source['chests']]
                     + [s['stock'] for s in source['sites'] if s['kind'] == 'merchant']
                     + [{r['reward']: 1} for r in source['ruins']]):
        expected.update({catalog[k][1]: n for k, n in contents.items() if k != 'G000000000'})
    actual = Counter(game_map['initial_potions']) + Counter(game_map['initial_items'])
    for row in game_map['chests'] + game_map['ruins'] + game_map['merchants'] + game_map['enemy_loot']:
        actual.update(row.get('potions', {}))
        actual.update(row.get('items', {}))
    assert actual == expected
    assert [actual[k] for k in ('invulnerability', 'bark', 'protection', 'iron_skin', 'highfather')] == [5, 3, 1, 3, 3]
    names = action_names(game_map)
    assert len(names) == len(set(names))


def test_high_spell_bits_do_not_alias_lower_spells():
    bits = PackedBits(101)
    @jax.jit
    def select():
        selected = bits.bits[3] | bits.bits[35] | bits.bits[70] | bits.bits[100]
        return bits.contains(selected, bits.bits)
    active = select().tolist()
    assert [i for i, value in enumerate(active) if value] == [3, 35, 70, 100]


def test_reset_native_hero_mana_modifiers_and_capital_queue(empire_game):
    env, state = empire_game
    assert state.mana.tolist() == [0, 1500, 0, 0, 0] and state.gold == 500
    assert state.position.tolist() == [37, 18]
    assert [r['name'] for r in env.progression.rows if r.get('name') == 'Renard'] == ['Renard']
    assert sum(i != 0 for i in state.unit_ids[:6].tolist()) == 1
    assert state.learned_spells.shape == (2,) and env.casting.book_count == 45
    assert not env.scenario.enabled and not env.script_events.enabled and not env.bot.enabled
    capital = env.map_config['enemy_capitals'][0]
    visitor = next(i for i in capital['guards'] if i != capital['guard'])
    at = jax.jit(env.territory.enemy_at)
    position = jnp.array(capital['position'], jnp.int32)
    assert at(state, position) == visitor
    assert at(state.replace(alive=state.alive.at[visitor].set(False)), position) == capital['guard']
    dragon = next(row for row in env.progression.rows if row.get('game_id') == 'g000uu5021' and row['max_hp'] == 840)
    assert dragon['armor'] == 10 and dragon['regeneration'] == 10 + int(dragon['game_data']['REGEN'])
    assert jnp.all(jnp.isfinite(compiled_method(env, 'observation')(state)))


def test_enemy_loot_once_and_fort_requires_both_defenders(empire_game):
    env, state = empire_game
    grant = jax.jit(env.enemy_loot.grant)
    entry = next(r for r in env.map_config['enemy_loot'] if r.get('potions', {}).get('invulnerability'))
    index = entry['guards'][0]
    potion = env.potion_rules.keys.index('invulnerability')
    after = grant(state, state.replace(alive=state.alive.at[index].set(False)))
    assert after.potions[potion] == state.potions[potion] + entry['potions']['invulnerability']
    twice = grant(after, after)
    chex.assert_trees_all_equal(after.potions, twice.potions)
    chex.assert_trees_all_equal(after.item_inventory, twice.item_inventory)
    capital = env.map_config['enemy_capitals'][0]
    guard, visitor = capital['guard'], next(i for i in capital['guards'] if i != capital['guard'])
    # Defeating the garrison out of order cannot release the shared fort store.
    partial = grant(state, state.replace(alive=state.alive.at[guard].set(False)))
    chex.assert_trees_all_equal(partial.potions, state.potions)
    complete = grant(partial, partial.replace(alive=partial.alive.at[visitor].set(False)))
    assert complete.potions[env.potion_rules.keys.index('life')] == state.potions[env.potion_rules.keys.index('life')] + 3


def test_foreign_spell_purchase_above_bit_31_and_duplicate_offer(empire_game):
    env, state = empire_game
    book = [r['id'] for r in env.casting.rows[:env.casting.book_count]]
    shop = env.site_rules
    offer_index = next(i for i, o in enumerate(shop.spell_offers) if book.index(o['spell']) >= 32)
    offer = shop.spell_offers[offer_index]
    spell_index = book.index(offer['spell'])
    position = jnp.array(shop.sites[offer['site']]['interaction_tiles'][0], jnp.int32)
    before = state.replace(position=position, gold=jnp.int32(10000), mana=jnp.full(5, 10000, jnp.int32))
    action = jnp.int32(shop.spell_start + offer_index)
    assert compiled_method(env, 'action_mask')(before)[action]
    after = jax.jit(shop.apply)(before, action)
    learned = env.casting.book_bitset.contains(after.learned_spells, env.casting.book_bits)
    assert learned[spell_index] and not learned[spell_index % 32]
    assert jax.jit(env.casting._book)(after)[spell_index]
    assert not compiled_method(env, 'action_mask')(after)[action]
    assert after.gold == before.gold - offer['price']


def test_large_unlimited_mercenary_uses_two_cells_and_leadership(empire_game):
    env, state = empire_game
    rules = env.recruitment
    offer = next(i for i, row in enumerate(rules.metadata['offers']) if row['mercenary'] and row['size'] == 2)
    camp = rules.metadata['offers'][offer]['camp']
    position = jnp.array(env.site_rules.sites[camp]['interaction_tiles'][0], jnp.int32)
    before = state.replace(position=position, gold=jnp.int32(10000))
    action = jnp.int32(rules.start + offer)
    available = jax.jit(rules.available)
    assert available(before)[offer]
    after, _ = jax.jit(lambda s, a: rules.apply(s, a, env._clear_roster_slots))(before, action)
    slot = int(after.last_target)
    assert slot < 3 and after.unit_ids[slot] == rules.ids[offer] and after.unit_ids[slot + 3] == 0
    assert rules.composition(after)[1] == 2 and after.mercenary_stock[offer - 5] == 1
    assert not available(after)[offer]  # only one leadership point remains
    apply = jax.jit(lambda s, a: rules.apply(s, a, env._clear_roster_slots))
    dismissed, _ = apply(after, jnp.int32(rules.dismiss_start + slot))
    assert available(dismissed)[offer]
    rehired, reward = apply(dismissed, action)
    assert rehired.mercenary_stock[offer - 5] == 1 and rehired.gold == after.gold - rules.costs[offer]
    assert reward == 0  # repeatedly hiring/dismissing cannot farm the vacancy reward


def test_real_step_grants_inventory_after_magic_kill(empire_game):
    env, state = empire_game
    entry = next(r for r in env.map_config['enemy_loot'] if r.get('potions', {}).get('invulnerability'))
    target = entry['guards'][0]
    spell = next(i for i, r in enumerate(env.casting.rows) if r['id'] == 'g000ss0004')
    maximum = env.casting.enemy_max_hp(state).astype(jnp.int32)
    before = state.replace(position=env.opponent_positions[target],
        alive=jnp.zeros_like(state.alive).at[target].set(True).at[env.map_config['objective_enemy']].set(True),
        learned_spells=env.casting.book_bits[spell], mana=jnp.full(5, 10000, jnp.int32),
        enemy_wounds=state.enemy_wounds.at[target].set(jnp.maximum(0, maximum[target] - 1)))
    action = jnp.int32(env.casting.start + spell)
    assert compiled_method(env, 'action_mask')(before)[action]
    after, _ = compiled_method(env, 'step')(before, action)
    assert not after.alive[target] and not after.won
    potion = env.potion_rules.keys.index('invulnerability')
    assert after.potions[potion] == before.potions[potion] + entry['potions']['invulnerability']
