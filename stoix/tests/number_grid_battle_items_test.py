"""GPU contracts: hero battle slots (orbs, talismans, battle potions) on the Default map.

Reference: Disciples2-RL campaign_env_inventory battle items and battle_env hero-item actions.
"""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stoix.envs.number_grid import DEFEND, REST, WAIT, action_names, load_map
from stoix.envs.number_grid_battle_items import (BATTLE_ITEM, ITEM_EQUIPPED, MAGIC_BY_KEY, MAGIC_ITEMS,
                                                 ORBS, TALISMANS, BattleItemRules)
from stoix.envs.number_grid_items import ITEMS, USABLE, ItemRules
from stoix.tests.number_grid_fixtures import enemy_roster_state



DEFAULT = load_map('default')
NAMES = action_names(DEFAULT)
NE = 1  # [28, 7] -> enemy 11 at [27, 8]


def owned(env, state, *keys):
    ids = [env.item_rules.keys.index(key) for key in keys]
    return state.replace(item_inventory=jnp.array(ids+[-1]*(env.item_rules.capacity-len(ids)), jnp.int32))


def slots(env, state, *keys):
    rules = env.battle_item_rules
    kinds = [rules.keys.index(key) if key else -1 for key in keys]
    return state.replace(battle_items=jnp.array(kinds, jnp.int32),
                         battle_item_charges=jnp.array([int(rules.uses[k]) if k >= 0 else 0 for k in kinds], jnp.int32),
                         battle_item_used=jnp.zeros(2, bool))


def allowed(mask, start, end):
    return [NAMES[i] for i in range(start, end) if mask[i]]


@pytest.mark.core
def test_catalog_and_default_actions(default_game):
    env, state, _, _ = default_game
    rules = env.battle_item_rules
    assert (len(ORBS), len(TALISMANS)) == (44, 41)
    assert [p['game_id'] for p in ORBS[:43]] == [f'G000IG90{i:02d}' for i in range(1, 44)]
    assert [p['game_id'] for p in TALISMANS] == [f'G000IG91{i:02d}' for i in range(1, 42)]
    assert ORBS[43]['key'] == 'orb_of_lycanthropy' and ORBS[43]['game_id'] is None  # reference-only
    assert [MAGIC_BY_KEY[k]['price'] for k in ('vampire_orb', 'zombie_talisman', 'orb_of_flame',
                                               'talisman_of_nightmare')] == [800, 400, 1000, 600]
    assert {p['key'] for p in ITEMS if p['category'] == USABLE} == set(MAGIC_BY_KEY)
    # Reference order: battle potions, then the scenario's orbs and talismans.
    assert rules.keys == ('healing', 'restoration', 'ointment', 'life',
                          'vampire_orb', 'lich_orb', 'orb_of_life', 'zombie_talisman')
    equip, use = NAMES[rules.equip_start:rules.use_start], NAMES[rules.use_start:rules.end]
    assert equip == tuple('equip_'+k for k in rules.keys) and len(use) == 24
    assert use[:7] == tuple(f'item1_ally_{t}' for t in range(6))+('item2_ally_0',)
    assert use[12] == 'item1_enemy_0' and use[-1] == 'item2_enemy_5'
    assert rules.observation_size == 20 and state.battle_items.tolist() == [-1, -1]


@pytest.mark.core
def test_equip_needs_access_and_happens_once_per_day(default_game):
    env, initial, step, mask = default_game
    rules = env.battle_item_rules
    equip = {k: rules.equip_start+i for i, k in enumerate(rules.keys)}
    healing = env.potion_rules.keys.index('healing')
    tome = env.item_rules.keys.index('tome_arcanum')
    stocked = owned(env, initial, 'vampire_orb', 'zombie_talisman', 'tome_arcanum')
    stocked = stocked.replace(potions=initial.potions.at[healing].set(1))
    # Potions need no lore; Tome of Arcanum allows orbs, level 10 (sorcery lore) everything.
    assert allowed(mask(stocked), rules.equip_start, rules.use_start) == ['equip_healing']
    book = stocked.replace(equipped=stocked.equipped.at[3].set(tome))
    assert allowed(mask(book), rules.equip_start, rules.use_start) == ['equip_healing', 'equip_vampire_orb']
    lore = stocked.replace(unit_levels=stocked.unit_levels.at[1].set(10))
    assert allowed(mask(lore), rules.equip_start, rules.use_start) == [
        'equip_healing', 'equip_vampire_orb', 'equip_zombie_talisman']
    # Default needs book lore (level 11) to read a book: rest unequips it and the orb's access.
    unread, _ = step(step(book, jnp.int32(equip['vampire_orb']))[0], jnp.int32(REST))
    assert unread.equipped[3] == -1 and unread.battle_items.tolist() == [-1, -1]
    first, _ = step(lore, jnp.int32(equip['vampire_orb']))
    assert first.battle_items.tolist() == [rules.keys.index('vampire_orb'), -1]
    assert first.last_event == ITEM_EQUIPPED and first.battle_item_equipped_today
    assert not jnp.any(mask(first)[rules.equip_start:rules.use_start])  # once per day
    rested, _ = step(first, jnp.int32(REST))
    assert allowed(mask(rested), rules.equip_start, rules.use_start) == [
        'equip_healing', 'equip_zombie_talisman']  # the one orb is in a slot
    full, _ = step(rested, jnp.int32(equip['healing']))
    assert full.battle_items.tolist() == [rules.keys.index('vampire_orb'), rules.keys.index('healing')]
    assert full.battle_item_charges.tolist() == [1, 1]
    rested, _ = step(full, jnp.int32(REST))
    assert not jnp.any(mask(rested)[rules.equip_start:rules.use_start])  # both slots taken
    # Reference sync: losing the orb or the last potion empties its slot.
    lost, _ = step(owned(env, rested, 'zombie_talisman'), jnp.int32(REST))
    assert lost.battle_items.tolist() == [-1, rules.keys.index('healing')]
    drunk, _ = step(rested.replace(potions=rested.potions.at[healing].set(0)), jnp.int32(REST))
    assert drunk.battle_items.tolist() == [rules.keys.index('vampire_orb'), -1]


@pytest.fixture(scope='module')
def item_battle(default_game):
    env, initial, step, _ = default_game
    battle, _ = step(initial.replace(position=jnp.array([28, 7], jnp.int32)), jnp.int32(NE))
    assert battle.in_battle and battle.enemy == 11
    battle = owned(env, battle, 'orb_of_life', 'zombie_talisman')
    return slots(env, battle, 'orb_of_life', 'zombie_talisman').replace(
        actor=jnp.int32(1), hp=battle.hp.at[0].set(0))  # the hero acts; Squire 0 has fallen


@pytest.mark.core
def test_hero_revives_and_summons_on_its_turn(default_game, item_battle):
    env, _, step, mask = default_game
    rules, ids = env.battle_item_rules, env.progression.ids
    battle = item_battle
    assert allowed(mask(battle), rules.use_start, rules.end) == [
        'item1_ally_0', 'item2_ally_0', 'item2_ally_3', 'item2_ally_5']
    assert not jnp.any(mask(battle.replace(actor=jnp.int32(0)))[rules.use_start:rules.end])  # hero only
    # Orb of Life: 50% HP, the orb leaves the inventory, the turn is spent.
    revived, _ = step(battle, jnp.int32(NAMES.index('item1_ally_0')))
    assert revived.hp[0] == 50 and revived.last_event == BATTLE_ITEM and revived.actor != 1
    assert revived.battle_items.tolist() == [-1, rules.keys.index('zombie_talisman')]
    assert env.item_rules.keys.index('orb_of_life') not in revived.item_inventory.tolist()
    # Zombie Talisman: an independent summon; one of five charges, once per battle.
    summoned, _ = step(battle, jnp.int32(NAMES.index('item2_ally_3')))
    assert summoned.unit_ids[3] == ids['g000uu0087'] and summoned.summon_owner[3] == 3 and summoned.hp[3] == 170
    assert summoned.battle_item_charges.tolist() == [1, 4] and summoned.battle_item_used.tolist() == [False, True]
    again = summoned.replace(actor=jnp.int32(1), turn_phase=summoned.turn_phase.at[1].set(0))
    assert allowed(mask(again), rules.use_start, rules.end) == ['item1_ally_0']
    # The mask and step share one predicate: a masked item action changes nothing.
    rejected, _ = step(battle, jnp.int32(NAMES.index('item1_enemy_0')))
    assert rejected.actor == 1 and rejected.battle_items.tolist() == battle.battle_items.tolist()
    # Victory: the summon leaves, the fallen Squire stays dead, the talisman keeps 4 charges.
    won, _ = step(again.replace(hp=again.hp.at[6:].set(0)), jnp.int32(DEFEND))
    for _ in range(6):
        if won.in_battle:
            won, _ = step(won, jnp.int32(WAIT))  # post-battle healing turns
    assert not won.in_battle
    assert won.unit_ids[3] == 0 and won.hp[0] == 0 and not jnp.any(won.summon_owner >= 0)
    assert won.battle_items.tolist() == battle.battle_items.tolist() and won.battle_item_charges.tolist() == [1, 4]


@pytest.mark.core
def test_battle_potion_heals_and_is_drunk(default_game, item_battle):
    env, _, step, mask = default_game
    healing = env.potion_rules.keys.index('healing')
    battle = slots(env, item_battle, 'healing', None)
    battle = battle.replace(potions=battle.potions.at[healing].set(2), hp=battle.hp.at[0].set(40))
    assert allowed(mask(battle), env.battle_item_rules.use_start, env.battle_item_rules.end) == [
        f'item1_ally_{t}' for t in (0, 1, 2, 4)]
    healed, _ = step(battle, jnp.int32(NAMES.index('item1_ally_0')))
    assert healed.hp[0] == 90 and healed.potions[healing] == 1 and healed.battle_items.tolist() == [-1, -1]


ENEMIES = ['goblin', 'orc', 'goblin', 'g000uu0090', 'g000uu0087', 'goblin_archer']  # Templar wards, Zombie
CASES = (  # key, enemy side, target, (ally slot, hp) before use
    ('orb_of_fire', 1, 0, None), ('talisman_of_thunder', 1, 1, None), ('orb_of_nosferat', 1, 0, (1, 50)),
    ('orb_of_elder_vampires', 1, 0, (0, 10)), ('orb_of_poison', 1, 1, None), ('orb_of_frost', 1, 0, None),
    ('orb_of_flame', 1, 0, None), ('orb_of_vigor', 0, 0, None), ('orb_of_rage', 0, 0, None),
    ('orb_of_healing', 0, 0, (0, 10)), ('orb_of_restoration', 0, 0, (0, 10)), ('orb_of_fear', 1, 1, None),
    ('talisman_of_nightmare', 1, 1, None), ('orb_of_witches', 1, 2, None), ('orb_of_lycanthropy', 1, 1, None),
    ('orb_of_weakening', 1, 1, None), ('elf_lord_orb', 0, 3, None), ('orb_of_life', 0, 0, (0, 0)))


@pytest.fixture(scope='module')
def effects(default_game):
    """Every reference effect through one vmapped kernel on a Default copy."""
    original, initial, _, _ = default_game
    env = copy.copy(original)
    game_map = dict(DEFAULT, initial_items={p['key']: 1 for p in MAGIC_ITEMS})
    env.item_rules = ItemRules(game_map)
    rules = env.battle_item_rules = BattleItemRules(game_map, env.potion_rules, env.item_rules,
                                                    env.progression, env.construction)
    for flag in ('has_fire', 'has_water', 'has_powerups', 'has_paralysis', 'has_witches', 'has_wights',
                 'has_lycanthropy', 'has_weakening', 'has_item_weakening', 'has_item_summons', 'has_cures'):
        setattr(env, flag, True)
    begin = jax.jit(env._begin_battle)(initial.replace(enemy=jnp.int32(11)))
    base = enemy_roster_state(env, begin, ENEMIES).replace(
        actor=jnp.int32(1), item_inventory=jnp.full(env.item_rules.capacity, -1, jnp.int32))
    count = len(CASES)
    kinds = np.array([[rules.keys.index(key), -1] for key, *_ in CASES], np.int32)
    hp = np.repeat(np.asarray(base.hp)[None], count, 0)
    for row, (*_, wound) in enumerate(CASES):
        if wound:
            hp[row, wound[0]] = wound[1]
    batch = jax.tree.map(lambda x: jnp.repeat(x[None], count, 0), base).replace(
        battle_items=jnp.asarray(kinds), hp=jnp.asarray(hp),
        battle_item_charges=jnp.asarray(np.where(kinds >= 0, np.asarray(rules.uses)[kinds], 0), jnp.int32))
    actions = jnp.array([rules.use_start+side*12+target for _, side, target, _ in CASES], jnp.int32)

    @jax.jit
    @jax.vmap
    def apply(state, action):
        valid = rules.use_available(env, state)[action-rules.use_start]
        after, hp, _, info = rules.use(env, state, state.hp, state.wards_used, action,
                                       jnp.zeros(env.random_size), env._combat_stats(state),
                                       env._combat_traits(state))
        return valid, after, hp, info

    valid, after, hp, info = apply(batch, actions)
    keys = [key for key, *_ in CASES]
    return env, base, keys, valid, after, hp, info


def test_reference_item_effects(effects):
    env, base, keys, valid, after, hp, info = effects
    row = {key: i for i, key in enumerate(keys)}
    rules, ids = env.battle_item_rules, env.progression.ids
    assert bool(jnp.all(valid))
    start = base.hp.tolist()

    def lost(key):
        return [a-b for a, b in zip(start, hp[row[key]].tolist())]

    # Party fire 25 at 80% accuracy; the Templar's fire ward absorbs it and is spent.
    templar = 1 << 9
    assert lost('orb_of_fire')[6:] == [25, 25, 25, 0, 25, 25] and int(info['ward'][row['orb_of_fire']]) == templar
    assert lost('talisman_of_thunder')[6:] == [0, 100, 0, 0, 0, 0]
    # Drain: the Zombie is immune to death; the hero heals half of the damage dealt.
    nosferat = lost('orb_of_nosferat')
    assert nosferat[6:] == [25, 25, 25, 25, 0, 25] and int(info['immune'][row['orb_of_nosferat']]) == 1 << 10
    assert hp[row['orb_of_nosferat'], 1] == 50+125//2
    assert hp[row['orb_of_elder_vampires'], 0] == 100  # the hero is unhurt: the party shares the pool
    # DOTs: single and party, duration 1..6 from the shared roll, the Templar ward blocks water/fire.
    for key, field, amount, targets in (('orb_of_poison', 'poison', 20, [7]), ('orb_of_frost', 'water', 10, [6, 7, 8, 10, 11]),
                                        ('orb_of_flame', 'burn', 15, [6, 7, 8, 10, 11])):
        damage = getattr(after, field+'_damage')[row[key]]
        assert [int(i) for i in jnp.flatnonzero(damage == amount)] == targets, key
        assert jnp.all(getattr(after, field+'_turns')[row[key], jnp.array(targets)] == 1)
    original = env._original_primary(base)
    assert after.primary_override[row['orb_of_vigor'], 0] == jnp.rint(original[0]*1.25)
    rage = row['orb_of_rage']
    assert after.turn_phase[rage, 0] == 0 and after.bonus_turns[rage, 0] == 1 and info['extra'][rage, 0]
    assert hp[row['orb_of_healing'], 0] == 100 and hp[row['orb_of_restoration'], 0] == 40
    assert after.paralyzed[row['orb_of_fear'], 7] and after.retreating[row['talisman_of_nightmare'], 7]
    assert after.imp[row['orb_of_witches'], 8]
    lycan = row['orb_of_lycanthropy']
    assert after.decay_form[lycan, 7] == ids['g000uu0092'] and after.lycanthrope[lycan, 7]
    weak = row['orb_of_weakening']
    assert after.item_weakened[weak, 7] and after.weakened[weak, 7]
    assert after.primary_override[weak, 7] == jnp.rint(env._combat_stats(base)[7, 1]*.67)
    summon = row['elf_lord_orb']
    assert after.unit_ids[summon, 3] == ids['g000uu8009'] and after.summon_owner[summon, 3] == 3
    life = row['orb_of_life']
    assert hp[life, 0] == 50 and after.battle_items[life].tolist() == [-1, -1]
    # Talismans spend one charge; orbs leave the slot.
    assert after.battle_item_charges[row['talisman_of_thunder']].tolist() == [4, 0]
    assert after.battle_items[row['orb_of_fire']].tolist() == [-1, -1]
    assert int(rules.uses[rules.keys.index('talisman_of_thunder')]) == 5
