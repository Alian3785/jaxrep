"""Build the event-free Empire Capital challenge from the pinned Steam scenario.

Offline data conversion only: no JAX environment, policy or original game runs.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_bethrezens_retreat import UNIT_KEYS, footprint, formation, item_catalog, validate_geometry
from scripts.build_default_map import CURRENT, ROOT, RULES, UNITS, cell, dump

SNAPSHOT = ROOT / 'docs/maps/empire_alliance_war/source.json'
PLAYER, HERO, HOME, TARGET = 'S143PL0001', 'S143KC0000', 'S143FT0000', 'S143FT0001'


def build(snapshot):
    catalog = item_catalog()
    spell_catalog = json.loads((ROOT / 'stoix/envs/data/spells.json').read_text(encoding='utf-8'))['factions']
    spell_ids = {r['id'] for rows in spell_catalog.values() for r in rows}
    items = {r['ITEM_ID'].upper(): r for r in snapshot['globals']['gitem']}
    spells = {r['SPELL_ID'].upper(): r for r in snapshot['globals']['gspells']}

    def loot(inventory):
        result = dict(potions={}, items={})
        for gid, count in inventory.items():
            if gid == 'G000000000' or not count:
                continue
            category, key = catalog[gid]
            result[category][key] = result[category].get(key, 0) + count
        return result

    def resolved(army):
        return dict(army, units=[dict(u, big=UNITS[UNIT_KEYS[u['game_id']]]['size'] == 2) for u in army['units']])

    def overrides(army):
        # GModif/GModifL: this map has one modified blue dragon and one
        # stationary shaman with -2 leadership (no effect on its fixed roster).
        result = []
        by_slot = sorted(army['units'], key=lambda u: min(u['cells']) // 2 + 3 * (min(u['cells']) % 2))
        for u in by_slot:
            row, change = UNITS[UNIT_KEYS[u['game_id']]], {}
            if u['name']:
                change['name'] = u['name']
            if u['xp']:
                change['exp_current'] = u['xp']
            if row.get('hero') and army['source_id'] != HERO:
                # Native enemy leaders have only the modifiers recorded in SG;
                # the reference's automatic player skill ladder is not theirs.
                change['level_bonuses'] = []
            for mod in u['modifiers']:
                if mod == 'G000UM9001':
                    change['max_hp'] = row['max_hp'] * 120 // 100
                elif mod == 'G000UM9003':
                    change['armor'] = row['armor'] + 10
                elif mod == 'G000UM9009':
                    change['damage'] = round(row['damage'] * 1.1)
                elif mod == 'G000UM9035':
                    change['regeneration'] = row['regeneration'] + 10
                elif mod != 'G000UM9032':
                    raise ValueError('Unsupported scenario modifier: ' + mod)
            result.append(change)
        return result

    armies = [resolved(a) for a in snapshot['armies']]
    enemies = sorted((a for a in armies if a['owner'] != PLAYER), key=lambda a: a['source_id'] == a['building'])
    hero = next(a for a in armies if a['source_id'] == HERO)
    home = next(f for f in snapshot['forts'] if f['source_id'] == HOME)
    hero_roster, _ = formation(hero)
    rosters, levels = zip(*(formation(a) for a in enemies))
    blocked = {(x, y) for y, row in enumerate(snapshot['terrain']) for x, value in enumerate(row)
               if (value >> 3) & 7 == 4}
    for landmark in snapshot['landmarks']:
        blocked.update(footprint(landmark))
    for fort in snapshot['forts']:
        blocked.update(footprint(fort))
        blocked.discard(tuple(fort['marker_position']))
    managed = set().union(*(footprint(r) for r in snapshot['ruins'] + snapshot['sites']))
    blocked -= managed
    terrain = {name: [cell((x, y)) for y, row in enumerate(snapshot['terrain']) for x, value in enumerate(row)
                     if (value >> 3) & 7 == kind and (x, y) not in blocked | managed]
               for name, kind in (('forest', 1), ('water', 3))}
    terrain['road'] = [cell(p) for p in snapshot['roads'] if tuple(p) not in blocked | managed]
    roads = {tuple(p) for p in terrain['road']}
    for name in ('forest', 'water'):
        terrain[name] = [p for p in terrain[name] if tuple(p) not in roads]
    initial_inventory = Counter(home['inventory']) + Counter(hero['inventory'])
    target = next(i for i, a in enumerate(enemies) if a['source_id'] == TARGET)
    game_map = dict(
        **{k: CURRENT[k] for k in RULES},
        name='NumberGrid-146x146-empire_alliance_war', display_name='Empire-Alliance War — Capital',
        size=snapshot['size'] + 2, faction='empire', agent_number=1,
        agent_position=cell(home['marker_position']), capital_position=cell(home['marker_position']),
        opponent_positions=[cell(a['marker_position']) for a in enemies], opponent_numbers=[1] * len(enemies),
        hero_roster=hero_roster, hero_units=sum(k is not None for k in hero_roster), hero_hp=150,
        hero_combat_stats=overrides(hero), hero_leadership_capacity=3,
        enemy_rosters=list(rosters), enemy_initial_levels=list(levels), enemy_combat_stats=[overrides(a) for a in enemies],
        enemy_units=[sum(k is not None for k in roster) for roster in rosters],
        enemy_hp=[max(UNITS[k]['max_hp'] for k in roster if k) for roster in rosters],
        starting_gold=500, starting_mana=[0, 1500, 0, 0, 0], starting_built_buildings=[], starting_hero_abilities=[],
        initial_potions=loot(initial_inventory)['potions'], initial_items=loot(initial_inventory)['items'],
        battle_items=True, hero_skills=True, obstacles=sorted(cell(p) for p in blocked), terrain=terrain,
        allow_water_objects=True, campaign_objective='capital_guardian', objective_enemy=target,
        objective_reward_hero_only=True, army_victory_level_reward=5., army_victory_level_objective_only=True,
        scenario_events_enabled=False, scenario_events=[], scheduled_enemies=[],
        cities=[dict(name=f['name'], position=cell(f['marker_position']), level=f['level'], objective=False)
                for f in snapshot['forts'] if f['kind'] == 'city'],
        mines=[dict(position=cell(r['position']), kind=('gold', 'infernal', 'life', 'death', 'runes', 'elves')[r['resource']])
               for r in snapshot['resources']],
        chests=[dict(source_id=c['source_id'], position=cell(c['position']), **loot(c['inventory'])) for c in snapshot['chests']],
        ruins=[dict(source_id=r['source_id'], name=r['name'], position=cell(r['position']), entrance=cell(r['marker_position']),
                    gold=r['gold'], **loot({r['reward']: 1})) for r in snapshot['ruins']],
        merchants=[], spell_shops=[], trainers=[], mercenaries=[], enemy_capitals=[], enemy_loot=[],
        reference_enemy_ids=[a['source_id'] for a in enemies],
        reference_source='Steam Rise of the Elves exports/empire-alliance war.sg; docs/maps/empire_alliance_war/source.json',
        reference_sha256=snapshot['source']['sha256'],
        reconstruction_note='Original 144x144 terrain and initial placements; event-free Capital adaptation. See docs/EMPIRE_ALLIANCE_WAR.md.',
        unavailable_content=dict(spells=[], item_effects=['G000IG6004', 'G000IG6007', 'G000IG6008', 'G000IG6019']),
        own_capital_garrison=next(a for a in armies if a['source_id'] == HOME),
    )
    game_map.update(max_steps=45000, max_building_level=snapshot['limits']['MAX_UNIT'],
                    max_spell_level=snapshot['limits']['MAX_SPELL'], max_hero_level=snapshot['limits']['MAX_LEADER'],
                    battle_observation_version=max(45, CURRENT['battle_observation_version']))
    for i, army in enumerate(enemies):
        if not army['inventory']:
            continue
        # Visiting parties drop their inventory on defeat. A fort's shared store
        # is available only once BOTH its visitor and its garrison are defeated.
        guards = ([j for j, a in enumerate(enemies) if a['building'] == army['building']]
                  if army['source_id'] == army['building'] else [i])
        game_map['enemy_loot'].append(dict(source_id=army['source_id'], guards=guards, **loot(army['inventory'])))
    for site in snapshot['sites']:
        entry = dict(source_id=site['source_id'], name=site['name'], position=cell(site['position']))
        if site['kind'] == 'merchant':
            entry.update(loot(site['stock']))
            entry['prices'] = {catalog[k][1]: int(items[k]['VALUE'].split(':')[0][1:]) for k in site['stock']}
            game_map['merchants'].append(entry)
        elif site['kind'] == 'mage':
            entry['spells'], entry['unavailable_spells'] = [], []
            for gid in site['spells']:
                offer = dict(spell=gid.lower(), price=int(spells[gid]['BUY_C'].split(':')[0][1:]), stock=1)
                if gid.lower() in spell_ids:
                    entry['spells'].append(offer)
                else:
                    entry['unavailable_spells'].append(dict(offer, name=snapshot['globals']['texts'][spells[gid]['NAME_TXT'].lower()],
                                                            reason='Эффект ещё не реализован в JAX'))
                    game_map['unavailable_content']['spells'].append(dict(site=site['source_id'], **offer))
            game_map['spell_shops'].append(entry)
        elif site['kind'] == 'mercenary':
            entry['roster'] = [dict(unit=UNIT_KEYS[u['game_id']], stock=1, unlimited=not u['unique']) for u in site['units']]
            if any(u['level'] != UNITS[UNIT_KEYS[u['game_id']]]['level'] for u in site['units']):
                raise ValueError('Mercenary level differs from its base level')
            game_map['mercenaries'].append(entry)
        else:
            game_map['trainers'].append(entry)
    for fort in snapshot['forts']:
        if fort['kind'] == 'capital' and fort['owner'] != PLAYER:
            guards = [i for i, a in enumerate(enemies) if a['building'] == fort['source_id']]
            game_map['enemy_capitals'].append(dict(name=fort['name'], position=cell(fort['marker_position']),
                guard=target, guards=guards, footprint=sorted(cell(p) for p in footprint(fort))))
    game_map['city_defender_roles'] = [('capital' if a['kind'] == 'capital' else 'garrison')
                                       if a['kind'] in ('capital', 'city') and a['source_id'] == a['building'] else 'guard'
                                       for a in enemies]
    validate_geometry(game_map)
    return game_map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'maps/empire_alliance_war.json')
    args = parser.parse_args()
    game_map = build(json.loads(args.snapshot.read_text(encoding='utf-8')))
    args.output.write_text(dump(game_map), encoding='utf-8')
    print('Wrote', args.output, 'armies', len(game_map['enemy_rosters']), 'goal', game_map['objective_enemy'])
