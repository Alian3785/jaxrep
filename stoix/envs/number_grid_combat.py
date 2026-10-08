"""Static combat profiles and pure JAX hit checks; see docs/ATTACK_PROTECTIONS.md."""
import json
import math
from pathlib import Path

import jax.numpy as jnp

HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE = range(5)
STAT_NAMES = ('max_hp', 'damage', 'accuracy', 'armor', 'initiative')
ATTACK_TYPES = ('weapon', 'earth', 'fire', 'water', 'poison', 'death', 'mind', 'life', 'air')
ATTACK_LABELS = ('Оружие', 'Земля', 'Огонь', 'Вода', 'Яд', 'Смерть', 'Разум', 'Жизнь', 'Воздух')
EMPTY, MELEE, RANGED, AREA = range(4)
ROLES = {'melee': MELEE, 'ranged': RANGED, 'area': AREA}
UNITS = json.loads((Path(__file__).parent / 'data/units.json').read_text(encoding='utf-8'))
PROFILE_FIELDS = set(STAT_NAMES) | {'role', 'attack_type', 'immunities', 'protections', 'exp_kill', 'exp_required', 'exp_current'}


def _validated_stats(values):
    stats = {name: values[name] for name in STAT_NAMES}
    for name, value in stats.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'Invalid combat {name}')
    if (type(stats['max_hp']) is not int or not 1 <= stats['max_hp'] <= 1_000_000
            or type(stats['damage']) is not int or not 0 <= stats['damage'] <= 1_000_000
            or not 0 <= stats['accuracy'] <= 100 or not 0 <= stats['armor'] <= 100
            or type(stats['initiative']) is not int or not 0 <= stats['initiative'] <= 1000):
        raise ValueError('Combat stats outside supported ranges')
    stats['damage'] = min(stats['damage'], 400 if values.get('hero') else 300)
    stats['armor'] = min(stats['armor'], 90)
    return [stats[name] for name in STAT_NAMES]


def _source(value):
    if not isinstance(value, str) or value.lower() not in ATTACK_TYPES:
        raise ValueError('Unknown attack type')
    return ATTACK_TYPES.index(value.lower())


def _protection_mask(values):
    if not isinstance(values, list):
        raise ValueError('Immunities/protections must be lists of attack types')
    mask = 0
    for value in values:
        mask |= 1 << _source(value)
    return mask


def build_combat_tables(game_map):
    """Resolve named six-slot formations or explicit test profiles once before jit."""
    hero_count = game_map.get('hero_units', 6)
    counts = game_map['enemy_units']
    warriors = game_map.get('enemy_warrior_slots', [-1] * len(counts))
    hero_overrides = game_map.get('hero_combat_stats', [{} for _ in range(hero_count)])
    enemy_overrides = game_map.get('enemy_combat_stats', [[{} for _ in range(n)] for n in counts])
    if not isinstance(hero_overrides, list) or len(hero_overrides) != hero_count:
        raise ValueError('hero_combat_stats must contain one entry per hero unit')
    if (not isinstance(enemy_overrides, list) or len(enemy_overrides) != len(counts)
            or any(not isinstance(s, list) or len(s) != n for s, n in zip(enemy_overrides, counts))):
        raise ValueError('enemy_combat_stats must match the squad sizes')
    hero_roster = game_map.get('hero_roster')
    enemy_rosters = game_map.get('enemy_rosters')
    if enemy_rosters is not None and (not isinstance(enemy_rosters, list) or len(enemy_rosters) != len(counts)):
        raise ValueError('enemy_rosters must match all squads')

    def formation(count, overrides, roster, enemy=None):
        if roster is not None:
            if (not isinstance(roster, list) or len(roster) != 6
                    or sum(u is not None for u in roster) != count
                    or any(u is not None and (not isinstance(u, str) or u not in UNITS) for u in roster)):
                raise ValueError('Roster must have six slots and the declared count of known units')
        profiles, rows, traits = [], [], []
        index = 0
        for slot in range(6):
            occupied = roster[slot] is not None if roster is not None else slot < count
            if not occupied:
                profiles.append(None)
                rows.append([0]*5)
                traits.append([EMPTY, 0, 0, 0])
                continue
            if roster is not None:
                defaults = dict(UNITS[roster[slot]])
            else:
                warrior = slot == (game_map.get('hero_warrior_slot', -1) if enemy is None else warriors[enemy])
                mage = enemy is None and slot == game_map.get('hero_mage_slot', -1)
                role = 'warrior' if warrior else 'mage' if mage else 'archer'
                health = (game_map.get('warrior_hp', 100) if warrior else
                          game_map.get('mage_hp', game_map.get('hero_hp', 45)) if mage else
                          game_map.get('hero_hp', 45)) if enemy is None else game_map['enemy_hp'][enemy]
                defaults = dict(name='Воин' if warrior else 'Маг' if mage else 'Лучник', level=1,
                    role='melee' if warrior else 'area' if mage else 'ranged',
                    max_hp=health, damage=game_map.get(role+'_damage', 20 if mage else 25),
                    accuracy=100*game_map.get(role+'_accuracy', game_map.get('archer_accuracy', .8)),
                    armor=game_map.get(role+'_armor', 0), initiative=game_map.get(role+'_initiative', 50 if warrior else 60),
                    attack_type='fire' if mage else 'weapon', immunities=[], protections=[])
            override = overrides[index]
            index += 1
            if not isinstance(override, dict) or set(override) - PROFILE_FIELDS:
                raise ValueError('Unknown combat profile fields')
            values = {**defaults, **override}
            if values.get('upgrade_unavailable_reason'):
                raise ValueError(values['upgrade_unavailable_reason'])
            if not isinstance(values['role'], str) or values['role'] not in ROLES:
                raise ValueError('Unknown combat role')
            for field in ('exp_kill', 'exp_required', 'exp_current'):
                value = values.get(field, 0)
                if type(value) is not int or not 0 <= value <= 1_000_000:
                    raise ValueError('Invalid combat '+field)
            if values.get('exp_required', 0) > 0 and values.get('exp_current', 0) >= values['exp_required']:
                raise ValueError('Initial experience must be below the level threshold')
            stats = _validated_stats(values)
            source = _source(values['attack_type'])
            immune, wards = _protection_mask(values['immunities']), _protection_mask(values['protections'])
            profiles.append(dict(name=values['name'], level=values['level'], role=values['role'],
                                 attack_type=ATTACK_TYPES[source], immunities=immune, protections=wards))
            rows.append(stats)
            traits.append([ROLES[values['role']], source+1, immune, wards])
        return profiles, rows, traits

    hero_names, heroes, hero_traits = formation(hero_count, hero_overrides, hero_roster)
    squads, traits, enemies = [], [], []
    for enemy, count in enumerate(counts):
        names, rows, properties = formation(count, enemy_overrides[enemy],
            enemy_rosters[enemy] if enemy_rosters is not None else None, enemy)
        squads.append(heroes+rows)
        traits.append(hero_traits+properties)
        enemies.append(names)
    damages = sorted({s[DAMAGE] for squad in squads for s in squad})
    armors = sorted({s[ARMOR] for squad in squads for s in squad})
    damage_ids = [[damages.index(s[DAMAGE]) for s in squad] for squad in squads]
    armor_ids = [[armors.index(s[ARMOR]) for s in squad] for squad in squads]
    # Preserve Python float/round semantics, including half-integer edge cases.
    rolls = [[[[int(round((d+b)*(1.-a/100.)*(.5 if defend else 1.)))
                if d > 0 else 0 for b in range(6)] for defend in (False, True)]
              for a in armors] for d in damages]
    metadata = dict(heroes=hero_names, enemies=enemies,
                    attack_types=[dict(key=key, name=label, bit=1 << i)
                                  for i, (key, label) in enumerate(zip(ATTACK_TYPES, ATTACK_LABELS))])
    return (jnp.asarray(squads, jnp.float32), jnp.asarray(damage_ids, jnp.int32),
            jnp.asarray(armor_ids, jnp.int32), jnp.asarray(rolls, jnp.int32),
            max(s[INITIATIVE] for squad in squads for s in squad)+10.,
            jnp.asarray(traits, jnp.uint32), metadata,
            any(t[2] or t[3] for squad in traits for t in squad))


def accuracy_hits(accuracy, first, second):
    """Python reference: average two independent integer draws from 0..99."""
    a = jnp.clip(jnp.floor(first*100), 0, 99)
    b = jnp.clip(jnp.floor(second*100), 0, 99)
    return a+b < 2*accuracy
