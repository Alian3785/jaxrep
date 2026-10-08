"""Host-side stat tables and pure JAX rules for combat version 2.

Damage rounding deliberately uses Python's float/round semantics at map load,
including half-integer edge cases. The learner only indexes small GPU tables.
See docs/BASIC_COMBAT.md for sources and the limits of original-game parity.
"""
import math

import jax.numpy as jnp

HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE = range(5)
STAT_NAMES = ('max_hp', 'damage', 'accuracy', 'armor', 'initiative')


def _validated_stats(defaults, overrides):
    if not isinstance(overrides, dict) or set(overrides) - set(STAT_NAMES):
        raise ValueError('Combat stats must be a dictionary of the five supported characteristics')
    values = {**defaults, **overrides}
    for name, value in values.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'Invalid combat {name}')
    if (type(values['max_hp']) is not int or not 1 <= values['max_hp'] <= 1_000_000
            or type(values['damage']) is not int or not 0 <= values['damage'] <= 1_000_000
            or not 0 <= values['accuracy'] <= 100 or not 0 <= values['armor'] <= 100
            or type(values['initiative']) is not int or not 0 <= values['initiative'] <= 1000):
        raise ValueError('Combat stats outside supported ranges')
    # Current roles are ordinary damage dealers, without special damage caps.
    values['damage'] = min(values['damage'], 300)
    values['armor'] = min(values['armor'], 90)
    return [values[name] for name in STAT_NAMES]


def build_combat_tables(game_map):
    """Resolve role defaults and optional per-unit overrides once, before jit."""
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

    def unit(slot, enemy=None):
        warrior = slot == (game_map.get('hero_warrior_slot', -1) if enemy is None else warriors[enemy])
        mage = enemy is None and slot == game_map.get('hero_mage_slot', -1)
        role = 'warrior' if warrior else 'mage' if mage else 'archer'
        health = (game_map.get('warrior_hp', 100) if warrior else
                  game_map.get('mage_hp', game_map.get('hero_hp', 45)) if mage else
                  game_map.get('hero_hp', 45)) if enemy is None else game_map['enemy_hp'][enemy]
        defaults = dict(max_hp=health,
                        damage=game_map.get(role + '_damage', 20 if mage else 25),
                        accuracy=100 * game_map.get(role + '_accuracy', game_map.get('archer_accuracy', .8)),
                        armor=game_map.get(role + '_armor', 0),
                        initiative=game_map.get(role + '_initiative', 50 if warrior else 60))
        overrides = hero_overrides[slot] if enemy is None else enemy_overrides[enemy][slot]
        return _validated_stats(defaults, overrides)

    heroes = [unit(i) if i < hero_count else [0] * 5 for i in range(6)]
    squads = [heroes + [unit(i, e) if i < n else [0] * 5 for i in range(6)]
              for e, n in enumerate(counts)]
    damages = sorted({s[DAMAGE] for squad in squads for s in squad})
    armors = sorted({s[ARMOR] for squad in squads for s in squad})
    damage_ids = [[damages.index(s[DAMAGE]) for s in squad] for squad in squads]
    armor_ids = [[armors.index(s[ARMOR]) for s in squad] for squad in squads]
    # Only damage/armor values present in this map are materialized (usually a few).
    rolls = [[[[int(round((d + b) * (1. - a / 100.) * (.5 if defend else 1.)))
                if d > 0 else 0 for b in range(6)] for defend in (False, True)]
              for a in armors] for d in damages]
    return (jnp.asarray(squads, jnp.float32), jnp.asarray(damage_ids, jnp.int32),
            jnp.asarray(armor_ids, jnp.int32), jnp.asarray(rolls, jnp.int32),
            max(s[INITIATIVE] for squad in squads for s in squad) + 10.)


def accuracy_hits(accuracy, first, second):
    """Python reference: average two independent integer draws from 0..99."""
    a = jnp.clip(jnp.floor(first * 100), 0, 99)
    b = jnp.clip(jnp.floor(second * 100), 0, 99)
    return a + b < 2 * accuracy
