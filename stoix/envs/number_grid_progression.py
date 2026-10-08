"""Battle XP and building-backed promotions; fixed-shape JAX state, no host steps.

Rules and source pointers are recorded in docs/UNIT_PROGRESSION.md.
"""
import json

import jax.numpy as jnp

from stoix.envs.number_grid_combat import (
    UNITS, ROLES, HP, DAMAGE, ARMOR, STAT_NAMES,
    _validated_stats, _source, _protection_mask,
)
from stoix.envs.number_grid_buildings import CATALOG


class ProgressionRules:
    def __init__(self, game_map, construction):
        if 'hero_roster' not in game_map or 'enemy_rosters' not in game_map:
            raise ValueError('Unit progression requires named rosters')
        self.ids = {}
        rows = [dict(name='Пусто', level=0, exp_kill=0, exp_required=0,
                     exp_current=0, upgrades=[], faction='', role='ranged',
                     max_hp=0, damage=0, accuracy=0, armor=0, initiative=0,
                     attack_type='weapon', immunities=[], protections=[],
                     growth=dict(threshold=10, early=[0]*5, late=[0]*5,
                                 kill_early=0, kill_late=0))]
        for key, values in UNITS.items():
            if not values.get('upgrade_unavailable_reason'):
                self.ids[key] = len(rows)
                rows.append(dict(values, key=key))
        cache = {}

        def roster_ids(roster, overrides):
            result, occupied = [], 0
            for key in roster:
                if key is None:
                    result.append(0)
                    continue
                override = overrides[occupied]
                occupied += 1
                if not override:
                    result.append(self.ids[key])
                    continue
                signature = json.dumps([key, override], sort_keys=True)
                if signature not in cache:
                    cache[signature] = len(rows)
                    rows.append({**UNITS[key], **override, 'key': key})
                result.append(cache[signature])
            return result

        hero = roster_ids(game_map['hero_roster'], game_map.get(
            'hero_combat_stats', [{} for _ in range(game_map['hero_units'])]))
        enemy_overrides = game_map.get('enemy_combat_stats',
                                      [[{} for _ in range(n)] for n in game_map['enemy_units']])
        enemies = [roster_ids(roster, overrides) for roster, overrides in
                   zip(game_map['enemy_rosters'], enemy_overrides)]
        levels = [r['level'] for r in rows]
        self.rows = rows
        self.sizes = jnp.array([0]+[r.get('size', 1) for r in rows[1:]], jnp.int32)
        self.base_levels = jnp.array(levels, jnp.int32)
        self.initial_ids = jnp.array(hero+[0]*6, jnp.int32)
        self.initial_levels = self.base_levels[self.initial_ids]
        initial_xp = jnp.array([r.get('exp_current', 0) for r in rows], jnp.int32)
        self.initial_xp = initial_xp[self.initial_ids]
        enemy_ids = jnp.array(enemies, jnp.int32)
        # Opponents have no capital, so their forms are immutable scenario data.
        self.enemy_ids = enemy_ids
        self.initial_enemies = jnp.stack((self.base_levels[enemy_ids], initial_xp[enemy_ids]), axis=-1)
        self.base_stats = jnp.array([[0]*5]+[_validated_stats(r) for r in rows[1:]], jnp.float32)
        self.traits = jnp.array([[0]*4]+[[ROLES[r['role']], _source(r['attack_type'])+1,
            _protection_mask(r['immunities']), _protection_mask(r['protections'])] for r in rows[1:]], jnp.uint32)
        self.thresholds = jnp.array([r['growth']['threshold'] for r in rows], jnp.int32)
        self.early = jnp.array([r['growth']['early'] for r in rows], jnp.float32)
        self.late = jnp.array([r['growth']['late'] for r in rows], jnp.float32)
        self.kill_base = jnp.array([r['exp_kill'] for r in rows], jnp.int32)
        self.kill_early = jnp.array([r['growth']['kill_early'] for r in rows], jnp.int32)
        self.kill_late = jnp.array([0 if r.get('hero') else r['growth']['kill_late'] for r in rows], jnp.int32)
        # A tiny lookup covers the early levels; later growth is linear. This
        # avoids gathering five separate profile arrays on every environment step.
        self.level_anchor = max(max(r['growth']['threshold'],
            max((b['level'] for b in r.get('level_bonuses', [])), default=0)) for r in rows)
        # Heroes use the reference's rounded x1.1 kill XP, not GDynUpgr's
        # fixed increment. Tabulate every value representable by the JAX int32
        # state; saturation only concerns otherwise overflowing synthetic levels.
        hero_kills = {}
        for i, row in enumerate(rows):
            if row.get('hero'):
                curve = [row['exp_kill']] * (row['level']+1)
                while 0 < curve[-1] < 2**31-1:
                    following = min(2**31-1, int(round(curve[-1]*1.1)))
                    if following == curve[-1]:
                        break
                    curve.append(following)
                hero_kills[i] = curve
        self.kill_anchor = max(self.level_anchor, max((len(c)-1 for c in hero_kills.values()), default=0))

        def grown_stats(row, level):
            extra = max(level-row['level'], 0)
            early = min(extra, max(row['growth']['threshold']-row['level'], 0))
            late = extra-early
            if not row.get('level_bonuses'):
                return [row[k]+early*a+late*b for k, a, b in zip(
                    STAT_NAMES, row['growth']['early'], row['growth']['late'])]
            values = [row[k] for k in STAT_NAMES]
            for next_level in range(row['level']+1, level+1):
                increment = row['growth']['early' if next_level <= row['growth']['threshold'] else 'late']
                values = [v+inc for v, inc in zip(values, increment)]
                for bonus in row['level_bonuses']:
                    if bonus['level'] == next_level:
                        index = STAT_NAMES.index(bonus['stat'])
                        value = values[index]*bonus['multiplier']+bonus['bonus']
                        values[index] = int(value+.5) if bonus['rounding'] == 'half_up' else round(value)
                values[2] = min(values[2], 100)
            return values

        def killed_value(i, row, level):
            if i in hero_kills:
                return hero_kills[i][min(level, len(hero_kills[i])-1)]
            extra = max(level-row['level'], 0)
            early = min(extra, max(row['growth']['threshold']-row['level'], 0))
            return row['exp_kill']+early*row['growth']['kill_early']+(extra-early)*row['growth']['kill_late']

        self.stat_levels = jnp.array([[grown_stats(r, level) for r in rows]
                                     for level in range(self.level_anchor+1)], jnp.float32)
        self.kill_levels = jnp.array([[killed_value(i, r, level) for i, r in enumerate(rows)]
                                     for level in range(self.kill_anchor+1)], jnp.int32)
        self.stat_caps = jnp.array([[jnp.inf, 400 if r.get('hero') else 300, 100, 90, jnp.inf]
                                    for r in rows], jnp.float32)
        self.required = jnp.array([r['exp_required'] for r in rows], jnp.int32)
        self.required_increment = jnp.array([r.get('exp_increment', 0) for r in rows], jnp.int32)
        self.dynamic = jnp.array([i > 0 and not r['upgrades'] for i, r in enumerate(rows)])
        targets, bits = [], []
        for row in rows:
            choices, required_bits = [], []
            for key in row['upgrades']:
                target = UNITS[key]
                building = next((i for i, b in enumerate(construction.rows)
                                 if b['unit'] == target['name']), None)
                usable = (key in self.ids and row['faction'] == construction.faction
                          and building is not None)
                choices.append(self.ids[key] if usable else 0)
                required_bits.append((1 << building) if usable else 0)
            # The reference's order decides between multiple built alternatives.
            targets.append((choices+[0, 0])[:2])
            bits.append((required_bits+[0, 0])[:2])
            if len(choices) > 2:
                raise ValueError('Progression catalogue supports at most two direct evolutions')
        self.targets = jnp.array(targets, jnp.int32)
        self.building_bits = jnp.array(bits, jnp.uint32)
        # Preserve Python float/round for every possible damage (400 for heroes, 300 for other units).
        # Integer armour growth reaches its cap within this lookup. The first
        # ten levels use the early increment, all later levels the late one.
        armor_rows = []
        self.armor_max_level = self.level_anchor+100
        for level in range(self.armor_max_level+1):
            armor_rows.append([min(90, grown_stats(row, level)[ARMOR]) for row in rows])
        armors = sorted({value for row in armor_rows for value in row})
        self.armor_ids = jnp.array([[armors.index(a) for a in row] for row in armor_rows], jnp.int32)
        self.damage_rolls = jnp.array([[[[int(round((damage+bonus)*(1-armor/100)*(.5 if defend else 1)))
             if damage else 0 for bonus in range(6)] for defend in (False, True)]
             for armor in armors] for damage in range(401)], jnp.int32)
        self.has_protections = any(r['immunities'] or r['protections'] for r in rows)
        self.metadata = []
        for i, row in enumerate(rows):
            options = []
            for target in row['upgrades']:
                target_row = UNITS[target]
                buildings = CATALOG['factions'][row['faction']]['buildings']
                building = next((b for b in buildings if b['unit'] == target_row['name']), None)
                options.append(dict(name=target_row['name'],
                    building=building['name'] if building else None,
                    supported=not bool(target_row.get('upgrade_unavailable_reason')),
                    reason=target_row.get('upgrade_unavailable_reason', '')))
            self.metadata.append(None if i == 0 else dict(
                name=row['name'], key=row['key'], faction=row['faction'], level=row['level'],
                hero=row.get('hero', False), level_bonuses=row.get('level_bonuses', []),
                role=row['role'], size=row.get('size', 1), attack_type=row['attack_type'],
                immunities=_protection_mask(row['immunities']),
                protections=_protection_mask(row['protections']), upgrades=options,
                exp_required=row['exp_required'], exp_kill=row['exp_kill']))

    def stats(self, ids, levels):
        anchor = jnp.minimum(levels, self.level_anchor)
        values = self.stat_levels[anchor, ids] + jnp.maximum(levels-self.level_anchor, 0)[..., None]*self.late[ids]
        return jnp.minimum(values, self.stat_caps[ids])

    def required_xp(self, ids, levels):
        return self.required[ids] + jnp.maximum(levels-self.base_levels[ids], 0)*self.required_increment[ids]

    def experience(self, ids, levels, current):
        killed = (self.kill_levels[jnp.minimum(levels, self.kill_anchor), ids]
                  + jnp.maximum(levels-self.kill_anchor, 0)*self.kill_late[ids])
        return jnp.stack((levels, killed, self.required_xp(ids, levels), current), axis=-1)

    def damage(self, state, actor, targets, guarded, bonuses):
        stats = self.stats(state.unit_ids, state.unit_levels)
        return self.damage_rolls[stats[actor, DAMAGE].astype(jnp.int32),
                                 self.armor_ids[jnp.minimum(state.unit_levels[targets], self.armor_max_level),
                                                state.unit_ids[targets]],
                                 guarded.astype(jnp.int32), bonuses]

    def finish(self, state, hp, escaped, victory, lost, withdrawal, bank):
        """Award once, before automatic resurrection; one promotion per battle."""
        ended = victory | lost | withdrawal
        winners = (jnp.arange(12) < 6) == victory
        recipients = (hp > 0) & ~escaped & winners & ended & (state.unit_ids != 0)
        count = jnp.maximum(jnp.sum(recipients), 1)
        total = jnp.where(victory, bank[1], bank[0])
        # floor(total/count + .5), exact integer arithmetic, not bankers' round.
        award = (2*total+count)//(2*count)
        gains = jnp.where(recipients, award, 0)
        xp = state.unit_xp + gains
        ids, levels = state.unit_ids, state.unit_levels
        required = self.required_xp(ids, levels)
        reached = (gains > 0) & (required > 0) & (xp >= required)
        choices, bits = self.targets[ids], self.building_bits[ids]
        # Opponents have no capital in this scenario. They still retain XP and
        # can gain dynamic levels if their form has no further evolution.
        built = ((state.buildings & bits) == bits) & (bits != 0) & (jnp.arange(12)[:, None] < 6)
        built &= (state.blocked_buildings & bits) == 0
        target = jnp.where(built[:, 0], choices[:, 0], jnp.where(built[:, 1], choices[:, 1], 0))
        evolved = reached & (target != 0)
        grown = reached & self.dynamic[ids]
        changed = evolved | grown
        ids = jnp.where(evolved, target, ids)
        levels = jnp.where(evolved, self.base_levels[ids], levels + grown.astype(jnp.int32))
        xp = jnp.where(changed, 0, jnp.where(reached, required-1, xp))
        new_hp = jnp.where(changed, self.stats(ids, levels)[:, HP].astype(jnp.int32), hp)
        saved = jnp.stack((levels[6:], xp[6:]), axis=-1)
        enemies = state.enemy_progress.at[state.enemy].set(
            jnp.where(ended, saved, state.enemy_progress[state.enemy]))
        promoted = jnp.sum(jnp.where(changed, jnp.left_shift(jnp.uint32(1), jnp.arange(12, dtype=jnp.uint32)), jnp.uint32(0)))
        return ids, levels, xp, enemies, new_hp, gains, promoted
