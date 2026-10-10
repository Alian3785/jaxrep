"""Battle XP and building-backed promotions; fixed-shape JAX state, no host steps.

Rules and source pointers are recorded in docs/UNIT_PROGRESSION.md.
"""
from functools import lru_cache
import json

import jax
import jax.numpy as jnp

from stoix.envs.number_grid_combat import (
    UNITS, ROLES, HP, DAMAGE, ARMOR, STAT_NAMES,
    _validated_stats, _source, _protection_mask, uses_aoe_accuracy_falloff,
)
from stoix.envs.number_grid_buildings import CATALOG
from stoix.envs.number_grid_effects import CAPITAL_GUARDS
from stoix.envs.number_grid_casting import cast_summon_keys


@lru_cache(maxsize=8)
def _damage_rolls(armors):
    """Immutable device lookup shared by scenarios with the same armor values."""
    return jnp.array([[[[int(round((damage+bonus)*(1-armor/100)*(.5 if defend else 1)))
        if damage else 0 for bonus in range(6)] for defend in (False,True)]
        for armor in armors] for damage in range(401)],jnp.int32)


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
        self.doppelgangers = jnp.array([r.get('unit_type') == 'Doppelganger' for r in rows])
        self.copy_forbidden = jnp.array([r.get('game_data', {}).get('UNIT_CAT') == '8'
            or r.get('name') in ('Драллиаан','Лаклаан','Drulliaan','Laclaan') for r in rows])
        self.imp_copy_id = next(i for i,r in enumerate(rows) if r.get('game_id') == 'g000uu5034')
        self.fenrir_copy_id = self.ids.get('g000uu0190', next(i for i,r in enumerate(rows) if r.get('game_id') == 'g000uu0190'))
        summon_types = {'Summoner':1, 'Occultmaster':2, 'Lyf':2, 'Laclaan':2}
        self.summon_modes = jnp.array([summon_types.get(r.get('unit_type'),0) for r in rows],jnp.int32)
        self.immediate_dragons = jnp.array([r.get('unit_type') == 'Laclaan' for r in rows])
        self.summon_small = jnp.array([( [self.ids[k] for k in r.get('summon_pool',[])
            if UNITS[k].get('size',1) == 1]+[0]*8)[:8] for r in rows],jnp.int32)
        self.summon_large = jnp.array([( [self.ids[k] for k in r.get('summon_pool',[])
            if UNITS[k].get('size',1) == 2]+[0]*8)[:8] for r in rows],jnp.int32)
        self.hermits = jnp.array([r.get('unit_type') == 'Hermit' for r in rows])
        self.alchemists = jnp.array([r.get('unit_type') == 'Alchemist' for r in rows])
        self.patriarchs = jnp.array([r.get('unit_type') == 'Patriach' for r in rows])
        power_types = {'Travnitsa':1.25,'Novice':1.5,'Dwarfdruid':1.75,'Arhidruid':2.}
        self.power_factors = jnp.array([power_types.get(r.get('unit_type'),0.) for r in rows])
        self.power_cures = jnp.array([r.get('unit_type') in ('Dwarfdruid','Arhidruid') for r in rows])
        ward_types = {'Sundancer':4,'Sylfid':256,'Deva roshi':270}
        self.healer_ward_elements = jnp.array([ward_types.get(r.get('unit_type'),0) for r in rows],jnp.uint32)
        self.mass_healers = jnp.array([r.get('unit_type') in ('Profit','Deva roshi') for r in rows])
        self.mass_cures = jnp.array([r.get('unit_type') == 'Profit' and r['name'] in ('Аббатиса','Прорицательница','Matriarch','Prophetess') for r in rows])
        post_names = {'Служка','Жрец','Священник','Архангел','Медиум','Оракул','Дева рощи','Патриарх','Клирик','Аббатиса','Прорицательница','Нейтральный эльфийский оракул','Солнечная танцовщица','Сильфида'}
        self.post_healers = jnp.array([r['name'] in post_names and r['role'] == 'healer' for r in rows])
        self.wights = jnp.array([r.get('unit_type') == 'Wight' for r in rows])
        self.has_wights = any(r.get('unit_type') == 'Wight' for r in rows)
        self.lower_forms = jnp.array([self.ids.get(r.get('lower_form', ''), 0) for r in rows], jnp.int32)
        self.neutrals = jnp.array([r.get('neutral', r['faction'] == 'neutral') for r in rows])
        self.aoe_accuracy_falloff = jnp.array([uses_aoe_accuracy_falloff(r) for r in rows])
        self.has_aoe_accuracy_falloff = any(uses_aoe_accuracy_falloff(r) for r in rows)
        self.wolf_lord = jnp.array([r.get('unit_type') == 'Wolf Lord' for r in rows])
        self.has_wolf_lord = any(r.get('unit_type') == 'Wolf Lord' for r in rows)
        self.shatterers = jnp.array([r.get('unit_type') in ('Teurg','Aleman') for r in rows])
        self.has_shatterers = any(r.get('unit_type') in ('Teurg','Aleman') for r in rows)
        self.witches = jnp.array([r.get('unit_type') in ('Witch','Succub') for r in rows])
        self.has_witches = any(r.get('unit_type') in ('Witch','Succub') for r in rows)
        self.centaurs = jnp.array([r.get('unit_type') == 'Centaur Savage' for r in rows])
        self.has_centaurs = any(r.get('unit_type') == 'Centaur Savage' for r in rows)
        self.double_strike = jnp.array([r.get('unit_type') in ('Demon', 'Elfarcher') for r in rows])
        self.has_double_strike = any(r.get('unit_type') in ('Demon', 'Elfarcher') for r in rows)
        self.cached_poisoners = jnp.array([(r['name'] == 'Ниддог' or r.get('unit_type') in ('Dregazul','Spider')) for r in rows])
        self.poisoners = jnp.array([(r['name'] == 'Ниддог' or r.get('unit_type') in ('Dregazul','Spider')) or r.get('unit_type') in ('Death','Dead dragon') for r in rows])
        used_ids = set(hero) | {i for squad in enemies for i in squad}
        # Player promotions can introduce a poisoner absent from the initial
        # roster; follow supported forms before eliminating the effect at jit.
        pending, reached = list(hero), set(hero)
        while pending:
            row = rows[pending.pop()]
            for target in row['upgrades']:
                target_id = self.ids.get(target)
                if target_id is not None and target_id not in reached:
                    reached.add(target_id)
                    pending.append(target_id)
        used_ids |= reached
        # Summon spells bring their unit into a battle of its own.
        used_ids |= {self.ids[key] for key in cast_summon_keys(game_map)}
        # Enemy forms never promote, but their summons can introduce additional
        # combat behavior. Follow those pools without enabling enemy upgrades.
        pending = list(used_ids)
        while pending:
            for target in rows[pending.pop()].get('summon_pool',[]):
                target_id = self.ids[target]
                if target_id not in used_ids:
                    used_ids.add(target_id)
                    pending.append(target_id)
        self.has_healers = any(rows[i]['role'] == 'healer' for i in used_ids)
        self.has_copies = any(rows[i].get('unit_type') == 'Doppelganger' for i in used_ids)
        self.has_summons = any(rows[i].get('unit_type') in summon_types for i in used_ids)
        self.has_hermits = any(rows[i].get('unit_type') == 'Hermit' for i in used_ids)
        self.has_alchemists = any(rows[i].get('unit_type') == 'Alchemist' for i in used_ids)
        self.has_patriarchs = any(rows[i].get('unit_type') == 'Patriach' for i in used_ids)
        self.has_powerups = any(rows[i].get('unit_type') in power_types for i in used_ids)
        self.has_healer_wards = any(rows[i].get('unit_type') in ward_types for i in used_ids)
        self.has_cures = any(rows[i].get('unit_type') in ('Patriach','Dwarfdruid','Arhidruid') or (rows[i].get('unit_type') == 'Profit' and rows[i]['name'] in ('Аббатиса','Прорицательница','Matriarch','Prophetess')) for i in used_ids)
        self.has_poisoners = any((rows[i]['name'] == 'Ниддог' or rows[i].get('unit_type') in ('Dregazul','Spider')) or rows[i].get('unit_type') in ('Death','Dead dragon') for i in used_ids)
        self.cached_water = jnp.array([r.get('unit_type') == 'Ismir son' for r in rows])
        self.water_casters = jnp.array([r.get('unit_type') in ('Sentry','Ismir son','Drulliaan') for r in rows])
        self.has_water = any(rows[i].get('unit_type') in ('Sentry','Ismir son','Drulliaan') for i in used_ids)
        self.secondary_fear = jnp.array([r.get('unit_type') == 'Shamanka' for r in rows])
        self.has_secondary_fear = any(rows[i].get('unit_type') == 'Shamanka' for i in used_ids)
        self.fear_casters = jnp.array([r.get('unit_type') == 'Baroness' for r in rows])
        self.has_fear = any(rows[i].get('unit_type') in ('Baroness','Shamanka') for i in used_ids)
        self.weakeners = jnp.array([r.get('unit_type') == 'Tiamat' for r in rows])
        self.has_weakening = any(rows[i].get('unit_type') == 'Tiamat' for i in used_ids)
        self.secondary_paralysis_modes = jnp.array([1 if (r.get('game_id') in ('g000uu5026','g000uu5126') or r.get('unit_type') == 'Abyss Devil') else 2 if r.get('unit_type') in ('Betrezen','Uter','Uter Demon','Abyss Devil') else 0 for r in rows],jnp.int32)
        self.has_secondary_paralysis = any(rows[i].get('unit_type') in ('Betrezen','Uter','Uter Demon','Abyss Devil') for i in used_ids)
        self.ghost_modes = jnp.array([2 if r.get('game_id') in ('g000uu8044','g000uu8144') else 1 if r.get('unit_type') in ('Ghost','Shadow','Incub') else 0 for r in rows],jnp.int32)
        self.has_paralysis = self.has_fear or self.has_secondary_paralysis or any(rows[i].get('unit_type') in ('Ghost','Shadow','Incub') for i in used_ids)
        self.leech_modes = jnp.array([2 if r.get('unit_type') in ('Bone Lord','Highvampire') else 1 if r.get('unit_type') in ('Dregazul','Vampire') else 0 for r in rows],jnp.int32)
        self.has_leech = any(rows[i].get('unit_type') in ('Bone Lord','Dregazul','Vampire','Highvampire') for i in used_ids)
        self.cached_fire = jnp.array([r.get('unit_type') == 'Lord' for r in rows])
        self.fire_casters = jnp.array([r.get('unit_type') in ('Watcher','Lord','Gumtic') for r in rows])
        self.has_fire = any(rows[i].get('unit_type') in ('Watcher','Lord','Gumtic') for i in used_ids)
        self.secondary_base = jnp.array([[r.get('secondary_damage', 0), r.get('secondary_accuracy', 0)] for r in rows], jnp.float32)
        self.secondary_early = jnp.array([r['growth'].get('secondary_early', [0, 0]) for r in rows], jnp.float32)
        self.secondary_late = jnp.array([r['growth'].get('secondary_late', [0, 0]) for r in rows], jnp.float32)
        self.capital_guards = jnp.array([r['name'] in CAPITAL_GUARDS for r in rows])
        self.secondary_sources = jnp.array([_source(r['secondary_attack_type'])+1
            if r.get('secondary_attack_type') else 0 for r in rows], jnp.uint32)
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
        heavy_strike_ids = {'g000uu0019','g000uu0020','g000uu0044','g000uu0045','g000uu0047','g000uu0070','g000uu0071','g000uu0096','g000uu8009','g000uu8011'}
        support_types = {'Cliric','Profit','Patriach','Deva roshi','Sundancer','Sylfid','Ghost','Witch','Shadow','Incub','Succub','Baroness','Travnitsa','Novice','Alchemist','Dwarfdruid','Arhidruid'}
        self.stat_caps = jnp.array([[jnp.inf, jnp.inf if r.get('unit_type') in support_types else 400 if r.get('game_id') in heavy_strike_ids else 300, 100, 90, jnp.inf]
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
        armors = list(range(91))
        self.armor_ids = jnp.array([[armors.index(a) for a in row] for row in armor_rows], jnp.int32)
        self.shatter_armor_ids = jnp.array([[[armors.index(a if k == 0 else max(0, int(a)-15*k))
                                            for k in range(7)] for a in row] for row in armor_rows], jnp.int32)
        self.damage_rolls = _damage_rolls(tuple(armors))
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
                unit_type=row.get('unit_type', ''),
                secondary_attack_type=row.get('secondary_attack_type', ''),
                role=row['role'], size=row.get('size', 1), attack_type=row['attack_type'],
                immunities=_protection_mask(row['immunities']),
                protections=_protection_mask(row['protections']), upgrades=options,
                exp_required=row['exp_required'], exp_kill=row['exp_kill']))

    def raw_stats(self, ids, levels):
        anchor = jnp.minimum(levels, self.level_anchor)
        return self.stat_levels[anchor, ids] + jnp.maximum(levels-self.level_anchor, 0)[..., None]*self.late[ids]

    def stats(self, ids, levels):
        return jnp.minimum(self.raw_stats(ids,levels), self.stat_caps[ids])

    def secondary_stats(self, ids, levels):
        extra = jnp.maximum(levels-self.base_levels[ids], 0)
        early = jnp.minimum(extra, jnp.maximum(self.thresholds[ids]-self.base_levels[ids], 0))
        values = self.secondary_base[ids] + early[..., None]*self.secondary_early[ids] + (extra-early)[..., None]*self.secondary_late[ids]
        return jnp.minimum(values, jnp.array([300., 100.]))

    def required_xp(self, ids, levels):
        return self.required[ids] + jnp.maximum(levels-self.base_levels[ids], 0)*self.required_increment[ids]

    def experience(self, ids, levels, current):
        killed = (self.kill_levels[jnp.minimum(levels, self.kill_anchor), ids]
                  + jnp.maximum(levels-self.kill_anchor, 0)*self.kill_late[ids])
        return jnp.stack((levels, killed, self.required_xp(ids, levels), current), axis=-1)

    def damage(self, state, actor, targets, guarded, bonuses, damage_value=None, armor_shreds=None, imps=None, armor_values=None):
        if damage_value is None:
            damage_value = self.stats(state.unit_ids, state.unit_levels)[actor, DAMAGE]
        levels, ids = jnp.minimum(state.unit_levels[targets], self.armor_max_level), state.unit_ids[targets]
        armor_ids = (self.armor_ids[levels, ids] if armor_shreds is None
                     else self.shatter_armor_ids[levels, ids, jnp.minimum(armor_shreds[targets], 6)])
        if imps is not None:
            armor_ids = jnp.where(imps[targets], 0, armor_ids)
        if armor_values is not None:
            armor_ids = jnp.clip(armor_values,0,90).astype(jnp.int32)
        return self.damage_rolls[damage_value.astype(jnp.int32), armor_ids,
                                 guarded.astype(jnp.int32), bonuses]

    def finish(self, state, hp, escaped, victory, lost, withdrawal, bank, experience_book=False):
        """Award once, before automatic resurrection; one promotion per battle."""
        ended = victory | lost | withdrawal
        winners = (jnp.arange(12) < 6) == victory
        recipients = (hp > 0) & ~escaped & winners & ended & (state.unit_ids != 0)
        recipients &= self.required[state.unit_ids] > 0
        count = jnp.maximum(jnp.sum(recipients), 1)
        total = jnp.where(victory, bank[1], bank[0])
        # floor(total/count + .5), exact integer arithmetic, not bankers' round.
        scale=jnp.where(victory & experience_book,5,4)
        whole,remainder=total//count,total%count
        extra=jnp.where(scale==5,whole//4,0)
        fraction=jnp.where(scale==5,whole%4,0)*count+remainder*scale
        award=whole+extra+(fraction+2*count)//(4*count)
        gains = jnp.where(recipients, award, 0)
        if self.has_patriarchs:
            cutoffs = state.revival_xp_cutoff.reshape(2,6)
            masks = recipients.reshape(2,6)
            gains = jax.vmap(revival_xp_shares,in_axes=(None,0,0,None))(total,cutoffs,masks,scale).reshape(12)
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

def revival_xp_shares(total,cutoffs,recipients,scale=4):
    """Six final winners, each eligible only after their last revival cutoff.

    Splits every kill among eligible final survivors, exactly half-up. Quotients
    plus sixtieths avoid both float ties and multiplying a large XP bank by 60.
    """
    cuts = jnp.where(recipients,cutoffs,total)
    ordered = jnp.sort(cuts)
    interval = jnp.concatenate((ordered[1:],total[None]))-ordered
    counts = jnp.arange(1,7,dtype=jnp.int32)
    whole,remainder = interval//counts,interval%counts
    fraction = remainder*(60//counts)
    include = recipients[:,None] & (ordered[None,:] >= cuts[:,None])
    whole=jnp.sum(jnp.where(include,whole,0),axis=1)
    fraction=jnp.sum(jnp.where(include,fraction,0),axis=1)
    # Apply Tome of War before final half-up rounding, including late revivals.
    extra=jnp.where(scale==5,whole//4,0)
    remainder=jnp.where(scale==5,whole%4,0)
    return whole+extra+(remainder*60+fraction*scale+120)//240
