"""Map casts of researched spells. Sources: docs/SPELL_CASTING.md.

Python reference campaign_env_magic.py: offensive spells hit one nearest
living stack (Chebyshev distance, then the lower stack index, no pathfinding).
Damage and debuffs skip stacks on city/ruin sites; summons reach any stack and
immediately start a separate battle of the summoned unit alone. Support spells
affect the hero stack until the next REST. Every check is shared by mask/step.
"""
import jax.numpy as jnp
import numpy as np

from stoix.envs.number_grid_combat import ATTACK_TYPES, HP
from stoix.envs.number_grid_spells import shop_spell_ids, spell_book
from stoix.envs.number_grid_territory import CAPITAL_REGENERATION, CITY_BONUS

SPELL_CAST, SPELL_KILL, SUMMON_CAST = 48, 49, 50
KINDS = ('damage', 'debuff', 'summon_battle', 'moves', 'heal', 'buff', 'ward', 'health_bonus')
DAMAGE, DEBUFF, SUMMON, MOVES, HEAL, BUFF, WARD, HEALTH = range(len(KINDS))
# Per-slot battle modifiers: HP add, damage x, accuracy x, armour add, initiative x.
MOD_STATS = ('health', 'damage', 'accuracy', 'armor', 'initiative')
NEUTRAL = (0., 1., 1., 0., 1.)
CAST_REWARD = .005  # reward_spell_cast; magic kills use reward_magic_enemy_defeat_multiplier=0.
SUMMON_ENGAGE_REWARD = .05  # reward_summon_hero_battle_engage
FRONT_SLOT, REAR_SLOT = 0, 3  # reference positions 7 and 10
FIELD_REGENERATION, OWN_LAND_REGENERATION = 15, 5


def casting_enabled(game_map):
    return bool(game_map.get('spell_casting', False))


def _rows(game_map):
    own, extras = spell_book(game_map)
    return own + extras


def cast_action_names(game_map):
    if not casting_enabled(game_map):
        return ()
    return tuple('cast_' + row['id'] for row in _rows(game_map))


def cast_summon_keys(game_map):
    """Summoned forms enter the compiled catalogue features of their faction."""
    if not casting_enabled(game_map):
        return ()
    return tuple(row['cast']['unit'] for row in _rows(game_map) if row['cast']['kind'] == 'summon_battle')


class SpellCastRules:
    def __init__(self, game_map, research, construction, progression, territory, ruins, start):
        own, extras = spell_book(game_map)
        self.rows = own + extras
        self.count = len(self.rows)
        self.start, self.end = start, start + self.count
        self.limit = 2 if construction.lord['id'] == 'mage' else 1
        sold = set(shop_spell_ids(game_map))
        if sold:
            # A purchase bypasses the ruler's level limit for casting, never research
            # (_is_spellbook_spell_blocked_by_typeoflord). Other factions' spells have
            # their own actions, limited to one cast a day (_spell_shop_cast_limit_reached).
            self.bits = jnp.left_shift(jnp.uint32(1), jnp.arange(self.count, dtype=jnp.uint32))
            self.allowed = jnp.concatenate((research.allowed, jnp.ones(len(extras), bool))) | jnp.array(
                [row['id'] in sold for row in self.rows])
            self.limits = jnp.array([self.limit] * len(own) + [1] * len(extras), jnp.int32)
        else:
            self.bits, self.allowed, self.limits = research.bits, research.allowed, self.limit
        casts = [row['cast'] for row in self.rows]
        kinds = [KINDS.index(c['kind']) for c in casts]
        self.kinds = jnp.array(kinds, jnp.int32)
        self.costs = jnp.array([c['cast_mana'] for c in casts], jnp.int32)
        self.offensive = jnp.array([k in (DAMAGE, DEBUFF, SUMMON) for k in kinds])
        self.field_only = jnp.array([k in (DAMAGE, DEBUFF) for k in kinds])
        self.amounts = jnp.array([c.get('amount', 0) for c in casts], jnp.int32)
        self.element_bits = jnp.array([1 << ATTACK_TYPES.index(c['element']) if 'element' in c and k == DAMAGE else 0
                                       for c, k in zip(casts, kinds)], jnp.uint32)
        # Original game: a percentage of the hero's full allowance (boots and level
        # included), fraction dropped like boots; the reference uses MOVES_PER_TURN.
        self.percents = jnp.array([c.get('percent', 0) for c in casts], jnp.int32)
        mods = np.tile(np.array(NEUTRAL, np.float32), (self.count, 1))
        wards, forest, water = [0] * self.count, 0, 0
        for i, (c, k) in enumerate(zip(casts, kinds)):
            if k in (BUFF, DEBUFF) and c['stat'] == 'terrain':
                forest |= (1 << i) if c['terrain'] == 'forest' else 0
                water |= (1 << i) if c['terrain'] == 'water' else 0
            elif k in (BUFF, DEBUFF):
                column = MOD_STATS.index(c['stat'])
                mods[i, column] = c['amount'] if c['stat'] == 'armor' else c['multiplier']
            elif k == HEALTH:
                mods[i, 0] = c['amount']
            elif k == WARD:
                wards[i] = 1 << ATTACK_TYPES.index(c['element'])
        self.mods = jnp.array(mods)
        self.ward_bits = jnp.array(wards, jnp.uint32)
        self.forest_bits, self.water_bits = jnp.uint32(forest), jnp.uint32(water)
        # A support effect cannot be renewed; terrain spells also exclude the same terrain.
        blockers = [1 << i | (forest if forest >> i & 1 else 0) | (water if water >> i & 1 else 0)
                    for i in range(self.count)]
        self.blockers = jnp.array(blockers, jnp.uint32)
        self.persistent = jnp.array([k in (BUFF, WARD, HEALTH) for k in kinds])
        self.has_mods = bool((mods != np.array(NEUTRAL)).any())
        self.has_health = HEALTH in kinds
        self.has_wards = any(wards)
        self.has_terrain = bool(forest or water)
        self.summon_ids = jnp.array([progression.ids[c['unit']] if k == SUMMON else 0
                                     for c, k in zip(casts, kinds)], jnp.int32)
        # _build_summoned_blue_team/_entry_stand: weapon or large units stand ahead.
        units = [progression.rows[progression.ids[c['unit']]] if k == SUMMON else None for c, k in zip(casts, kinds)]
        slots = [FRONT_SLOT if u is None or u['attack_type'] == 'weapon' or u.get('size', 1) == 2 else REAR_SLOT
                 for u in units]
        self.summon_slots = jnp.array(slots, jnp.int32)
        self.progression, self.territory, self.size = progression, territory, int(game_map['size'])
        positions = game_map['opponent_positions']
        self.enemy_count = len(positions)
        self.positions = jnp.array(positions, jnp.int32)
        protected = np.array([city >= 0 for city in np.asarray(territory.enemy_city)])
        if ruins is not None:
            protected |= np.asarray(ruins.enemy_ruin) >= 0
        if territory.has_capital_guards:
            protected |= np.asarray(territory.capital_guards)  # capital location, not a field stack
        self.field = jnp.array(~protected)
        # Reference _heal_wounded_enemy_teams_for_turn: the objective dragon recovers fully.
        full = game_map.get('enemy_full_regeneration', [])
        if any(type(i) is not int or not 0 <= i < len(game_map['opponent_positions']) for i in full):
            raise ValueError('enemy_full_regeneration must list stack indices')
        self.full_regeneration = jnp.array([i in full for i in range(len(game_map['opponent_positions']))]) if full else None
        self.enemy_city = territory.enemy_city
        self.enemy_ids = progression.enemy_ids
        self.occupied = self.enemy_ids != 0
        traits = progression.traits[self.enemy_ids]
        self.blocks = traits[..., 2] | traits[..., 3]  # immunity or ward stops spell damage
        base = np.asarray(progression.stats(self.enemy_ids, progression.base_levels[self.enemy_ids])[..., HP])
        self.base_totals = jnp.array(np.maximum(base.sum(axis=1), 1), jnp.float32)
        self.city_bonus = jnp.array(CITY_BONUS, jnp.int32)
        self.observation_size = 2 * self.count + 2 * self.enemy_count + 5
        self.metadata = dict(start=start, daily_limit=self.limit, reward=CAST_REWARD,
            summon_engage_reward=SUMMON_ENGAGE_REWARD,
            spells=[dict(id=row['id'], name=row['name'], action=start + i, **row['cast'],
                         level=row['level'], description=row['description'], foreign=i >= len(own), sold=row['id'] in sold,
                         daily_limit=self.limit if i < len(own) else 1,
                         summon_slot=slots[i] if kinds[i] == SUMMON else None,
                         unit_name=(progression.rows[progression.ids[row['cast']['unit']]]['name']
                                    if row['cast']['kind'] == 'summon_battle' else None))
                    for i, row in enumerate(self.rows)])

    # Targeting -----------------------------------------------------------
    def targets(self, state):
        """Nearest living stack and nearest field stack with their distances."""
        distance = jnp.max(jnp.abs(self.positions - state.position), axis=1)
        # One lexicographic key: distance, then the lower stack index.
        key = distance * self.enemy_count + jnp.arange(self.enemy_count)
        worst = jnp.iinfo(jnp.int32).max
        any_key = jnp.where(state.alive, key, worst)
        field_key = jnp.where(state.alive & self.field, key, worst)
        any_target, field_target = jnp.argmin(any_key), jnp.argmin(field_key)
        any_target = jnp.where(any_key[any_target] < worst, any_target, -1).astype(jnp.int32)
        field_target = jnp.where(field_key[field_target] < worst, field_target, -1).astype(jnp.int32)
        return any_target, field_target, distance

    def enemy_max_hp(self, state):
        return self.progression.stats(self.enemy_ids, state.enemy_progress[..., 0])[..., HP].astype(jnp.int32)

    # Masks ---------------------------------------------------------------
    def available(self, state, max_hp, cap, targets=None, index=None):
        any_target, field_target, _ = self.targets(state) if targets is None else targets
        sel = (lambda x: x) if index is None else (lambda x: x[index])
        learned = (state.learned_spells & sel(self.bits)) != 0
        limits = self.limits if isinstance(self.limits, int) else sel(self.limits)
        ready = (sel(self.allowed) & learned & (sel(state.spell_casts) < limits)
                 & jnp.all(state.mana >= sel(self.costs), axis=-1) & ~state.in_battle & ~state.done)
        living = (state.hp[:6] > 0) & (state.unit_ids[:6] != 0)
        kind = sel(self.kinds)
        support = jnp.any(living) & jnp.where(kind == MOVES, state.movement_points < cap,
            jnp.where(kind == HEAL, jnp.any(living & (state.hp[:6] < max_hp[:6])),
                      (state.spell_effects & sel(self.blockers)) == 0))
        offense = jnp.where(sel(self.field_only), field_target >= 0, any_target >= 0)
        return ready & jnp.where(sel(self.offensive), offense, support)

    def quotes(self, state, max_hp, cap):
        """UI: status code and the exact stack each offensive spell would hit."""
        any_target, field_target, _ = targets = self.targets(state)
        target = jnp.where(self.offensive, jnp.where(self.field_only, field_target, any_target), -1)
        return dict(status=self.status(state, max_hp, cap, targets), target=target)

    def status(self, state, max_hp, cap, targets=None):
        """0 ready, 1 unknown, 2 used today, 3 mana, 4 no target/effect, 5 battle, 6 done."""
        learned = self.allowed & ((state.learned_spells & self.bits) != 0)
        mana = jnp.all(state.mana >= self.costs, axis=-1)
        code = jnp.where(self.available(state, max_hp, cap, targets), 0, 4)
        code = jnp.where(mana, code, 3)
        code = jnp.where(state.spell_casts < self.limits, code, 2)
        code = jnp.where(learned, code, 1)
        return jnp.where(state.done, 6, jnp.where(state.in_battle, 5, code))

    # World transitions ---------------------------------------------------
    def apply(self, state, action, max_hp, cap, targets=None):
        targets = self.targets(state) if targets is None else targets
        any_target, field_target, _ = targets
        index = jnp.clip(action - self.start, 0, self.count - 1)
        valid = (action >= self.start) & (action < self.end) & self.available(state, max_hp, cap, targets, index)
        kind = self.kinds[index]
        target = jnp.where(self.field_only[index], field_target, any_target)
        t = jnp.maximum(target, 0)
        # Damage: a direct hit on every living unit of the stack, ignoring armour.
        striking = valid & (kind == DAMAGE)
        maximum = self.progression.stats(self.enemy_ids[t], state.enemy_progress[t, :, 0])[:, HP].astype(jnp.int32)
        current = jnp.where(self.occupied[t], jnp.maximum(maximum - state.enemy_wounds[t], 0), 0)
        hit = striking & (current > 0) & ((self.blocks[t] & self.element_bits[index]) == 0)
        remaining = jnp.where(hit, jnp.maximum(current - self.amounts[index], 0), current)
        killed = striking & ~jnp.any(remaining > 0)
        wounds = state.enemy_wounds.at[t].set(jnp.where(striking, maximum - remaining, state.enemy_wounds[t]))
        debuffs = state.enemy_spell_effects.at[t].set(jnp.where(
            killed, jnp.uint32(0), state.enemy_spell_effects[t] | jnp.where(valid & (kind == DEBUFF), self.bits[index], jnp.uint32(0))))
        # Hero stack support.
        living = (state.hp[:6] > 0) & (state.unit_ids[:6] != 0)
        healing = valid & (kind == HEAL) & living
        healed = jnp.where(healing, jnp.minimum(max_hp[:6], state.hp[:6] + self.amounts[index]), state.hp[:6])
        restore = cap * self.percents[index] // 100
        movement = jnp.where(valid & (kind == MOVES), jnp.minimum(cap, state.movement_points + restore),
                             state.movement_points)
        summon = valid & (kind == SUMMON)
        amount = jnp.where(striking, jnp.sum(current - remaining), jnp.where(
            kind == HEAL, jnp.sum(healed - state.hp[:6]), movement - state.movement_points))
        state = state.replace(
            mana=state.mana - jnp.where(valid, self.costs[index], 0),
            spell_casts=state.spell_casts.at[index].add(valid.astype(jnp.int32)),
            enemy_wounds=wounds, enemy_spell_effects=debuffs,
            alive=state.alive.at[t].set(state.alive[t] & ~killed),
            number=state.number + killed.astype(jnp.int32),
            hp=state.hp.at[:6].set(healed), movement_points=movement,
            spell_effects=state.spell_effects | jnp.where(valid & self.persistent[index], self.bits[index], jnp.uint32(0)),
            summon_marks=state.summon_marks.at[t].set(state.summon_marks[t] | summon),
            last_cast_spell=jnp.where(valid, index, -1).astype(jnp.int32),
            last_spell_target=jnp.where(valid & self.offensive[index], target, -1).astype(jnp.int32),
            last_spell_amount=jnp.where(valid & ~self.offensive[index] | striking, amount, 0).astype(jnp.int32),
            last_event=jnp.where(valid, jnp.where(killed, SPELL_KILL, jnp.where(summon, SUMMON_CAST, SPELL_CAST)),
                                 state.last_event))
        state = self._summon(state, summon, index, t)
        return state, CAST_REWARD * valid.astype(jnp.float32), summon

    def _summon(self, state, summon, index, target):
        """Stash the hero roster; the summoned unit fights alone at its native level."""
        uid = self.summon_ids[index]
        level = self.progression.base_levels[uid]
        health = self.progression.stats(uid, level)[HP].astype(jnp.int32)
        slots = jnp.arange(6) == self.summon_slots[index]
        party = jnp.stack((state.unit_ids[:6], state.unit_levels[:6], state.unit_xp[:6], state.hp[:6]))
        def replace(values, value):
            return values.at[:6].set(jnp.where(summon, jnp.where(slots, value, 0), values[:6]))
        return state.replace(spell_party=jnp.where(summon, party, state.spell_party),
            unit_ids=replace(state.unit_ids, uid), unit_levels=replace(state.unit_levels, level),
            unit_xp=replace(state.unit_xp, 0), hp=replace(state.hp, health),
            summon_battle=summon, enemy=jnp.where(summon, target, state.enemy))

    def engage_bonus(self, state, engage):
        """Hero attack on a stack a summon already fought this day."""
        marked = engage & state.summon_marks[jnp.maximum(state.enemy, 0)]
        return state.replace(summon_marks=state.summon_marks.at[jnp.maximum(state.enemy, 0)].set(
            state.summon_marks[jnp.maximum(state.enemy, 0)] & ~engage)), SUMMON_ENGAGE_REWARD * marked.astype(jnp.float32)

    def rest(self, state, resting):
        """New day: limits/effects expire; living wounded enemies regenerate."""
        maximum, wounds = self.enemy_max_hp(state), state.enemy_wounds
        current = jnp.maximum(maximum - wounds, 0)
        owned = self.territory.owned(state, self.positions)
        city = jnp.maximum(self.enemy_city, 0)
        percent = (jnp.where(owned, OWN_LAND_REGENERATION, FIELD_REGENERATION)
                   + jnp.where(self.enemy_city >= 0, self.city_bonus[state.city_levels[city]] if self.territory.count else 0, 0))
        if self.territory.has_capital_guards:
            percent = percent + jnp.where(self.territory.capital_guards, CAPITAL_REGENERATION, 0)
        heal = jnp.maximum(1, jnp.rint(maximum * percent[:, None] / 100.).astype(jnp.int32))
        if self.full_regeneration is not None:
            heal = jnp.where(self.full_regeneration[:, None], maximum, heal)
        recovering = resting & state.alive[:, None] & self.occupied & (current > 0) & (wounds > 0)
        return state.replace(
            enemy_wounds=jnp.where(recovering, jnp.maximum(wounds - heal, 0), wounds),
            spell_casts=jnp.where(resting, 0, state.spell_casts),
            spell_effects=jnp.where(resting, jnp.uint32(0), state.spell_effects),
            enemy_spell_effects=jnp.where(resting, jnp.uint32(0), state.enemy_spell_effects),
            summon_marks=state.summon_marks & ~resting,
            last_cast_spell=jnp.int32(-1), last_spell_target=jnp.int32(-1), last_spell_amount=jnp.int32(0))

    # Battles -------------------------------------------------------------
    def modifiers(self, bits):
        active = (bits & self.bits) != 0
        additive = jnp.sum(jnp.where(active[:, None], self.mods, 0.), axis=0)
        product = jnp.prod(jnp.where(active[:, None], self.mods, 1.), axis=0)
        return jnp.array([additive[0], product[1], product[2], additive[3], product[4]])

    def begin_battle(self, state, hp):
        """Fix per-slot modifiers once: living hero units (never a summon) and the stack's own units."""
        neutral = jnp.array(NEUTRAL, jnp.float32)
        buffed = jnp.concatenate(((hp[:6] > 0) & ~state.summon_battle, jnp.zeros(6, bool)))
        debuffed = (jnp.arange(12) >= 6) & (state.unit_ids != 0)
        blue = self.modifiers(state.spell_effects)
        red = self.modifiers(state.enemy_spell_effects[jnp.maximum(state.enemy, 0)])
        mods = jnp.where(buffed[:, None], blue, jnp.where(debuffed[:, None], red, neutral))
        wards = jnp.bitwise_or.reduce(jnp.where((state.spell_effects & self.bits) != 0, self.ward_bits, jnp.uint32(0)))
        if self.has_health:
            hp = hp + jnp.where(buffed, mods[:, 0], 0).astype(jnp.int32)
        return state.replace(spell_mods=mods, spell_wards=jnp.where(buffed, wards, jnp.uint32(0))), hp

    def finish_battle(self, before, after, hp, victory, back):
        """Persist enemy wounds; return the stashed roster after a summon battle.

        Reference _restore_blue_team_after_battle: the health bonus leaves as
        max(0, HP - bonus); a promoted unit keeps its new full HP.
        """
        t = jnp.maximum(before.enemy, 0)
        maximum = self.progression.stats(after.unit_ids[6:], after.unit_levels[6:])[:, HP].astype(jnp.int32)
        wounds = jnp.where(victory | ~self.occupied[t], 0, jnp.maximum(maximum - hp[6:], 0))
        summon = before.summon_battle
        restore = back & summon
        party = before.spell_party
        def restored(values, row):
            return values.at[:6].set(jnp.where(restore, row, values[:6]))
        if self.has_health:
            promoted = ((after.last_promoted >> jnp.arange(12, dtype=jnp.uint32)) & 1) != 0
            bonus = jnp.where(back & ~promoted, before.spell_mods[:, 0], 0.).astype(jnp.int32)
            after = after.replace(hp=jnp.maximum(after.hp - bonus, 0))
        return after.replace(
            enemy_wounds=after.enemy_wounds.at[t].set(jnp.where(back, wounds, after.enemy_wounds[t])),
            enemy_spell_effects=after.enemy_spell_effects.at[t].set(jnp.where(victory, jnp.uint32(0), after.enemy_spell_effects[t])),
            summon_marks=after.summon_marks.at[t].set(after.summon_marks[t] & ~(victory & summon)),
            unit_ids=restored(after.unit_ids, party[0]), unit_levels=restored(after.unit_levels, party[1]),
            unit_xp=restored(after.unit_xp, party[2]), hp=restored(after.hp, party[3]),
            last_xp=jnp.where(restore & (jnp.arange(12) < 6), 0, after.last_xp),
            last_promoted=jnp.where(restore, after.last_promoted & jnp.uint32(0xFC0), after.last_promoted),
            summon_battle=summon & ~back,
            spell_mods=jnp.where(back, jnp.array(NEUTRAL, jnp.float32), after.spell_mods),
            spell_wards=jnp.where(back, jnp.uint32(0), after.spell_wards))

    def stats(self, state, values, slot=None):
        """Reference layer order: potions, then map spells, then equipment."""
        mods = state.spell_mods if slot is None else state.spell_mods[slot]
        return jnp.stack((values[..., 0] + mods[..., 0], jnp.rint(values[..., 1] * mods[..., 1]),
                          jnp.minimum(100., jnp.rint(values[..., 2] * mods[..., 2])),
                          jnp.maximum(0., values[..., 3] + mods[..., 3]),
                          jnp.rint(values[..., 4] * mods[..., 4])), axis=-1)

    def secondary(self, state, values, slot):
        mods = state.spell_mods[slot]
        return jnp.stack((jnp.rint(values[..., 0] * mods[..., 1]),
                          jnp.minimum(100., jnp.rint(values[..., 1] * mods[..., 2]))), axis=-1)

    # Terrain/observation -------------------------------------------------
    def terrain(self, state):
        return (state.spell_effects & self.forest_bits) != 0, (state.spell_effects & self.water_bits) != 0

    def observation(self, state):
        any_target, field_target, distance = self.targets(state)
        scale = self.size - 1
        return jnp.concatenate((
            state.spell_casts / self.limits,
            ((state.spell_effects & self.bits) != 0).astype(jnp.float32),
            jnp.sum(state.enemy_wounds, axis=1) / self.base_totals,
            ((state.enemy_spell_effects != 0) + 2. * state.summon_marks) / 3.,
            jnp.array([state.summon_battle, any_target >= 0, field_target >= 0,
                       jnp.where(any_target >= 0, distance[jnp.maximum(any_target, 0)] / scale, 1.),
                       jnp.where(field_target >= 0, distance[jnp.maximum(field_target, 0)] / scale, 1.)], jnp.float32)))
