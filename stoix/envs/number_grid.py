"""Incremental archer/warrior/mage battles, simulated entirely in JAX.

A step is a map move or ONE unit action. Enemy turns and completing a retreat
use the sole legal CONTINUE action. Saved maps can still use numeric rules.
"""
import dataclasses
import json
import math
from pathlib import Path
import jax
import jax.numpy as jnp
from flax import struct
from stoix.envs.number_grid_combat import (
    HP, DAMAGE, ACCURACY, ARMOR, INITIATIVE, MELEE, AREA, HEALER, build_combat_tables, accuracy_hits,
)
from stoix.envs.number_grid_capital import CapitalRules, HEAL_START, REVIVE_START, CAPITAL_ACTIONS
from stoix.envs.number_grid_potions import PotionRules, POTION_START, POTION_ACTIONS
from stoix.envs.number_grid_progression import ProgressionRules
from stoix.envs.number_grid_buildings import (
    BuildingRules, BUILD_START, BUILD_SLOTS, DAILY_GOLD,
)
from stoa import AddActionMaskWrapper, ArraySpace, DictSpace, DiscreteSpace
from stoix.envs.number_grid_legacy import (
    DIRECTIONS, NumberGrid as NumericNumberGrid, NumberGridState,
)

MAP = json.loads((Path(__file__).resolve().parents[2] / 'number_grid_map.json').read_text())
SHOOT, DEFEND, WAIT, RETREAT, CONTINUE = 8, 14, 15, 16, 17
REST = BUILD_START + BUILD_SLOTS
BASE_ACTIONS, ACTIONS = REST + 1, POTION_START + POTION_ACTIONS
MAX_MOVEMENT_POINTS, MOVE_COST = 20, 2
BATTLE_ENTRY_COST = (MAX_MOVEMENT_POINTS + 1) // 2
ACTION_NAMES = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW') + tuple(
    f'shoot_{i}' for i in range(6)) + ('defend', 'wait', 'retreat', 'continue') + tuple(
        f'build_{i}' for i in range(BUILD_SLOTS)) + ('rest',) + tuple(
            f'heal_{i}' for i in range(6)) + tuple(f'revive_{i}' for i in range(6)) + tuple(
                f'potion_{kind}_{slot}' for kind in range(4) for slot in range(6))
MOVE, ENGAGE, HIT, MISS, GUARD, DELAY, FLEE, ESCAPE, VICTORY, DEFEAT, WITHDRAW, LIMIT = range(12)
BUILD, RESTED, IMMUNE, WARD, HEAL = 12, 13, 14, 15, 16


@struct.dataclass
class BattleState(NumberGridState):
    potions: jax.Array  # shared party inventory: heal 50/100/200, revive
    last_potion: jax.Array
    recovery_balance: jax.Array  # victory-funded HP/kill allowances; debt is retained
    last_service_cost: jax.Array
    gold: jax.Array
    map_steps: jax.Array  # successful map movements, excluding combat/invalid actions
    movement_points: jax.Array
    day: jax.Array  # starts at 1; only REST begins the next strategic turn
    buildings: jax.Array  # uint32 bit set: one bit per construction action
    blocked_buildings: jax.Array
    built_today: jax.Array
    last_building: jax.Array
    battle_key: jax.Array
    in_battle: jax.Array
    enemy: jax.Array
    origin: jax.Array
    hp: jax.Array
    wards_used: jax.Array  # source bits consumed per unit during this battle
    last_immune: jax.Array  # bit set of unit slots blocked by immunity
    last_ward: jax.Array  # bit set of unit slots blocked by a one-hit ward
    priority: jax.Array
    turn_phase: jax.Array  # 0: normal turn, 1: waiting, 2: finished
    defended: jax.Array
    retreating: jax.Array
    escaped: jax.Array
    actor: jax.Array
    round: jax.Array
    last_event: jax.Array
    last_actor: jax.Array
    last_target: jax.Array  # -1 for area attacks or non-attacks
    last_damage: jax.Array  # HP removed (or restored for HEAL), summed over targets
    unit_ids: jax.Array
    unit_levels: jax.Array
    unit_xp: jax.Array
    enemy_progress: jax.Array
    battle_xp: jax.Array  # killed-unit XP totals by victim side
    last_xp: jax.Array  # actual XP shares before caps/reset
    last_promoted: jax.Array  # promoted slot bits
    battle_steps: jax.Array
    player_turns: jax.Array
    enemy_turns: jax.Array


class NumberGrid(NumericNumberGrid):
    def __init__(self, max_steps=None, map_config=None):
        game_map = MAP if map_config is None else map_config
        super().__init__(max_steps=max_steps, map_config=game_map)
        self.battle_mode = bool(game_map.get('battle_mode', False))
        if not self.battle_mode:
            return
        self.combat_rules_version = game_map.get('combat_rules_version', 1)
        if type(self.combat_rules_version) is not int or self.combat_rules_version not in (1, 2):
            raise ValueError('Unsupported combat_rules_version')
        self.basic_combat = self.combat_rules_version == 2
        self.progression_enabled = bool(game_map.get('unit_progression', False))
        if self.progression_enabled and not self.basic_combat:
            raise ValueError('Unit progression requires basic combat')
        self.capital_enabled = bool(game_map.get('capital_services', False))
        if self.capital_enabled and not self.progression_enabled:
            raise ValueError('Capital services require named units and progression')
        self.potions_enabled = 'initial_potions' in game_map
        if self.potions_enabled and not self.capital_enabled:
            raise ValueError('Map potions require the current capital/progression environment')
        if self.potions_enabled:
            self.potion_rules = PotionRules(game_map['initial_potions'])
        self.num_actions = (ACTIONS if self.potions_enabled else CAPITAL_ACTIONS if self.capital_enabled
                            else BASE_ACTIONS if self.basic_combat else BUILD_START)
        if self.basic_combat:
            self.construction = BuildingRules(game_map)
            penalty = game_map.get("rest_penalty_per_point", .001)
            if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty < 0:
                raise ValueError("rest_penalty_per_point must be finite and nonnegative")
            self.rest_penalty_per_point = float(penalty)
        self.random_size = 36 if self.basic_combat else 13
        self.mask_walls = True
        self.hero_count = int(game_map.get('hero_units', 6))
        self.hero_hp = int(game_map.get('hero_hp', 45))
        self.damage = int(game_map.get('archer_damage', 25))
        self.accuracy = float(game_map.get('archer_accuracy', .8))
        # Missing role fields preserve the rules of saved all-archer maps.
        self.mage_slot = game_map.get('hero_mage_slot', -1)
        self.mage_damage = game_map.get('mage_damage', 20)
        if (type(self.mage_slot) is not int
                or not (-1 <= self.mage_slot < self.hero_count)
                or type(self.mage_damage) is not int or self.mage_damage <= 0):
            raise ValueError('Invalid mage battle configuration')
        self.warrior_slot = game_map.get('hero_warrior_slot', -1)
        self.warrior_hp = game_map.get('warrior_hp', 100)
        self.warrior_damage = game_map.get('warrior_damage', 25)
        self.warrior_accuracy = game_map.get('warrior_accuracy', .8)
        self.warrior_initiative = game_map.get('warrior_initiative', 50)
        if (type(self.warrior_slot) is not int
                or not (-1 <= self.warrior_slot < self.hero_count)
                or (self.warrior_slot >= 0 and self.warrior_slot == self.mage_slot)
                or any(type(value) is not int or value <= 0 for value in
                       (self.warrior_hp, self.warrior_damage, self.warrior_initiative))
                or type(self.warrior_accuracy) not in (int, float)
                or not 0 <= self.warrior_accuracy <= 1):
            raise ValueError('Invalid warrior battle configuration')
        self.max_rounds = int(game_map.get('battle_max_rounds', 75))
        counts, health = game_map['enemy_units'], game_map['enemy_hp']
        if not (1 <= self.hero_count <= 6 and self.hero_hp > 0 and self.damage > 0
                and 0 <= self.accuracy <= 1 and self.max_rounds > 0):
            raise ValueError('Invalid archer battle configuration')
        if len(counts) != self.num_opponents or len(health) != self.num_opponents:
            raise ValueError('Every enemy needs troop count and hit points')
        for count, hp in zip(counts, health):
            if not (type(count) is int and type(hp) is int and 1 <= count <= 6 and hp > 0):
                raise ValueError('Invalid enemy count or HP')
            if not self.basic_combat and not (count <= self.hero_count and hp <= self.hero_hp
                    and (count < self.hero_count or hp < self.hero_hp)):
                raise ValueError('Every enemy must be strictly weaker in count or HP')
        enemy_roles = game_map.get('enemy_warrior_slots', [-1] * self.num_opponents)
        if (not isinstance(enemy_roles, list) or len(enemy_roles) != self.num_opponents
                or any(type(slot) is not int or not -1 <= slot < count
                       for slot, count in zip(enemy_roles, counts))):
            raise ValueError('Enemy warrior slots must identify an existing unit or -1 per squad')
        self.enemy_warrior_slots = jnp.asarray(enemy_roles, jnp.int32)
        self.has_enemy_warriors = any(slot >= 0 for slot in enemy_roles)
        self.enemy_counts = jnp.asarray(counts, jnp.int32)
        self.enemy_health = jnp.asarray(health, jnp.int32)
        self.slots = jnp.arange(12)
        self.hero_full = jnp.where(jnp.arange(6) < self.hero_count, self.hero_hp, 0)
        if self.warrior_slot >= 0:
            self.hero_full = self.hero_full.at[self.warrior_slot].set(self.warrior_hp)
            # Front slots 0..2, rear slots 3..5, ordered top to bottom.
            # First nonempty tier: near front, far front, near rear, far rear.
            enemy_slots = jnp.arange(6)
            near = jnp.abs(enemy_slots % 3 - self.warrior_slot % 3) <= 1
            self.melee_tiers = (enemy_slots // 3) * 2 + (~near).astype(jnp.int32)
            self.initiative_scale = jnp.ones(12).at[self.warrior_slot].set(
                self.warrior_initiative / 60)
        if self.has_enemy_warriors:
            unit_slots = jnp.arange(6)
            near = jnp.abs(unit_slots[None, :] % 3 - unit_slots[:, None] % 3) <= 1
            self.enemy_melee_tiers = (unit_slots[None, :] // 3) * 2 + (~near).astype(jnp.int32)
            warriors = (self.slots[None, :] == self.warrior_slot) | (
                (self.slots[None, :] >= 6)
                & (self.slots[None, :] - 6 == self.enemy_warrior_slots[:, None]))
            self.squad_initiative_scale = jnp.where(warriors, self.warrior_initiative / 60, 1.)
        if self.basic_combat:
            (self.stats_table, self.damage_ids, self.armor_ids,
             self.damage_rolls, self.priority_scale, self.combat_traits, self.combat_info,
             self.has_protections) = build_combat_tables(game_map)
            profiles = [self.combat_info['heroes']+squad for squad in self.combat_info['enemies']]
            self.size_table = jnp.array([[p['size'] if p else 0 for p in squad] for squad in profiles], jnp.int32)
            self.has_healers = any(p and p['role'] == 'healer' for squad in profiles for p in squad)
            unit_slots = jnp.arange(6)
            near = jnp.abs(unit_slots[None, :] % 3 - unit_slots[:, None] % 3) <= 1
            self.all_melee_tiers = (unit_slots[None, :] // 3) * 2 + (~near).astype(jnp.int32)
            self.hero_full = self.stats_table[0, :6, HP].astype(jnp.int32)
        if self.progression_enabled:
            self.progression = ProgressionRules(game_map, self.construction)
            self.combat_info['catalogue'] = self.progression.metadata
            self.has_protections = self.progression.has_protections
            self.priority_scale = 110.
        if self.capital_enabled:
            self.capital = CapitalRules(self.progression, self.construction, game_map['agent_position'])
        self.restored_hp = jnp.concatenate((self.hero_full, jnp.zeros(6, jnp.int32)))
        # User-defined prototype rule: ceil(10% max HP) once per strategic rest.
        self.rest_healing = jnp.concatenate(((self.hero_full + 9) // 10, jnp.zeros(6, jnp.int32)))
        self.observation_version = int(game_map.get('battle_observation_version', 1))
        if self.basic_combat and self.observation_version not in (7, 8, 9, 10, 11, 12):
            raise ValueError('Combat rules version 2 requires observation version 7, 8, 9, 10, 11 or 12')
        if not self.basic_combat and self.observation_version in (4, 5, 6, 7, 8, 9, 10, 11, 12):
            raise ValueError('Observation versions 7–12 require combat rules version 2')
        if (self.warrior_slot >= 0 or self.has_enemy_warriors) and self.observation_version not in (3, 7, 8, 9, 10, 11, 12):
            raise ValueError('Warrior maps require battle_observation_version 3, 7, 8, 9, 10, 11 or 12')
        self.observation_size = (4 + 5 * self.num_opponents + (141 if self.basic_combat else 48) + 6 if self.observation_version in (2, 3, 7, 8, 9, 10, 11, 12)
                                 else self.observation_size + 2 * self.num_opponents + 12 * 8 + 6)

        if self.observation_version in (8, 9, 10, 11, 12):
            self.observation_size += 48
        if self.progression_enabled != (self.observation_version in (9, 10, 11, 12)):
            raise ValueError('Unit progression requires observation version 9, 10, 11 or 12')
        if self.progression_enabled:
            self.observation_size += 60
        if self.observation_version >= 10:
            self.observation_size += 12
        if self.capital_enabled != (self.observation_version in (11, 12)):
            raise ValueError("Capital services require observation version 11 or 12")
        if self.capital_enabled:
            self.observation_size += 17
        if self.potions_enabled != (self.observation_version == 12):
            raise ValueError("Map potions require observation version 12")
        if self.potions_enabled:
            self.observation_size += 4

    def reset(self, rng_key, env_params=None):
        if not self.battle_mode:
            return super().reset(rng_key, env_params)
        visited = jnp.zeros(self.visited_words, jnp.uint32)
        if self.exploration_bonus:
            r, c = self.map_config['agent_position']
            cell = r * self.size + c
            visited = visited.at[cell // 32].set(jnp.uint32(1 << (cell % 32)))
        zero = jnp.int32(0)
        state = BattleState(
            position=jnp.asarray(self.map_config['agent_position'], jnp.int32),
            number=jnp.int32(self.initial_number), alive=jnp.ones(self.num_opponents, bool),
            step_count=zero, done=jnp.bool_(False), won=jnp.bool_(False), lost=jnp.bool_(False),
            visited=visited, battle_key=rng_key, in_battle=jnp.bool_(False), enemy=jnp.int32(-1),
            origin=jnp.asarray(self.map_config['agent_position'], jnp.int32),
            hp=self.restored_hp, wards_used=jnp.zeros(12, jnp.uint32),
            last_immune=jnp.uint32(0), last_ward=jnp.uint32(0), priority=jnp.zeros(12), turn_phase=jnp.zeros(12, jnp.int32),
            defended=jnp.zeros(12, bool), retreating=jnp.zeros(12, bool), escaped=jnp.zeros(12, bool),
            actor=zero, round=zero, last_event=jnp.int32(MOVE), last_actor=jnp.int32(-1),
            last_target=jnp.int32(-1), last_damage=zero, battle_steps=zero,
            player_turns=zero, enemy_turns=zero,
            unit_ids=(self.progression.initial_ids if self.progression_enabled else jnp.zeros(0, jnp.int32)),
            unit_levels=(self.progression.initial_levels if self.progression_enabled else jnp.zeros(0, jnp.int32)),
            unit_xp=(self.progression.initial_xp if self.progression_enabled else jnp.zeros(0, jnp.int32)),
            enemy_progress=(self.progression.initial_enemies if self.progression_enabled else jnp.zeros((0, 6, 2), jnp.int32)),
            battle_xp=jnp.zeros(2, jnp.int32), last_xp=jnp.zeros(12, jnp.int32), last_promoted=jnp.uint32(0),
            potions=(self.potion_rules.initial_counts if self.potions_enabled else jnp.zeros(4, jnp.int32)),
            last_potion=jnp.int32(-1), recovery_balance=jnp.zeros(2, jnp.int32), last_service_cost=zero,
            gold=zero, map_steps=zero,
            movement_points=jnp.int32(MAX_MOVEMENT_POINTS), day=jnp.int32(1),
            buildings=jnp.uint32(0),
            blocked_buildings=(self.construction.initial_blocked if self.basic_combat else jnp.uint32(0)),
            built_today=jnp.bool_(False), last_building=jnp.int32(-1),
        )
        return state, self._timestep(state, jnp.float32(0), first=True)

    def turn_metadata(self):
        return {"movement_points": MAX_MOVEMENT_POINTS, "move_cost": MOVE_COST,
                "battle_entry_cost": BATTLE_ENTRY_COST, "attack_requires_points": 1,
                "income": DAILY_GOLD, "rest_action": REST,
                "regeneration_percent": 10, "regeneration_rounding": "ceil", "automatic_revive": False}

    def map_commands(self, state):
        """UI/export quotes: exact target stack (-1 for a move) and point cost."""
        destinations = state.position[None] + self.directions
        matches = state.alive[None, :] & jnp.all(
            destinations[:, None, :] == self.opponent_positions[None, :, :], axis=-1)
        attacking = jnp.any(matches, axis=1)
        targets = jnp.where(attacking, jnp.argmax(matches, axis=1), -1)
        costs = jnp.where(attacking, jnp.minimum(state.movement_points, BATTLE_ENTRY_COST), MOVE_COST)
        return jnp.stack((targets, costs), axis=1)

    def rest_penalty(self, state):
        return state.movement_points * jnp.float32(self.rest_penalty_per_point)

    def unit_stats(self, state):
        """Five effective characteristics per slot; empty/world enemy slots are zero."""
        values = self._combat_stats(state)
        return jnp.where(((self.slots < 6) | state.in_battle)[:, None], values, 0.)

    def _combat_stats(self, state):
        if self.progression_enabled:
            return self.progression.stats(state.unit_ids, state.unit_levels)
        return self.stats_table[jnp.maximum(state.enemy, 0)]

    def _combat_traits(self, state):
        if self.progression_enabled:
            return self.progression.traits[state.unit_ids]
        return self.combat_traits[jnp.maximum(state.enemy, 0)]

    def unit_experience(self, state):
        if not self.progression_enabled:
            return jnp.zeros((12, 4), jnp.int32)
        values = self.progression.experience(state.unit_ids, state.unit_levels, state.unit_xp)
        return jnp.where(((self.slots < 6) | state.in_battle)[:, None], values, 0)

    def unit_traits(self, state):
        traits = self._combat_traits(state)
        traits = traits.at[:, 3].set(traits[:, 3] & ~state.wards_used)
        return jnp.where(((self.slots < 6) | state.in_battle)[:, None], traits, 0)

    def unit_sizes(self, state):
        sizes = (self.progression.sizes[state.unit_ids] if self.progression_enabled
                 else self.size_table[jnp.maximum(state.enemy, 0)])
        return jnp.where((self.slots < 6) | state.in_battle, sizes, 0)

    def _actor_is_healer(self, state):
        return self._combat_traits(state)[state.actor, 0] == HEALER

    def _actor_is_melee(self, state):
        return self._combat_traits(state)[state.actor, 0] == MELEE

    def max_hp(self, state):
        if self.basic_combat:
            return self.unit_stats(state)[:, HP].astype(jnp.int32)
        enemy = jnp.maximum(state.enemy, 0)
        opponents = jnp.where(jnp.arange(6) < self.enemy_counts[enemy], self.enemy_health[enemy], 0)
        return jnp.concatenate((self.hero_full, jnp.where(state.in_battle, opponents, 0)))

    def observation(self, state):
        if not self.battle_mode:
            return super().observation(state)
        context = jnp.asarray([
            state.in_battle, (state.enemy + 1) / self.num_opponents,
            state.round / self.max_rounds, state.actor / 11,
            state.origin[0] / (self.size - 1), state.origin[1] / (self.size - 1)
        ], jnp.float32)
        if self.basic_combat:
            context = jnp.concatenate((context, self.construction.observation(state), jnp.asarray(
                [state.gold / 1000., state.movement_points / MAX_MOVEMENT_POINTS], jnp.float32)))
        if self.observation_version in (2, 3, 7, 8, 9, 10, 11, 12):
            # No duplicate max-HP arrays or obsolete numeric battle strengths.
            # Signed queue priority contains both order and waiting/acted status.
            world = jnp.stack((self.opponent_positions[:,0] / (self.size-1),
                               self.opponent_positions[:,1] / (self.size-1), state.alive,
                               self.enemy_counts / 6, self.enemy_health / self.hero_hp),axis=1)
            active = (state.hp > 0) & ~state.escaped
            queue = jnp.where(active & (state.turn_phase == 0), state.priority,
                             jnp.where(active & (state.turn_phase == 1), -state.priority, 0.))
            hp_scale = (jnp.maximum(self.max_hp(state), 1) if self.observation_version in (3, 7, 8, 9, 10, 11, 12)
                        else self.hero_hp)
            if self.basic_combat:
                queue /= self.priority_scale
            units = jnp.stack((state.hp / hp_scale, queue, state.defended,
                               jnp.where(state.escaped, 1., state.retreating * .5)),axis=1)
            if self.basic_combat:
                units = jnp.concatenate((units, self.unit_stats(state) /
                                         jnp.asarray([100., 300., 100., 100., 100.])), axis=1)
            if self.potions_enabled:
                context = jnp.concatenate((self.potion_rules.observation(state), context))
            if self.capital_enabled:
                context = jnp.concatenate((self.capital.observation(state, self.size), context))
            if self.progression_enabled:
                xp = self.unit_experience(state).astype(jnp.float32)
                ids = jnp.where((self.slots < 6) | state.in_battle, state.unit_ids, 0)
                growth = jnp.stack((ids / (len(self.progression.rows)-1), xp[:, 0]/100.,
                                   xp[:, 1]/1000., xp[:, 2]/10000., xp[:, 3]/jnp.maximum(xp[:, 2], 1)), axis=1)
                context = jnp.concatenate((growth.reshape(-1), context))
            if self.observation_version in (8, 9, 10, 11, 12):
                # Compact exact source/bitset encoding; keeps the GPU policy input small.
                context = jnp.concatenate((self.unit_traits(state).reshape(-1) /
                                           jnp.tile(jnp.array([4. if self.observation_version >= 10 else 3., 9., 511., 511.]), 12), context))
            if self.observation_version >= 10:
                context = jnp.concatenate((context, self.unit_sizes(state) / 2.))
            return jnp.concatenate((state.position / (self.size-1),
                                    jnp.asarray([state.number / self.number_scale,
                                                 state.step_count / self.max_steps],jnp.float32),
                                    world.reshape(-1), units.reshape(-1), context))
        base = super().observation(state)
        units = jnp.stack((state.hp / self.hero_hp, self.max_hp(state) / self.hero_hp,
                           state.priority, state.turn_phase / 2, state.defended,
                           state.retreating, state.escaped,
                           (self.slots == state.actor) & state.in_battle), axis=1)
        return jnp.concatenate((base, self.enemy_counts / 6, self.enemy_health / self.hero_hp,
                                units.reshape(-1), context))

    def _melee_targets(self, state, enemy_side=False):
        """Same reachability for both sides; pending retreat still occupies a slot."""
        if self.basic_combat:
            side = 0 if enemy_side else 6
            own = 6 if enemy_side else 0
            active = (state.hp[side:side+6] > 0) & ~state.escaped[side:side+6]
            tiers = self.all_melee_tiers[state.actor % 6]
            blocked = (state.actor % 6 >= 3) & jnp.any(
                (state.hp[own:own+3] > 0) & ~state.escaped[own:own+3])
        elif enemy_side:
            active = (state.hp[:6] > 0) & ~state.escaped[:6]
            tiers = self.enemy_melee_tiers[jnp.clip(state.actor - 6, 0, 5)]
            blocked = (state.actor >= 9) & jnp.any((state.hp[6:9] > 0) & ~state.escaped[6:9])
        else:
            active = (state.hp[6:] > 0) & ~state.escaped[6:]
            tiers = self.melee_tiers
            blocked = (jnp.any((state.hp[:3] > 0) & ~state.escaped[:3])
                       if self.warrior_slot >= 3 else False)
        first_tier = jnp.min(jnp.where(active, tiers, 4))
        return active & (tiers == first_tier) & ~jnp.bool_(blocked)

    def _enemy_is_warrior(self, state):
        return (state.actor >= 6) & (state.actor - 6 == self.enemy_warrior_slots[state.enemy])

    def _round_priority(self, random_values, enemy=0, state=None):
        if self.progression_enabled and state is not None:
            return self._combat_stats(state)[:, INITIATIVE] + random_values[:12] * 10
        if self.basic_combat:
            # BAT_INIT=10: discrete bonus 0..9 plus a fractional random tie key.
            # The fractional part only orders ties; it is not a displayed stat.
            return self.stats_table[enemy, :, INITIATIVE] + random_values[:12] * 10
        # Preserve existing random initiative (1..2), scaled by base initiative.
        # Archers/mage stay at 60; a warrior at 50 is slower, but not always last.
        priority = random_values[1:] + 1
        if self.has_enemy_warriors:
            priority *= self.squad_initiative_scale[enemy]
        elif self.warrior_slot >= 0:
            priority *= self.initiative_scale
        return priority

    def action_mask(self, state):
        if not self.battle_mode:
            return super().action_mask(state)
        destination = state.position[None] + self.directions
        occupied = jnp.any(state.alive[None, :] & jnp.all(
            destination[:, None, :] == self.opponent_positions[None, :, :], axis=-1), axis=1)
        movement = super().action_mask(state) & ~state.in_battle
        if self.basic_combat:
            movement &= state.movement_points >= jnp.where(occupied, 1, MOVE_COST)
        controlled = state.in_battle & (state.actor < 6) & ~state.retreating[state.actor]
        targets = (state.hp[6:] > 0) & ~state.escaped[6:] & controlled
        if self.basic_combat:
            if self.has_healers:
                targets = jnp.where(self._actor_is_healer(state),
                    (state.hp[:6] > 0) & ~state.escaped[:6] & controlled, targets)
            targets &= ~self._actor_is_melee(state) | self._melee_targets(state)
        elif self.warrior_slot >= 0:
            targets &= (state.actor != self.warrior_slot) | self._melee_targets(state)
        mask = jnp.concatenate((movement, targets, jnp.asarray([
            controlled, controlled & (state.turn_phase[state.actor] == 0),
            controlled, state.in_battle & ~controlled])))
        if self.basic_combat:
            mask = jnp.concatenate((mask, self.construction.available(state),
                                    jnp.asarray([~state.in_battle & ~state.done])))
        if self.capital_enabled:
            heal, revive = self.capital.available(state, self.max_hp(state))
            mask = jnp.concatenate((mask, heal, revive))
        if self.potions_enabled:
            mask = jnp.concatenate((mask, self.potion_rules.available(state, self.max_hp(state)).reshape(-1)))
        return jnp.where(state.done, jnp.arange(self.num_actions) == CONTINUE, mask)

    def _timestep(self, state, reward, first=False):
        ts = super()._timestep(state, reward, first)
        if self.battle_mode:
            ts = ts.replace(extras={**ts.extras, 'battle_transition': jnp.bool_(False),
                                    'player_battle_transition': jnp.bool_(False),
                                    'enemy_battle_transition': jnp.bool_(False),
                                    'battle_victory': jnp.bool_(False),
                                    'building_constructed': jnp.bool_(False),
                                    'turn_ended': jnp.bool_(False),
                                    'rest_penalty': jnp.float32(0)})
        return ts

    def _begin_battle(self, state, key=None, random_values=None):
        if key is None:
            key, random_key = jax.random.split(state.battle_key)
            random_values = jax.random.uniform(random_key, (self.random_size,))
        if self.progression_enabled:
            enemy_progress = state.enemy_progress[state.enemy]
            state = state.replace(unit_ids=state.unit_ids.at[6:].set(self.progression.enemy_ids[state.enemy]),
                                  unit_levels=state.unit_levels.at[6:].set(enemy_progress[:, 0]),
                                  unit_xp=state.unit_xp.at[6:].set(enemy_progress[:, 1]),
                                  battle_xp=jnp.zeros(2, jnp.int32),
                                  last_xp=jnp.zeros(12, jnp.int32), last_promoted=jnp.uint32(0))
        priority = self._round_priority(random_values, state.enemy, state)
        enemy_hp = jnp.where(jnp.arange(6) < self.enemy_counts[state.enemy],
                             self.enemy_health[state.enemy], 0)
        if self.basic_combat:
            enemy_hp = self._combat_stats(state)[6:, HP].astype(jnp.int32)
        hp = jnp.concatenate((state.hp[:6] if self.basic_combat else self.hero_full, enemy_hp))
        actor = jnp.argmax(jnp.where(hp > 0, priority, -100)).astype(jnp.int32)
        return state.replace(
            battle_key=key, in_battle=jnp.bool_(True), hp=hp, priority=priority,
            wards_used=jnp.zeros(12, jnp.uint32), last_immune=jnp.uint32(0), last_ward=jnp.uint32(0),
            turn_phase=jnp.zeros(12, jnp.int32), defended=jnp.zeros(12, bool),
            retreating=jnp.zeros(12, bool), escaped=jnp.zeros(12, bool),
            actor=actor, round=jnp.int32(1), last_event=jnp.int32(ENGAGE),
            last_actor=jnp.int32(-1), last_target=jnp.int32(-1), last_damage=jnp.int32(0),
        )

    def _world_step(self, state, action, key, random_values):
        destination = state.position + self.directions[jnp.clip(action, 0, 7)]
        # A directional command aimed at an occupied tile attacks that exact stack.
        # The attacker stays on its own tile, including after victory/retreat.
        targeted = state.alive & jnp.all(self.opponent_positions == destination, axis=1)
        engage = (action < 8) & jnp.any(targeted)
        moved = (action < 8) & ~engage  # step() checks bounds and movement points.
        position = jnp.where(moved, destination, state.position)
        enemy = jnp.argmax(targeted).astype(jnp.int32)
        map_steps = state.map_steps + moved.astype(jnp.int32)
        resting = (action == REST) if self.basic_combat else jnp.bool_(False)
        gold = state.gold + jnp.where(resting, DAILY_GOLD, 0)
        next_state = state.replace(position=position, origin=state.position,
                                   gold=gold, map_steps=map_steps, last_service_cost=jnp.int32(0),
                                   last_xp=jnp.zeros(12, jnp.int32), last_promoted=jnp.uint32(0),
                                   last_immune=jnp.uint32(0), last_ward=jnp.uint32(0),
                                   enemy=jnp.where(engage, enemy, state.enemy),
                                   last_event=jnp.int32(MOVE), last_actor=jnp.int32(-1),
                                   last_target=jnp.int32(-1), last_damage=jnp.int32(0))
        bonus = jnp.float32(0)
        if self.basic_combat:
            rest_max = self.max_hp(state) if self.progression_enabled else self.restored_hp
            rest_healing = (rest_max+9)//10 if self.progression_enabled else self.rest_healing
            building = jnp.clip(action - BUILD_START, 0, BUILD_SLOTS - 1)
            building_action = (action >= BUILD_START) & (action < REST)
            next_state = next_state.replace(
                gold=gold-jnp.where(building_action, self.construction.costs[building], 0),
                buildings=state.buildings | jnp.where(building_action, self.construction.bits[building], jnp.uint32(0)),
                blocked_buildings=state.blocked_buildings | jnp.where(
                    building_action, self.construction.blocks[building], jnp.uint32(0)),
                built_today=~resting & (state.built_today | building_action),
                movement_points=jnp.where(resting, MAX_MOVEMENT_POINTS,
                                          jnp.maximum(0, state.movement_points - jnp.where(
                                              engage, BATTLE_ENTRY_COST, moved.astype(jnp.int32) * MOVE_COST))),
                day=state.day + resting.astype(jnp.int32),
                hp=jnp.where(resting & (state.hp > 0),
                             jnp.minimum(state.hp + rest_healing, rest_max), state.hp),
                last_building=jnp.where(building_action, building, -1),
                last_event=jnp.where(resting, RESTED, jnp.where(building_action, BUILD, MOVE)),
            )
            bonus += jnp.where(building_action, jnp.float32(self.construction.reward), 0.)
            bonus -= jnp.where(resting, self.rest_penalty(state), 0.)
        if self.capital_enabled:
            next_state, recovery_bonus = self.capital.apply(next_state, action, rest_max)
            bonus += recovery_bonus
        if self.potions_enabled:
            next_state = self.potion_rules.apply(next_state, action, rest_max)
        if self.exploration_bonus:
            cell = position[0] * self.size + position[1]
            word, bit = cell // 32, jnp.left_shift(jnp.uint32(1), (cell % 32).astype(jnp.uint32))
            first = (state.visited[word] & bit) == 0
            bonus += jnp.where(first & moved, jnp.float32(self.exploration_bonus), 0.)
            next_state = next_state.replace(visited=state.visited.at[word].set(state.visited[word] | jnp.where(moved, bit, jnp.uint32(0))))
        next_state = jax.lax.cond(engage, lambda s: self._begin_battle(s, key, random_values), lambda s: s, next_state)
        return next_state, bonus

    def _enemy_action(self, state, random_values):
        valid = (state.hp[:6] > 0) & ~state.escaped[:6]
        base_damage = self.damage
        if self.basic_combat:
            valid &= ~self._actor_is_melee(state) | self._melee_targets(state, enemy_side=True)
        elif self.has_enemy_warriors:
            warrior = self._enemy_is_warrior(state)
            valid &= ~warrior | self._melee_targets(state, enemy_side=True)
            base_damage = jnp.where(warrior, self.warrior_damage, base_damage)
        damage = jnp.where(state.defended[:6], (base_damage + 1) // 2, base_damage)
        if self.basic_combat:
            # Use minimum damage for a guaranteed kill, without peeking at hit RNG.
            if self.progression_enabled:
                damage = self.progression.damage(state, state.actor, jnp.arange(6), state.defended[:6], 0)
            else:
                damage = self.damage_rolls[self.damage_ids[state.enemy, state.actor],
                                           self.armor_ids[state.enemy, :6],
                                           state.defended[:6].astype(jnp.int32), 0]
        kill = valid & (state.hp[:6] <= damage)
        candidates = valid & jnp.where(jnp.any(kill), kill, True)
        # HP dominates the random tie breaker, including among killable targets.
        score = jnp.where(candidates, state.hp[:6] + random_values[:6] * .5, 1e9)
        attack = SHOOT + jnp.argmin(score).astype(jnp.int32)
        action = jnp.where(jnp.any(valid), attack, DEFEND) if self.basic_combat or self.has_enemy_warriors else attack
        if self.basic_combat and self.has_healers:
            # Reference: lowest absolute HP among wounded living allies, self included.
            wounded = (state.hp[6:] > 0) & ~state.escaped[6:] & (state.hp[6:] < self._combat_stats(state)[6:, HP])
            score = jnp.where(wounded, state.hp[6:] + random_values[:6] * .5, 1e9)
            heal = jnp.where(jnp.any(wounded), SHOOT + jnp.argmin(score).astype(jnp.int32), DEFEND)
            action = jnp.where(self._actor_is_healer(state), heal, action)
        return action

    def _battle_step(self, state, action, key, random_values):
        actor = state.actor
        escaping = state.retreating[actor]
        action = jnp.where(actor >= 6, self._enemy_action(
            state, random_values[30:36] if self.basic_combat else random_values[1:]), action)
        action = jnp.where(escaping, CONTINUE, action)
        attack = (action >= SHOOT) & (action < DEFEND)
        healer = self._actor_is_healer(state) if self.basic_combat and self.has_healers else jnp.bool_(False)
        target = jnp.clip(action - SHOOT, 0, 5) + jnp.where((actor < 6) ^ healer, 6, 0)
        accuracy, single_damage = self.accuracy, self.damage
        if self.warrior_slot >= 0 or self.has_enemy_warriors:
            warrior = actor == self.warrior_slot
            if self.has_enemy_warriors:
                warrior |= self._enemy_is_warrior(state)
            accuracy = jnp.where(warrior, self.warrior_accuracy, accuracy)
            single_damage = jnp.where(warrior, self.warrior_damage, single_damage)
        hit = attack & (random_values[0] < accuracy)
        area_attack = jnp.bool_(False)
        wards_used = state.wards_used
        immune_slots, ward_slots = jnp.uint32(0), jnp.uint32(0)
        if self.basic_combat:
            stats = self._combat_stats(state)
            target_slots = jnp.arange(6) + jnp.where(actor < 6, 6, 0)
            traits = self._combat_traits(state)
            area_attack = traits[actor, 0] == AREA
            targets = ((area_attack | (target_slots == target))
                       & (state.hp[target_slots] > 0) & ~state.escaped[target_slots])
            hits = accuracy_hits(stats[actor, ACCURACY], random_values[12:18], random_values[18:24])
            bonuses = jnp.minimum((random_values[24:30] * 6).astype(jnp.int32), 5)
            if self.progression_enabled:
                damage = self.progression.damage(state, actor, target_slots, state.defended[target_slots], bonuses)
            else:
                damage = self.damage_rolls[self.damage_ids[state.enemy, actor],
                                           self.armor_ids[state.enemy, target_slots],
                                           state.defended[target_slots].astype(jnp.int32), bonuses]
            connected = attack & ~healer & targets & hits
            effective = connected
            if self.has_protections:
                source_bit = jnp.left_shift(jnp.uint32(1), jnp.maximum(traits[actor, 1], 1)-1)
                immune = connected & ((traits[target_slots, 2] & source_bit) != 0)
                warded = connected & ~immune & ((traits[target_slots, 3] &
                                                  ~wards_used[target_slots] & source_bit) != 0)
                wards_used = wards_used.at[target_slots].set(
                    wards_used[target_slots] | jnp.where(warded, source_bit, jnp.uint32(0)))
                slot_bits = jnp.left_shift(jnp.uint32(1), target_slots.astype(jnp.uint32))
                immune_slots = jnp.sum(jnp.where(immune, slot_bits, jnp.uint32(0)))
                ward_slots = jnp.sum(jnp.where(warded, slot_bits, jnp.uint32(0)))
                effective &= ~immune & ~warded
            removed = jnp.where(effective, jnp.minimum(damage, state.hp[target_slots]), 0)
            zeros = jnp.zeros(6, jnp.int32)
            hp = state.hp - jnp.where(actor < 6, jnp.concatenate((zeros, removed)),
                                     jnp.concatenate((removed, zeros)))
            applied = jnp.sum(removed)
            hit = attack & jnp.any(targets & hits)
            if self.has_healers:
                # Healing has no hit roll, armor/defence reduction, or ward consumption.
                healed = jnp.where(attack & healer & (state.hp[target] > 0) & ~state.escaped[target],
                    jnp.minimum(stats[actor, DAMAGE], jnp.maximum(stats[target, HP]-state.hp[target], 0)), 0).astype(jnp.int32)
                hp = hp.at[target].add(healed)
                applied = jnp.where(healer, healed, applied)
                hit |= attack & healer
        elif self.mage_slot >= 0:
            # A fixed hero role is inferable from the actor/HP slots already in
            # observation v2. One hit roll covers the entire spell, without
            # extra RNG, loops, or host operations in the vmapped learner.
            area_attack = actor == self.mage_slot
            targets = jnp.where(area_attack, self.slots >= 6, self.slots == target)
            targets &= (state.hp > 0) & ~state.escaped
            base_damage = jnp.where(area_attack, self.mage_damage, single_damage)
            damage = jnp.where(state.defended, (base_damage + 1) // 2, base_damage)
            removed = jnp.where(hit & targets, jnp.minimum(damage, state.hp), 0)
            hp = state.hp - removed
            applied = jnp.sum(removed)
        else:
            damage = jnp.where(state.defended[target], (single_damage + 1) // 2, single_damage)
            applied = jnp.where(hit, jnp.minimum(damage, state.hp[target]), 0)
            hp = state.hp.at[target].add(-applied)
        phases = state.turn_phase.at[actor].set(jnp.where(action == WAIT, 1, 2))
        defended = state.defended.at[actor].set(action == DEFEND)
        retreating = state.retreating.at[actor].set(action == RETREAT)
        escaped = state.escaped.at[actor].set(escaping)
        active = (hp > 0) & ~escaped
        normal, waiting = active & (phases == 0), active & (phases == 1)
        new_round = ~jnp.any(normal | waiting)
        phases = jnp.where(new_round, jnp.zeros(12, jnp.int32), phases)
        priority = jnp.where(new_round, self._round_priority(random_values, state.enemy, state), state.priority)
        round_number = state.round + new_round.astype(jnp.int32)
        scores = jnp.where(active & (phases == 0), priority,
                            jnp.where(active & (phases == 1), -priority, -1e9 if self.basic_combat else -100))
        next_actor = jnp.argmax(scores).astype(jnp.int32)
        defended = defended.at[next_actor].set(False)
        lost, victory = ~jnp.any(hp[:6] > 0), ~jnp.any(hp[6:] > 0)
        withdrawal = ~jnp.any(active[:6]) & ~lost
        timeout = new_round & (round_number > self.max_rounds) & ~victory & ~lost & ~withdrawal
        alive = state.alive.at[state.enemy].set(~victory)
        won = ~jnp.any(alive) & ~lost
        back = victory | withdrawal
        event = jnp.where(attack, jnp.where(hit, HIT, MISS),
                            jnp.where(action == DEFEND, GUARD,
                            jnp.where(action == WAIT, DELAY,
                            jnp.where(action == RETREAT, FLEE, ESCAPE))))
        event = jnp.where(attack & healer, HEAL, event)
        event = jnp.where(attack & (applied == 0) & (immune_slots != 0), IMMUNE, event)
        event = jnp.where(attack & (applied == 0) & (ward_slots != 0), WARD, event)
        event = jnp.where(victory, VICTORY, jnp.where(lost, DEFEAT,
                            jnp.where(withdrawal, WITHDRAW, jnp.where(timeout, LIMIT, event))))
        progress = {}
        if self.progression_enabled:
            killed = (state.hp > 0) & (hp == 0)
            kill_xp = self.progression.experience(state.unit_ids, state.unit_levels, state.unit_xp)[:, 1]
            bank = state.battle_xp + jnp.sum(jnp.where(killed, kill_xp, 0).reshape(2, 6), axis=1)
            ids, levels, xp, enemies, hp, gains, promoted = self.progression.finish(
                state, hp, escaped, victory, lost, withdrawal, bank)
            progress = dict(unit_ids=ids, unit_levels=levels, unit_xp=xp,
                            enemy_progress=enemies, battle_xp=bank,
                            last_xp=gains, last_promoted=promoted)
        if self.capital_enabled:
            # Enemies start each fight at full HP, cannot escape or resurrect,
            # and victory kills the whole squad. Thus the reference per-victim
            # damage cap equals their initial max HP, regardless of healer turns.
            maximum = self._combat_stats(state)[6:, HP].astype(jnp.int32)
            credit = jnp.array([jnp.sum(maximum), jnp.sum(maximum > 0)], jnp.int32)
            progress["recovery_balance"] = state.recovery_balance + jnp.where(victory & ~lost, credit, 0)
        recovered_hp = self.restored_hp
        if self.basic_combat:
            # Both survivors and fallen units retain their HP; revival is paid.
            recovered_hp = jnp.where(self.restored_hp > 0, hp, 0)
        next_state = state.replace(
            battle_key=key, hp=jnp.where(back, recovered_hp, hp), **progress,
            wards_used=jnp.where(back, jnp.uint32(0), wards_used),
            last_immune=immune_slots, last_ward=ward_slots,
            in_battle=~back, position=jnp.where(withdrawal, state.origin, state.position),
            alive=alive, number=state.number + victory.astype(jnp.int32), won=won, lost=lost,
            done=won | lost | timeout, priority=jnp.where(back, 0., priority),
            turn_phase=jnp.where(back, 0, phases), defended=defended & ~back,
            retreating=retreating & ~back, escaped=escaped & ~back,
            actor=jnp.where(back, 0, next_actor), round=jnp.where(back, 0, round_number),
            last_event=event.astype(jnp.int32), last_actor=actor,
            last_target=jnp.where(attack & ~area_attack, target, -1), last_damage=applied,
            battle_steps=state.battle_steps + 1,
            player_turns=state.player_turns + ((actor < 6) & ~escaping).astype(jnp.int32),
            enemy_turns=state.enemy_turns + ((actor >= 6) & ~escaping).astype(jnp.int32),
        )
        reward = victory.astype(jnp.float32) + 3 * won - lost.astype(jnp.float32)
        return next_state, reward

    def step(self, state, action, env_params=None):
        if not self.battle_mode:
            return super().step(state, action, env_params)
        # Validate only the selected move/target here. The full action mask is
        # generated once for the next observation, not twice per transition.
        destination = state.position + self.directions[jnp.clip(action, 0, 7)]
        occupied = jnp.any(state.alive & jnp.all(self.opponent_positions == destination, axis=1))
        world_valid = ((action >= 0) & (action < 8)
                       & jnp.all((destination > 0) & (destination < self.size - 1)))
        if self.basic_combat:
            building = jnp.clip(action - BUILD_START, 0, BUILD_SLOTS-1)
            world_valid &= state.movement_points >= jnp.where(occupied, 1, MOVE_COST)
            world_valid |= ((action >= BUILD_START) & (action < REST)
                            & self.construction.available(state, building)) | (action == REST)
        if self.capital_enabled:
            slot = jnp.clip(jnp.where(action < REVIVE_START, action-HEAL_START, action-REVIVE_START), 0, 5)
            heal, revive = self.capital.available(state, self.max_hp(state), slot)
            world_valid |= ((action >= HEAL_START) & (action < REVIVE_START) & heal
                            | (action >= REVIVE_START) & (action < CAPITAL_ACTIONS) & revive)
        if self.potions_enabled:
            index = jnp.clip(action-POTION_START, 0, POTION_ACTIONS-1)
            available = self.potion_rules.available(state, self.max_hp(state), index // 6, index % 6)
            world_valid |= (action >= POTION_START) & (action < ACTIONS) & available
        controlled = (state.actor < 6) & ~state.retreating[state.actor]
        target_slot = jnp.clip(action - SHOOT, 0, 5)
        own_target = self._actor_is_healer(state) if self.basic_combat and self.has_healers else jnp.bool_(False)
        target = target_slot + jnp.where(own_target, 0, 6)
        attack_valid = ((action >= SHOOT) & (action < DEFEND)
                        & (state.hp[target] > 0) & ~state.escaped[target])
        if self.basic_combat:
            attack_valid &= ~self._actor_is_melee(state) | self._melee_targets(state)[target_slot]
        elif self.warrior_slot >= 0:
            attack_valid &= ((state.actor != self.warrior_slot)
                             | self._melee_targets(state)[target_slot])
        unit_valid = attack_valid | (action == DEFEND) | (action == RETREAT) | ((action == WAIT) & (state.turn_phase[state.actor] == 0))
        battle_valid = jnp.where(controlled, unit_valid, action == CONTINUE)
        valid = jnp.where(state.in_battle, battle_valid, world_valid) & ~state.done
        # Share one random draw between the vmapped map/battle branches.
        key, random_key = jax.random.split(state.battle_key)
        random_values = jax.random.uniform(random_key, (self.random_size,))
        next_state, reward = jax.lax.cond(state.in_battle, self._battle_step, self._world_step,
                                         state, action, key, random_values)
        steps = state.step_count + 1
        next_state = next_state.replace(step_count=steps, done=next_state.done | (steps >= self.max_steps))
        fallback = state.replace(step_count=jnp.where(state.done, state.step_count, steps),
                                 done=state.done | (steps >= self.max_steps))
        next_state = jax.tree.map(lambda new, old: jnp.where(valid, new, old), next_state, fallback)
        rested = valid & ~state.in_battle & (action == REST)
        # REST has only the unused-point penalty: exhausting movement makes it free.
        reward = jnp.where(valid, reward, 0.) - jnp.where(rested, 0., jnp.float32(self.step_cost))
        ts = self._timestep(next_state, jnp.where(state.done, 0., reward))
        combat = state.in_battle & valid & ~state.done
        ts = ts.replace(extras={**ts.extras, 'battle_transition': combat,
                                'player_battle_transition': combat & (state.actor < 6) & ~state.retreating[state.actor],
                                'enemy_battle_transition': combat & (state.actor >= 6),
                                'battle_victory': combat & (next_state.last_event == VICTORY),
                                'building_constructed': valid & ~state.in_battle & (action >= BUILD_START) & (action < REST),
                                'turn_ended': rested,
                                'rest_penalty': (jnp.where(rested, self.rest_penalty(state), 0.)
                                                 if self.basic_combat else jnp.float32(0))})
        return next_state, ts

    def action_space(self, env_params=None):
        return DiscreteSpace(self.num_actions if self.battle_mode else 8, dtype=jnp.int32)

    def state_space(self, env_params=None):
        if not self.battle_mode:
            return super().state_space(env_params)
        state, _ = self.reset(jax.random.PRNGKey(0))
        return DictSpace({f.name: ArraySpace(getattr(state, f.name).shape, getattr(state, f.name).dtype)
                          for f in dataclasses.fields(state)})


def wrap_wall_action_mask(env):
    return AddActionMaskWrapper(env) if env.mask_walls else env
