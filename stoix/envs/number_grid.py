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
from stoix.envs.number_grid_potions import PotionRules, POTION_START, scenario_potions
from stoix.envs.number_grid_progression import ProgressionRules
from stoix.envs.number_grid_summoning import Summoning, COPY_ALLY, COPY_ACTIONS, advance_linked_queue
from stoix.envs.number_grid_chests import ChestRules
from stoix.envs.number_grid_items import ItemRules
from stoix.envs.number_grid_item_combat import equipment_attack, transform_imp, transform_decay
from stoix.envs.number_grid_wards import ward_tags, grant_wards, expire_wards
from stoix.envs.number_grid_effects import armor_after_shreds, source_protection, advance_poison_queue, advance_periodic_queue, vampiric_heal
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
BASE_ACTIONS, FENRIR = REST + 1, CAPITAL_ACTIONS
ACTIONS = POTION_START + 6*len(scenario_potions(MAP))
MAX_MOVEMENT_POINTS, MOVE_COST = 20, 2
BATTLE_ENTRY_COST = (MAX_MOVEMENT_POINTS + 1) // 2
ACTION_NAMES = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW') + tuple(
    f'shoot_{i}' for i in range(6)) + ('defend', 'wait', 'retreat', 'continue') + tuple(
        f'build_{i}' for i in range(BUILD_SLOTS)) + ('rest',) + tuple(
            f'heal_{i}' for i in range(6)) + tuple(f'revive_{i}' for i in range(6)) + ('transform_fenrir',) + tuple(f'copy_ally_{slot}' for slot in range(6)) + tuple(
                f'potion_{item["key"]}_{slot}' for item in scenario_potions(MAP) for slot in range(6))
MOVE, ENGAGE, HIT, MISS, GUARD, DELAY, FLEE, ESCAPE, VICTORY, DEFEAT, WITHDRAW, LIMIT = range(12)
BUILD, RESTED, IMMUNE, WARD, HEAL = 12, 13, 14, 15, 16
TRANSFORMED = 22
IMP_TRANSFORMED = 23
DECAY_TRANSFORMED = 24
FEARED = 27
POST_HEAL = 28
BUFFED = 29
BATTLE_REVIVED = 30
EXTRA_TURN = 31
SLOWED = 32
PARALYZED = 25
PARALYSIS_SKIP = 26


@struct.dataclass
class BattleState(NumberGridState):
    native_ids: jax.Array
    native_levels: jax.Array
    native_xp: jax.Array
    copied: jax.Array
    copy_stats: jax.Array
    copy_secondary: jax.Array
    copy_traits: jax.Array
    summon_owner: jax.Array
    preparation: jax.Array
    initiative_override: jax.Array  # native or exact temporary form/slow base
    saved_initiative_override: jax.Array
    slow_original: jax.Array  # -1: no Hermit slow; otherwise pre-slow base
    waited: jax.Array  # WAIT is still spent after an Alchemist grants another turn
    bonus_turns: jax.Array
    battle_revived: jax.Array  # one Patriarch revival per recipient per battle
    revival_xp_cutoff: jax.Array  # opposing XP bank at the last revival
    primary_override: jax.Array  # -1 means native/current-form damage
    preweak_damage: jax.Array
    powerup: jax.Array
    powerup_layered: jax.Array  # Lower Damage was applied over the current buff
    saved_primary_override: jax.Array
    saved_preweak_damage: jax.Array
    saved_powerup: jax.Array
    saved_powerup_layered: jax.Array
    healer_wards: jax.Array  # [12,4] caster ownership bits, shared elemental blocks
    ward_native_used: jax.Array  # sticky spent native wards while grants are active
    saved_healer_wards: jax.Array
    saved_ward_native_used: jax.Array
    post_victory: jax.Array  # 0 combat/map; 1 blue healing; 2 red healing
    pending_healers: jax.Array  # one native support action each, in formation order
    fenrir: jax.Array  # temporary Wolf Lord form; permanent identity/XP is retained
    armor_shreds: jax.Array  # successful 15-point Shatters, capped once armour reaches zero
    imp: jax.Array  # Witch/Succub form overlays Fenrir without changing HP or size
    imp_wards: jax.Array  # spent wards of the suspended natural form
    decay_form: jax.Array  # native catalogue ID of a Wight-lowered temporary form
    decay_shreds: jax.Array  # armour effects applied only to the temporary form
    feared: jax.Array  # distinguish magical fear from voluntary retreat for Cure
    enemy_initial_hp: jax.Array
    enemy_damage_credit: jax.Array
    enemy_killed_credit: jax.Array
    weakened: jax.Array  # Tiamat: current-form primary amount x0.68, no stacking
    saved_weakened: jax.Array  # debuff of the form suspended by Witch/Wight
    paralyzed: jax.Array  # finite: skip one activation
    long_paralyzed: jax.Array  # skip and roll 33% recovery each activation
    second_strike: jax.Array  # remaining second attack of the current Demon/Elfarcher
    poison_turns: jax.Array
    poison_damage: jax.Array
    poison_source: jax.Array  # cached poison caster slot, -1 for no active lock
    water_turns: jax.Array
    water_damage: jax.Array
    water_source: jax.Array
    last_water_damage: jax.Array
    burn_turns: jax.Array
    burn_damage: jax.Array
    burn_source: jax.Array
    last_burn_damage: jax.Array
    last_poison_damage: jax.Array
    activation_done: jax.Array  # once-per-round effects, unaffected by WAIT
    potions: jax.Array  # dense inventory of the scenario's obtainable potion types
    potion_doses: jax.Array  # [12,5] permanent doses by stat, retained across growth
    potion_bonus: jax.Array  # [12,7] cached permanent deltas, incl. secondary damage/accuracy
    potion_temporary: jax.Array  # [12,7] cached temporary stat deltas, incl. secondary
    potion_active: jax.Array  # one bit per temporary catalogue item, until rest
    potion_wards: jax.Array
    last_potion: jax.Array
    chest_alive: jax.Array
    last_loot: jax.Array  # potion counts granted by the most recent map transition
    item_inventory: jax.Array  # acquisition order, scenario item IDs; -1 empty
    equipped: jax.Array  # artifact x2, banner, book, boots; inventory retains all items
    equipment_bonus: jax.Array  # cached artifact/banner layer, after permanent/temporary potions
    last_item_loot: jax.Array
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


class NumberGrid(Summoning, NumericNumberGrid):
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
        self.has_copies = False
        self.has_summons = False
        self.has_wolf_lord = False
        self.has_shatterers = False
        self.has_witches = False
        self.has_wights = False
        self.has_centaurs = False
        self.has_double_strike = False
        self.has_poisoners = False
        self.has_paralysis = False
        self.has_secondary_paralysis = False
        self.has_weakening = False
        self.has_fear = False
        self.has_secondary_fear = False
        fear_teams = game_map.get('fear_paralysis_teams',[])
        if not isinstance(fear_teams,list) or any(t not in ('blue','red') for t in fear_teams):
            raise ValueError('fear_paralysis_teams must contain blue/red team names')
        self.fear_protected = jnp.array([('blue' in fear_teams)]*6+[('red' in fear_teams)]*6)
        self.has_leech = False
        self.has_cures = False
        self.has_powerups = False
        self.has_patriarchs = False
        self.has_alchemists = False
        self.has_hermits = False
        self.has_healer_wards = False
        self.has_water = False
        self.has_fire = False
        self.progression_enabled = bool(game_map.get('unit_progression', False))
        if self.progression_enabled and not self.basic_combat:
            raise ValueError('Unit progression requires basic combat')
        self.capital_enabled = bool(game_map.get('capital_services', False))
        if self.capital_enabled and not self.progression_enabled:
            raise ValueError('Capital services require named units and progression')
        self.potions_enabled = 'initial_potions' in game_map
        if self.potions_enabled and not self.capital_enabled:
            raise ValueError('Map potions require the current capital/progression environment')
        self.has_potion_buffs = False
        if self.potions_enabled:
            self.potion_rules = PotionRules(game_map)
            self.has_potion_buffs = self.potion_rules.has_buffs
        self.chests_enabled = 'chests' in game_map
        self.items_enabled = 'initial_items' in game_map or any(
            c.get('items') for c in game_map.get('chests',[]) if isinstance(c,dict))
        if self.items_enabled:
            if not self.potions_enabled:
                raise ValueError('Equipment requires the current map inventory/progression environment')
            self.item_rules = ItemRules(game_map)
        if self.chests_enabled:
            if not self.potions_enabled:
                raise ValueError('Chests require the map potion inventory')
            self.chest_rules = ChestRules(game_map,self.potion_rules,
                                         self.item_rules if self.items_enabled else None)
        self.num_actions = (POTION_START+self.potion_rules.action_count if self.potions_enabled else CAPITAL_ACTIONS if self.capital_enabled
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
            self.shatter_armor_ids = jnp.array(self.combat_info.pop('_shatter_armor_ids'), jnp.int32)
            profiles = [self.combat_info['heroes']+squad for squad in self.combat_info['enemies']]
            self.size_table = jnp.array([[p['size'] if p else 0 for p in squad] for squad in profiles], jnp.int32)
            self.mass_healers = jnp.array([[bool(p and p['unit_type'] in ('Profit','Deva roshi')) for p in squad] for squad in profiles])
            self.mass_cures = jnp.array([[bool(p and p['unit_type'] == 'Profit' and p['name'] in ('Аббатиса','Прорицательница','Matriarch','Prophetess')) for p in squad] for squad in profiles])
            power_types = {'Travnitsa':1.25,'Novice':1.5,'Dwarfdruid':1.75,'Arhidruid':2.}
            self.power_factors = jnp.array([[power_types.get(p['unit_type'],0.) if p else 0. for p in squad] for squad in profiles])
            self.power_cures = jnp.array([[bool(p and p['unit_type'] in ('Dwarfdruid','Arhidruid')) for p in squad] for squad in profiles])
            self.hermits = jnp.array([[bool(p and p['unit_type'] == 'Hermit') for p in squad] for squad in profiles])
            self.has_hermits = any(p and p['unit_type'] == 'Hermit' for squad in profiles for p in squad)
            self.alchemists = jnp.array([[bool(p and p['unit_type'] == 'Alchemist') for p in squad] for squad in profiles])
            self.has_alchemists = any(p and p['unit_type'] == 'Alchemist' for squad in profiles for p in squad)
            self.patriarchs = jnp.array([[bool(p and p['unit_type'] == 'Patriach') for p in squad] for squad in profiles])
            self.has_patriarchs = any(p and p['unit_type'] == 'Patriach' for squad in profiles for p in squad)
            self.has_powerups = any(p and p['unit_type'] in power_types for squad in profiles for p in squad)
            ward_types = {'Sundancer':4,'Sylfid':256,'Deva roshi':270}
            self.healer_ward_elements = jnp.array([[ward_types.get(p['unit_type'],0) if p else 0 for p in squad] for squad in profiles],jnp.uint32)
            self.has_healer_wards = any(p and p['unit_type'] in ward_types for squad in profiles for p in squad)
            self.has_protections |= self.has_healer_wards
            self.has_healers = any(p and p['role'] == 'healer' for squad in profiles for p in squad)
            self.has_cures = any(p and (p['unit_type'] in ('Patriach','Dwarfdruid','Arhidruid') or (p['unit_type'] == 'Profit' and p['name'] in ('Аббатиса','Прорицательница','Matriarch','Prophetess'))) for squad in profiles for p in squad)
            if not self.progression_enabled and any(p and p['unit_type'] == 'Wight' for squad in profiles for p in squad):
                raise ValueError('Wight requires named unit progression for its temporary forms')
            self.aoe_accuracy_falloff = jnp.array([[bool(p and p['aoe_accuracy_falloff']) for p in squad]
                                                  for squad in profiles])
            self.has_aoe_accuracy_falloff = any(p and p['aoe_accuracy_falloff'] for squad in profiles for p in squad)
            self.wolf_lord = jnp.array([[bool(p and p['unit_type'] == 'Wolf Lord') for p in squad] for squad in profiles])
            self.has_wolf_lord = any(p and p['unit_type'] == 'Wolf Lord' for squad in profiles for p in squad)
            self.shatterers = jnp.array([[bool(p and p['unit_type'] in ('Teurg','Aleman')) for p in squad] for squad in profiles])
            self.has_shatterers = any(p and p['unit_type'] in ('Teurg','Aleman') for squad in profiles for p in squad)
            self.witches = jnp.array([[bool(p and p['unit_type'] in ('Witch','Succub')) for p in squad] for squad in profiles])
            self.has_witches = any(p and p['unit_type'] in ('Witch','Succub') for squad in profiles for p in squad)
            self.centaurs = jnp.array([[bool(p and p['unit_type'] == 'Centaur Savage') for p in squad] for squad in profiles])
            self.has_centaurs = any(p and p['unit_type'] == 'Centaur Savage' for squad in profiles for p in squad)
            self.double_strike = jnp.array([[bool(p and p['unit_type'] in ('Demon', 'Elfarcher')) for p in squad] for squad in profiles])
            self.has_double_strike = any(p and p['unit_type'] in ('Demon', 'Elfarcher') for squad in profiles for p in squad)
            self.cached_poisoners = jnp.array([[bool(p and (p['name'] == 'Ниддог' or p['unit_type'] in ('Dregazul','Spider'))) for p in squad] for squad in profiles])
            self.poisoners = jnp.array([[bool(p and ((p['name'] == 'Ниддог' or p['unit_type'] in ('Dregazul','Spider')) or p['unit_type'] in ('Death','Dead dragon'))) for p in squad] for squad in profiles])
            self.has_poisoners = any(p and ((p['name'] == 'Ниддог' or p['unit_type'] in ('Dregazul','Spider')) or p['unit_type'] in ('Death','Dead dragon')) for squad in profiles for p in squad)
            self.cached_water = jnp.array([[bool(p and p['unit_type'] == 'Ismir son') for p in squad] for squad in profiles])
            self.water_casters = jnp.array([[bool(p and p['unit_type'] in ('Sentry','Ismir son','Drulliaan')) for p in squad] for squad in profiles])
            self.has_water = any(p and p['unit_type'] in ('Sentry','Ismir son','Drulliaan') for squad in profiles for p in squad)
            self.secondary_fear = jnp.array([[bool(p and p['unit_type'] == 'Shamanka') for p in squad] for squad in profiles])
            self.has_secondary_fear = any(p and p['unit_type'] == 'Shamanka' for squad in profiles for p in squad)
            self.fear_casters = jnp.array([[bool(p and p['unit_type'] == 'Baroness') for p in squad] for squad in profiles])
            self.has_fear = any(p and p['unit_type'] in ('Baroness','Shamanka') for squad in profiles for p in squad)
            self.weakeners = jnp.array([[bool(p and p['unit_type'] == 'Tiamat') for p in squad] for squad in profiles])
            self.has_weakening = any(p and p['unit_type'] == 'Tiamat' for squad in profiles for p in squad)
            self.secondary_paralysis_modes = jnp.array([[1 if p and (p['name'] == 'Русалка' or p['unit_type'] == 'Abyss Devil') else 2 if p and p['unit_type'] in ('Betrezen','Uter','Uter Demon','Abyss Devil') else 0 for p in squad] for squad in profiles],jnp.int32)
            self.has_secondary_paralysis = any(p and p['unit_type'] in ('Betrezen','Uter','Uter Demon','Abyss Devil') for squad in profiles for p in squad)
            self.ghost_modes = jnp.array([[2 if p and p['name'] == 'Тёмный эльф призрак' else 1 if p and p['unit_type'] in ('Ghost','Shadow','Incub') else 0 for p in squad] for squad in profiles],jnp.int32)
            self.has_paralysis = self.has_fear or self.has_secondary_paralysis or any(p and p['unit_type'] in ('Ghost','Shadow','Incub') for squad in profiles for p in squad)
            self.leech_modes = jnp.array([[2 if p and p['unit_type'] in ('Bone Lord','Highvampire') else 1 if p and p['unit_type'] in ('Dregazul','Vampire') else 0 for p in squad] for squad in profiles],jnp.int32)
            self.has_leech = any(p and p['unit_type'] in ('Bone Lord','Dregazul','Vampire','Highvampire') for squad in profiles for p in squad)
            self.cached_fire = jnp.array([[bool(p and p['unit_type'] == 'Lord') for p in squad] for squad in profiles])
            self.fire_casters = jnp.array([[bool(p and p['unit_type'] in ('Watcher', 'Lord', 'Gumtic')) for p in squad] for squad in profiles])
            self.has_fire = any(p and p['unit_type'] in ('Watcher', 'Lord', 'Gumtic') for squad in profiles for p in squad)
            self.secondary_values = jnp.array([[[p['secondary_damage'],p['secondary_accuracy']] if p else [0,0] for p in squad] for squad in profiles], jnp.float32)
            self.secondary_sources = jnp.array([[p['secondary_source'] if p else 0 for p in squad] for squad in profiles], jnp.uint32)
            self.capital_guards = jnp.array([[bool(p and p['capital_guard']) for p in squad] for squad in profiles])
            unit_slots = jnp.arange(6)
            near = jnp.abs(unit_slots[None, :] % 3 - unit_slots[:, None] % 3) <= 1
            self.all_melee_tiers = (unit_slots[None, :] // 3) * 2 + (~near).astype(jnp.int32)
            self.hero_full = self.stats_table[0, :6, HP].astype(jnp.int32)
        if self.progression_enabled:
            self.progression = ProgressionRules(game_map, self.construction)
            self.has_healers = self.progression.has_healers
            self.has_copies = self.progression.has_copies
            self.has_summons = self.progression.has_summons
            self.has_cures = self.progression.has_cures
            self.has_powerups = self.progression.has_powerups
            self.has_patriarchs = self.progression.has_patriarchs
            self.has_alchemists = self.progression.has_alchemists
            self.has_hermits = self.progression.has_hermits
            self.has_healer_wards = self.progression.has_healer_wards
            self.has_protections |= self.has_healer_wards
            self.combat_info['catalogue'] = self.progression.metadata
            self.has_protections = self.progression.has_protections or self.has_healer_wards
            self.has_aoe_accuracy_falloff = self.progression.has_aoe_accuracy_falloff
            self.has_wolf_lord = self.progression.has_wolf_lord
            self.has_shatterers = self.progression.has_shatterers
            self.has_witches = self.progression.has_witches
            self.has_wights = self.progression.has_wights
            self.has_centaurs = self.progression.has_centaurs
            self.has_double_strike = self.progression.has_double_strike
            self.has_poisoners = self.progression.has_poisoners
            self.has_paralysis = self.progression.has_paralysis
            self.has_secondary_paralysis = self.progression.has_secondary_paralysis
            self.has_weakening = self.progression.has_weakening
            self.has_fear = self.progression.has_fear
            self.has_secondary_fear = self.progression.has_secondary_fear
            self.has_leech = self.progression.has_leech
            self.has_water = self.progression.has_water
            self.has_fire = self.progression.has_fire
            self.priority_scale = 110.
        if self.items_enabled:
            self.equipment_heroes = jnp.array([bool(p and p['hero']) for p in self.progression.metadata])
            statuses = {p['status'] for _,p in self.item_rules.status_items}
            self.has_witches |= 'imp' in statuses
            self.has_wights |= 'decay' in statuses
            self.has_poisoners |= 'poison' in statuses
            self.has_paralysis |= 'paralysis' in statuses
            self.has_protections = True
        if self.has_wolf_lord or self.has_copies or self.has_summons:
            self.num_actions = POTION_START+(self.potion_rules.action_count if self.potions_enabled else 0)
        if self.has_witches:
            self.random_size = 37
        if self.has_poisoners or self.has_water or self.has_fire or self.has_wights:
            self.random_size = 67
        if self.has_paralysis:
            self.random_size = 68
        if self.has_copies or self.has_summons:
            self.random_size = 80
        if self.items_enabled and self.item_rules.status_items:
            self.random_size = max(self.random_size,80)+4
        if self.capital_enabled:
            self.capital = CapitalRules(self.progression, self.construction, game_map['agent_position'])
        self.restored_hp = jnp.concatenate((self.hero_full, jnp.zeros(6, jnp.int32)))
        # User-defined prototype rule: ceil(10% max HP) once per strategic rest.
        self.rest_healing = jnp.concatenate(((self.hero_full + 9) // 10, jnp.zeros(6, jnp.int32)))
        self.observation_version = int(game_map.get('battle_observation_version', 1))
        if self.basic_combat and self.observation_version not in (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
            raise ValueError('Combat rules version 2 requires observation version 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 or 18')
        if not self.basic_combat and self.observation_version in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
            raise ValueError('Observation versions 7вЂ“18 require combat rules version 2')
        if (self.warrior_slot >= 0 or self.has_enemy_warriors) and self.observation_version not in (3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
            raise ValueError('Warrior maps require battle_observation_version 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 or 18')
        self.observation_size = (4 + 5 * self.num_opponents + (141 if self.basic_combat else 48) + 6 if self.observation_version in (2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
                                 else self.observation_size + 2 * self.num_opponents + 12 * 8 + 6)

        if self.observation_version in (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
            self.observation_size += 48
        if self.progression_enabled != (self.observation_version in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)):
            raise ValueError('Unit progression requires observation version 9, 10, 11, 12, 13, 14, 15, 16, 17 or 18')
        if self.progression_enabled:
            self.observation_size += 60
        if self.observation_version >= 10:
            self.observation_size += 12
        if self.capital_enabled != (self.observation_version in (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)):
            raise ValueError("Capital services require observation version 11, 12, 13, 14, 15, 16, 17 or 18")
        if self.capital_enabled:
            self.observation_size += 17
        if self.potions_enabled != (self.observation_version in (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)):
            raise ValueError("Map potions require observation version 12, 13, 14, 15, 16, 17 or 18")
        if self.potions_enabled:
            self.observation_size += self.potion_rules.count+(36 if self.has_potion_buffs else 0)
        if self.chests_enabled != (self.observation_version in (13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)):
            raise ValueError('Map chests require observation version 13, 14, 15, 16, 17 or 18')
        if self.chests_enabled:
            self.observation_size += (1+self.chest_rules.features.shape[1])*self.chest_rules.count
        if self.items_enabled:
            self.observation_size += self.item_rules.capacity+5+self.item_rules.chest_loot.size
        if self.observation_version >= 14:
            self.observation_size += 12
        if self.observation_version >= 15:
            self.observation_size += 36
        if self.observation_version >= 16:
            self.observation_size += 12
        if self.observation_version >= 17:
            self.observation_size += 36
        if self.observation_version >= 18:
            self.observation_size += 36
        if self.observation_version >= 21:
            self.observation_size += 30
        if self.observation_version >= 22:
            self.observation_size += 13
        if self.observation_version >= 23:
            self.observation_size += 120
        if self.observation_version >= 24:
            self.observation_size += 72
        if self.observation_version >= 25:
            self.observation_size += 24
        if self.observation_version >= 26:
            self.observation_size += 24
        if self.observation_version >= 27:
            self.observation_size += 36
        if self.observation_version >= 28:
            self.observation_size += 48
        if self.observation_version >= 29:
            self.observation_size += 24

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
            native_ids=jnp.zeros(12,jnp.int32), native_levels=jnp.zeros(12,jnp.int32),
            native_xp=jnp.zeros(12,jnp.int32), copied=jnp.zeros(12,bool),
            copy_stats=jnp.zeros((12,5),jnp.float32), copy_traits=jnp.zeros((12,4),jnp.uint32),
            copy_secondary=jnp.zeros((12,2),jnp.float32),
            summon_owner=jnp.full(12,-1,jnp.int32), preparation=jnp.zeros(12,bool),
            fenrir=jnp.zeros(12, jnp.bool_),
            armor_shreds=jnp.zeros(12, jnp.int32),
            imp=jnp.zeros(12, jnp.bool_), imp_wards=jnp.zeros(12, jnp.uint32),
            activation_done=jnp.zeros(12, jnp.bool_),
            post_victory=jnp.int32(0),pending_healers=jnp.zeros(12,jnp.bool_),
            initiative_override=jnp.full(12,-1,jnp.int32),saved_initiative_override=jnp.full(12,-1,jnp.int32),
            slow_original=jnp.full(12,-1,jnp.int32),waited=jnp.zeros(12,bool),bonus_turns=jnp.zeros(12,jnp.int32),
            battle_revived=jnp.zeros(12,bool),revival_xp_cutoff=jnp.zeros(12,jnp.int32),
            primary_override=jnp.full(12,-1,jnp.int32),preweak_damage=jnp.full(12,-1,jnp.int32),
            powerup=jnp.zeros(12,bool),powerup_layered=jnp.zeros(12,bool),
            saved_primary_override=jnp.full(12,-1,jnp.int32),saved_preweak_damage=jnp.full(12,-1,jnp.int32),
            saved_powerup=jnp.zeros(12,bool),saved_powerup_layered=jnp.zeros(12,bool),
            healer_wards=jnp.zeros((12,4),jnp.uint32),ward_native_used=jnp.zeros(12,jnp.uint32),
            saved_healer_wards=jnp.zeros((12,4),jnp.uint32),saved_ward_native_used=jnp.zeros(12,jnp.uint32),
            feared=jnp.zeros(12,jnp.bool_),enemy_initial_hp=jnp.zeros(6,jnp.int32),
            enemy_damage_credit=jnp.zeros(6,jnp.int32),enemy_killed_credit=jnp.zeros(6,jnp.bool_),
            weakened=jnp.zeros(12,jnp.bool_),saved_weakened=jnp.zeros(12,jnp.bool_),
            paralyzed=jnp.zeros(12,jnp.bool_),long_paralyzed=jnp.zeros(12,jnp.bool_),
            decay_form=jnp.zeros(12, jnp.int32), decay_shreds=jnp.zeros(12, jnp.int32),
            position=jnp.asarray(self.map_config['agent_position'], jnp.int32),
            number=jnp.int32(self.initial_number), alive=jnp.ones(self.num_opponents, bool),
            step_count=zero, done=jnp.bool_(False), won=jnp.bool_(False), lost=jnp.bool_(False),
            visited=visited, battle_key=rng_key, in_battle=jnp.bool_(False), enemy=jnp.int32(-1),
            origin=jnp.asarray(self.map_config['agent_position'], jnp.int32),
            second_strike=jnp.bool_(False), poison_turns=jnp.zeros(12, jnp.int32),
            poison_damage=jnp.zeros(12, jnp.int32), poison_source=jnp.full(12, -1, jnp.int32),
            last_poison_damage=jnp.zeros(12, jnp.int32),
            water_turns=jnp.zeros(12,jnp.int32), water_damage=jnp.zeros(12,jnp.int32),
            water_source=jnp.full(12,-1,jnp.int32), last_water_damage=jnp.zeros(12,jnp.int32),
            burn_turns=jnp.zeros(12,jnp.int32),burn_damage=jnp.zeros(12,jnp.int32),
            burn_source=jnp.full(12,-1,jnp.int32),last_burn_damage=jnp.zeros(12,jnp.int32),
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
            potions=(self.potion_rules.initial_counts if self.potions_enabled else jnp.zeros(0, jnp.int32)),
            potion_doses=jnp.zeros((12,5),jnp.int32),potion_bonus=jnp.zeros((12,7)),
            potion_temporary=jnp.zeros((12,7)),
            potion_active=jnp.zeros(12,jnp.uint32),potion_wards=jnp.zeros(12,jnp.uint32),
            chest_alive=jnp.ones(self.chest_rules.count if self.chests_enabled else 0, bool),
            last_loot=jnp.zeros(self.potion_rules.count if self.potions_enabled else 0, jnp.int32),
            item_inventory=(self.item_rules.initial_inventory if self.items_enabled else jnp.zeros(0,jnp.int32)),
            equipped=jnp.full(5,-1,jnp.int32),equipment_bonus=jnp.zeros((12,7)),
            last_item_loot=jnp.zeros(self.item_rules.count if self.items_enabled else 0,jnp.int32),
            last_potion=jnp.int32(-1), recovery_balance=jnp.zeros(2, jnp.int32), last_service_cost=zero,
            gold=zero, map_steps=zero,
            movement_points=jnp.int32(MAX_MOVEMENT_POINTS), day=jnp.int32(1),
            buildings=jnp.uint32(0),
            blocked_buildings=(self.construction.initial_blocked if self.basic_combat else jnp.uint32(0)),
            built_today=jnp.bool_(False), last_building=jnp.int32(-1),
        )
        if self.items_enabled:
            state=self._refresh_equipment(state)
        return state, self._timestep(state, jnp.float32(0), first=True)

    def _equipment_hero(self,state):
        heroes=self.equipment_heroes[state.unit_ids[:6]]
        return jnp.where(jnp.any(heroes),jnp.argmax(heroes),-1)

    def _refresh_equipment(self,state):
        hero=self._equipment_hero(state)
        equipped=self.item_rules.choose(state,(hero>=0)&(state.hp[hero]>0),state.unit_levels[hero])
        old_cap=self.movement_cap(state)
        state=state.replace(equipped=equipped)
        cap=self.movement_cap(state)
        intrinsic=self._potion_intrinsic(state.unit_ids,state.unit_levels)
        bonus=self.item_rules.numeric_bonus(state,intrinsic,hero)
        return state.replace(equipment_bonus=bonus,
            movement_points=jnp.minimum(cap,state.movement_points+jnp.maximum(0,cap-old_cap)))

    def movement_cap(self,state):
        return (self.item_rules.movement_cap(state,MAX_MOVEMENT_POINTS)
                if self.items_enabled else jnp.int32(MAX_MOVEMENT_POINTS))

    def _equipment_wards(self,state):
        return jnp.where(self.slots==self._equipment_hero(state),self.item_rules.value(state,'ward',3),jnp.uint32(0))

    def turn_metadata(self):
        return {"movement_points": MAX_MOVEMENT_POINTS, "move_cost": MOVE_COST,
                "battle_entry_cost": BATTLE_ENTRY_COST, "attack_requires_points": 1,
                "income": DAILY_GOLD, "rest_action": REST,
                "regeneration_percent": 10, "regeneration_rounding": "ceil", "automatic_revive": False,
                "fenrir_action": FENRIR}

    def map_commands(self, state):
        """UI/export quotes: exact target stack (-1 for a move) and point cost."""
        destinations = state.position[None] + self.directions
        matches = state.alive[None, :] & jnp.all(
            destinations[:, None, :] == self.opponent_positions[None, :, :], axis=-1)
        attacking = jnp.any(matches, axis=1)
        targets = jnp.where(attacking, jnp.argmax(matches, axis=1), -1)
        costs = jnp.where(attacking, jnp.minimum(state.movement_points, (self.movement_cap(state)+1)//2), MOVE_COST)
        return jnp.stack((targets, costs), axis=1)

    def rest_penalty(self, state):
        return state.movement_points * jnp.float32(self.rest_penalty_per_point)

    def unit_stats(self, state):
        """Five effective characteristics per slot; empty/world enemy slots are zero."""
        values = self._combat_stats(state)
        return jnp.where(((self.slots < 6) | state.in_battle)[:, None], values, 0.)

    def _effective_ids(self, state):
        return jnp.where(state.decay_form > 0, state.decay_form, state.unit_ids) if self.has_wights else state.unit_ids

    def _effective_levels(self, state):
        return jnp.where(state.decay_form > 0, self.progression.base_levels[state.decay_form], state.unit_levels) if self.has_wights else state.unit_levels

    def _effective_shreds(self, state):
        return jnp.where(state.decay_form > 0, state.decay_shreds, state.armor_shreds) if self.has_wights else state.armor_shreds

    def _potion_intrinsic(self, ids, levels):
        values=self.progression.stats(ids,levels)
        # Potions change original power before weakening and the final damage cap.
        values=values.at[...,DAMAGE].set(self.progression.raw_stats(ids,levels)[...,DAMAGE])
        return jnp.concatenate((values,
                                self.progression.secondary_stats(ids,levels)),axis=-1)

    def _potion_stats(self,state,values,slot=None,temporary=True):
        if not self.has_potion_buffs and not self.items_enabled:
            return values
        bonus = state.potion_bonus[:,:5] if slot is None else state.potion_bonus[slot,:5]
        active = state.summon_owner < 0 if slot is None else state.summon_owner[slot] < 0
        values = values+jnp.where(active[...,None],bonus,0)
        if temporary:
            delta = state.potion_temporary[:,:5] if slot is None else state.potion_temporary[slot,:5]
            if self.items_enabled:
                delta=delta+(state.equipment_bonus[:,:5] if slot is None else state.equipment_bonus[slot,:5])
            values = values+jnp.where(active[...,None],delta,0)
        return values.at[...,ACCURACY].min(100.)

    def _natural_stats(self,state):
        return self._potion_stats(state,self.progression.stats(state.unit_ids,state.unit_levels))

    def _secondary_stats(self,state,slot,temporary=True):
        ids,levels=self._effective_ids(state)[slot],self._effective_levels(state)[slot]
        values=self.progression.secondary_stats(ids,levels)
        if self.has_potion_buffs:
            bonus=state.potion_bonus[slot,5:]
            if temporary:
                bonus=bonus+state.potion_temporary[slot,5:]
            natural=~state.copied[slot] & (state.decay_form[slot]==0) & (state.summon_owner[slot]<0)
            values=values+jnp.where(natural,bonus,0)
        if self.has_copies:
            values=jnp.where(state.copied[slot] & (state.decay_form[slot]==0),state.copy_secondary[slot],values)
        return jnp.where(state.imp[slot] | state.fenrir[slot],0.,values).at[1].min(100.)

    def _combat_stats(self, state, cap=True):
        if self.progression_enabled:
            # Lowered forms always have their native level. Fetching their
            # base row avoids a dependent level lookup through the growth table.
            values = self.progression.stats(state.unit_ids, state.unit_levels)
            if self.has_weakening or self.has_powerups or self.has_potion_buffs or self.items_enabled:
                values = values.at[:,DAMAGE].set(self.progression.raw_stats(state.unit_ids,state.unit_levels)[:,DAMAGE])
            values = self._potion_stats(state,values)
            if self.has_copies:
                values = jnp.where(state.copied[:,None],state.copy_stats,values)
                uncopied = self.progression.doppelgangers[state.unit_ids]
                fallback = uncopied & ~jnp.any(self._copy_targets(state))
                alternate_accuracy = jnp.minimum(100.,self.progression.raw_stats(state.unit_ids,state.unit_levels)[:,ACCURACY]-20)
                values = values.at[:,ACCURACY].set(jnp.where(fallback,alternate_accuracy,values[:,ACCURACY]))
            if self.has_wights:
                values = jnp.where((state.decay_form > 0)[:, None],
                                   self.progression.base_stats[state.decay_form], values)
        else:
            values = self.stats_table[jnp.maximum(state.enemy, 0)]
        if self.has_wolf_lord:
            values = values.at[:, HP].set(jnp.where(state.fenrir, 275, values[:, HP]))
            values = values.at[:, DAMAGE].set(jnp.where(state.fenrir, 90, values[:, DAMAGE]))
            values = values.at[:, INITIATIVE].set(jnp.where(state.fenrir, 65, values[:, INITIATIVE]))
        if self.has_shatterers:
            values = values.at[:, ARMOR].set(armor_after_shreds(values[:, ARMOR], self._effective_shreds(state)))
        if self.has_witches:
            big = (self.progression.sizes[self._effective_ids(state)] if self.progression_enabled
                   else self.size_table[jnp.maximum(state.enemy, 0)]) == 2
            values = values.at[:, DAMAGE].set(jnp.where(state.imp, jnp.where(big, 30, 20), values[:, DAMAGE]))
            values = values.at[:, ACCURACY].set(jnp.where(state.imp, jnp.where(big, 70, 80), values[:, ACCURACY]))
            values = values.at[:, ARMOR].set(jnp.where(state.imp, 0, values[:, ARMOR]))
            values = values.at[:, INITIATIVE].set(jnp.where(state.imp, jnp.where(big, 50, 30), values[:, INITIATIVE]))
        if self.has_weakening or self.has_powerups or self.has_potion_buffs or self.items_enabled:
            damage = jnp.rint(values[:,DAMAGE]*jnp.where(state.weakened,.68,1.))
            caps = (self.progression.stat_caps[self._effective_ids(state),DAMAGE] if self.progression_enabled
                    else jnp.full(12,300.))
            caps = jnp.where(state.imp | state.fenrir,300.,caps)
            if self.has_powerups:
                damage = jnp.where(state.primary_override >= 0,state.primary_override,damage)
            values = values.at[:,DAMAGE].set(jnp.minimum(damage,caps) if cap else damage)
        if self.has_hermits:
            values = values.at[:,INITIATIVE].set(jnp.where(state.initiative_override >= 0,state.initiative_override,values[:,INITIATIVE]))
        return values

    def _original_primary(self,state,saved=False):
        if self.progression_enabled:
            ids = state.unit_ids if saved else self._effective_ids(state)
            levels = state.unit_levels if saved else self._effective_levels(state)
            amount = self.progression.raw_stats(ids,levels)[:,DAMAGE]
            if self.has_potion_buffs or self.items_enabled:
                enhanced = amount+state.potion_bonus[:,DAMAGE]+state.potion_temporary[:,DAMAGE]
                if self.items_enabled:
                    enhanced += state.equipment_bonus[:,DAMAGE]
                amount = jnp.where((state.summon_owner < 0) & ~state.copied & ((state.decay_form == 0) | saved),enhanced,amount)
            if self.has_copies:
                amount=jnp.where(state.copied & ((state.decay_form==0) | saved),state.copy_stats[:,DAMAGE],amount)
        else:
            amount = self.stats_table[state.enemy,:,DAMAGE]
        if not saved and self.has_witches:
            big = (self.progression.sizes[self._effective_ids(state)] if self.progression_enabled else self.size_table[state.enemy]) == 2
            amount = jnp.where(state.imp,jnp.where(big,30,20),amount)
        # Fenrir retains the Wolf Lord original_damage field in the reference.
        return amount.astype(jnp.int32)

    def _capture_initiative_form(self,state,mask,first):
        if not self.has_hermits:
            return state
        original = self._combat_stats(state)[:,INITIATIVE].astype(jnp.int32)
        return state.replace(saved_initiative_override=jnp.where(first,original,state.saved_initiative_override),
            initiative_override=jnp.where(mask,-1,state.initiative_override))

    def _restore_initiative_form(self,state,mask):
        if not self.has_hermits:
            return state
        return state.replace(initiative_override=jnp.where(mask,state.saved_initiative_override,state.initiative_override),
            saved_initiative_override=jnp.where(mask,-1,state.saved_initiative_override))

    def _restore_slow(self,state,priority,phases,mask):
        if not self.has_hermits:
            return state,priority
        restore = mask & (state.slow_original >= 0)
        priority = jnp.where(restore & (phases < 2),jnp.maximum(priority,state.slow_original),priority)
        return state.replace(initiative_override=jnp.where(restore,state.slow_original,state.initiative_override),
            slow_original=jnp.where(restore,-1,state.slow_original)),priority

    def _alchemist_units(self,state):
        flags = (self.progression.alchemists[self._effective_ids(state)] if self.progression_enabled else self.alchemists[state.enemy])
        return flags & ~state.imp

    def _alchemist_targets(self,state):
        return (state.hp > 0) & ~state.escaped & ~state.retreating & ~self._alchemist_units(state) & (self.slots != state.actor) & ((self.slots < 6) == (state.actor < 6))

    def _actor_is_patriarch(self,state):
        if not self.has_patriarchs:
            return jnp.bool_(False)
        flag = (self.progression.patriarchs[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.patriarchs[state.enemy,state.actor])
        return flag & ~state.imp[state.actor]

    def _patriarch_targets(self,state):
        alive = (state.hp > 0) & ~state.escaped
        sizes = (self.progression.sizes[self._effective_ids(state)] if self.progression_enabled else self.size_table[state.enemy])
        natural_sizes = self.progression.sizes[state.unit_ids] if self.progression_enabled else sizes
        pair = jnp.where(self.slots%6 < 3,self.slots+3,self.slots-3)
        back = self.slots%6 >= 3
        covered = back & alive[pair] & (sizes[pair] == 2)
        footprint_free = ~covered & ~((natural_sizes == 2) & alive[pair])
        occupied = (state.unit_ids != 0) if self.progression_enabled else (self._combat_stats(state)[:,HP] > 0)
        revive = (state.hp <= 0) & ~state.battle_revived & footprint_free
        # Dead slots have zero effective initiative in the JAX scheduler.
        return occupied & ~state.escaped & ~covered & (alive | revive) & ((self.slots < 6) == (state.actor < 6))

    def _actor_power_cures(self,state):
        if not self.has_powerups or not self.has_cures:
            return jnp.bool_(False)
        flag = (self.progression.power_cures[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.power_cures[state.enemy,state.actor])
        return flag & ~state.imp[state.actor]

    def _actor_power_factor(self,state):
        if not self.has_powerups:
            return jnp.float32(0)
        factor = (self.progression.power_factors[self._effective_ids(state)[state.actor]] if self.progression_enabled
                  else self.power_factors[state.enemy,state.actor])
        return jnp.where(state.imp[state.actor],0.,factor)

    def _capture_damage_forms(self,state,mask,first):
        if not self.has_powerups:
            return state
        update = {}
        for name,empty in (('primary_override',-1),('preweak_damage',-1),('powerup',False),('powerup_layered',False)):
            update['saved_'+name] = jnp.where(first,getattr(state,name),getattr(state,'saved_'+name))
            update[name] = jnp.where(mask,empty,getattr(state,name))
        return state.replace(**update)

    def _restore_damage_forms(self,state,mask):
        if not self.has_powerups:
            return state
        update = {}
        for name,empty in (('primary_override',-1),('preweak_damage',-1),('powerup',False),('powerup_layered',False)):
            update[name] = jnp.where(mask,getattr(state,'saved_'+name),getattr(state,name))
            update['saved_'+name] = jnp.where(mask,empty,getattr(state,'saved_'+name))
        return state.replace(**update)

    def _expire_powerups(self,state,spent):
        if not self.has_powerups:
            return state
        update = {}
        for prefix in ('','saved_'):
            active = spent & getattr(state,prefix+'powerup')
            original = self._original_primary(state,saved=bool(prefix))
            restored = jnp.rint(original*jnp.where(getattr(state,prefix+'powerup_layered'),.68,1.)).astype(jnp.int32)
            update[prefix+'primary_override'] = jnp.where(active,restored,getattr(state,prefix+'primary_override'))
            update[prefix+'preweak_damage'] = jnp.where(active,original,getattr(state,prefix+'preweak_damage'))
            update[prefix+'powerup'] = getattr(state,prefix+'powerup') & ~spent
            update[prefix+'powerup_layered'] = getattr(state,prefix+'powerup_layered') & ~active
        return state.replace(**update)

    def _base_combat_traits(self, state):
        if self.progression_enabled:
            traits = self.progression.traits[self._effective_ids(state)]
            if self.has_copies:
                traits = jnp.where((state.copied & (state.decay_form == 0))[:,None],state.copy_traits,traits)
                fallback = self.progression.doppelgangers[self._effective_ids(state)] & ~jnp.any(self._copy_targets(state))
                traits = traits.at[:,0].set(jnp.where(fallback,MELEE,traits[:,0]))
                traits = traits.at[:,1].set(jnp.where(fallback,1,traits[:,1]))
        else:
            traits = self.combat_traits[jnp.maximum(state.enemy, 0)]
        if self.has_potion_buffs:
            natural = ~state.copied & (state.decay_form == 0) & (state.summon_owner < 0)
            traits = traits.at[:,3].set(traits[:,3] | jnp.where(natural,state.potion_wards,jnp.uint32(0)))
        if self.items_enabled:
            natural = ~state.copied & (state.decay_form == 0) & (state.summon_owner < 0)
            traits = traits.at[:,3].set(traits[:,3] | jnp.where(natural,self._equipment_wards(state),jnp.uint32(0)))
        if self.has_wolf_lord:
            traits = traits.at[:, 0].set(jnp.where(state.fenrir, MELEE, traits[:, 0]))
            traits = traits.at[:, 1].set(jnp.where(state.fenrir, 1, traits[:, 1]))
        if self.has_witches:
            traits = jnp.where(state.imp[:, None], jnp.array([MELEE, 1, 0, 0], jnp.uint32), traits)
        return traits

    def _combat_traits(self,state):
        traits = self._base_combat_traits(state)
        if self.has_healer_wards:
            traits = traits.at[:,3].set(traits[:,3] | ward_tags(state.healer_wards))
        return traits

    def _expire_healer_wards(self,state,used,casters):
        if not self.has_healer_wards:
            return state,used
        native = self._base_combat_traits(state)[:,3]
        saved_native = (self.progression.traits[state.unit_ids,3] if self.progression_enabled
                        else self.combat_traits[state.enemy,:,3])
        if self.has_potion_buffs:
            saved_native |= jnp.where(~state.copied & (state.summon_owner < 0),state.potion_wards,jnp.uint32(0))
        if self.items_enabled:
            saved_native |= jnp.where(~state.copied & (state.summon_owner < 0),self._equipment_wards(state),jnp.uint32(0))
        owners,sticky,used = expire_wards(state.healer_wards,state.ward_native_used,used,native,casters)
        saved,saved_sticky,saved_used = expire_wards(state.saved_healer_wards,state.saved_ward_native_used,state.imp_wards,saved_native,casters)
        return state.replace(healer_wards=owners,ward_native_used=sticky,
            saved_healer_wards=saved,saved_ward_native_used=saved_sticky,imp_wards=saved_used),used

    def _capture_ward_forms(self,state,mask,first):
        if not self.has_healer_wards:
            return state
        return state.replace(
            saved_healer_wards=jnp.where(first[:,None],state.healer_wards,state.saved_healer_wards),
            saved_ward_native_used=jnp.where(first,state.ward_native_used,state.saved_ward_native_used),
            healer_wards=jnp.where(mask[:,None],jnp.uint32(0),state.healer_wards),
            ward_native_used=jnp.where(mask,jnp.uint32(0),state.ward_native_used))

    def _restore_ward_forms(self,state,mask):
        if not self.has_healer_wards:
            return state
        return state.replace(healer_wards=jnp.where(mask[:,None],state.saved_healer_wards,state.healer_wards),
            ward_native_used=jnp.where(mask,state.saved_ward_native_used,state.ward_native_used),
            saved_healer_wards=jnp.where(mask[:,None],jnp.uint32(0),state.saved_healer_wards),
            saved_ward_native_used=jnp.where(mask,jnp.uint32(0),state.saved_ward_native_used))

    def _actor_is_wolf_lord(self, state):
        if not self.has_wolf_lord:
            return jnp.bool_(False)
        wolf = (self.progression.wolf_lord[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.wolf_lord[jnp.maximum(state.enemy, 0), state.actor])
        return wolf & ~state.fenrir[state.actor] & ~state.imp[state.actor]

    def _actor_is_witch(self, state):
        if not self.has_witches:
            return jnp.bool_(False)
        witch = (self.progression.witches[self._effective_ids(state)[state.actor]] if self.progression_enabled
                 else self.witches[jnp.maximum(state.enemy, 0), state.actor])
        return witch & ~state.imp[state.actor]

    def _actor_ghost_mode(self,state):
        if not self.has_paralysis:
            return jnp.int32(0)
        mode = (self.progression.ghost_modes[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.ghost_modes[state.enemy,state.actor])
        return jnp.where(state.imp[state.actor],0,mode)

    def _actor_is_fear_caster(self, state):
        if not self.has_fear:
            return jnp.bool_(False)
        caster = (self.progression.fear_casters[self._effective_ids(state)[state.actor]] if self.progression_enabled
                  else self.fear_casters[state.enemy,state.actor])
        return caster & ~state.imp[state.actor]

    def _actor_secondary_paralysis(self, state):
        if not self.has_secondary_paralysis:
            return jnp.int32(0)
        mode = (self.progression.secondary_paralysis_modes[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.secondary_paralysis_modes[state.enemy,state.actor])
        return jnp.where(state.imp[state.actor],0,mode)

    def _actor_double_strike(self, state):
        if not self.has_double_strike:
            return jnp.bool_(False)
        double = (self.progression.double_strike[self._effective_ids(state)[state.actor]] if self.progression_enabled
                  else self.double_strike[state.enemy, state.actor])
        return double & ~state.imp[state.actor]

    def _start_activation(self, state, roll, enabled=True):
        if not self.has_witches and not self.has_poisoners and not self.has_water and not self.has_fire and not self.has_wights:
            if self.has_hermits:
                return state.replace(activation_done=state.activation_done.at[state.actor].set(
                    state.activation_done[state.actor] | enabled))
            return state
        actor = state.actor
        lowered = state.decay_form[actor] > 0
        leaving = state.retreating[actor] & ~state.paralyzed[actor] & ~state.long_paralyzed[actor]
        recover = enabled & ~leaving & (state.imp[actor] | lowered) & ~state.activation_done[actor] & (roll < jnp.where(lowered, .01, .3))
        natural = (self._natural_stats(state)[actor]
                   if self.progression_enabled else self.stats_table[state.enemy, actor])
        if self.has_copies:
            natural = jnp.where(state.copied[actor],state.copy_stats[actor],natural)
        initiative, maximum = natural[INITIATIVE], natural[HP]
        if self.has_wolf_lord:
            initiative = jnp.where(state.fenrir[actor], 65., initiative)
            maximum = jnp.where(state.fenrir[actor], 275., maximum)
        hp = state.hp
        if self.has_wights:
            current_max = self.progression.base_stats[state.decay_form[actor], HP]
            restored = jnp.rint(hp[actor] / jnp.maximum(current_max, 1) * maximum).astype(jnp.int32)
            restored = jnp.where(hp[actor] > 0, jnp.maximum(restored, 1), 0)
            hp = hp.at[actor].set(jnp.where(recover & lowered, restored, hp[actor]))
        state = self._restore_initiative_form(state,(self.slots == actor) & recover)
        initiative = jnp.where(state.initiative_override[actor] >= 0,state.initiative_override[actor],initiative)
        priority = jnp.where(recover & (state.turn_phase[actor] != 1), initiative, state.priority[actor])
        state = self._restore_damage_forms(state,(self.slots == actor) & recover)
        state = self._restore_ward_forms(state,(self.slots == actor) & recover)
        return state.replace(hp=hp, weakened=state.weakened.at[actor].set(jnp.where(recover,state.saved_weakened[actor],state.weakened[actor])),
            imp=state.imp.at[actor].set(state.imp[actor] & ~recover),
            decay_form=state.decay_form.at[actor].set(jnp.where(recover, 0, state.decay_form[actor])),
            decay_shreds=state.decay_shreds.at[actor].set(jnp.where(recover, 0, state.decay_shreds[actor])),
            wards_used=state.wards_used.at[actor].set(jnp.where(recover, state.imp_wards[actor], state.wards_used[actor])),
            activation_done=state.activation_done.at[actor].set(state.activation_done[actor] | enabled),
            priority=state.priority.at[actor].set(priority))

    def unit_experience(self, state):
        if not self.progression_enabled:
            return jnp.zeros((12, 4), jnp.int32)
        values = self.progression.experience(state.unit_ids, state.unit_levels, state.unit_xp)
        if self.has_copies or self.has_summons:
            native = self.progression.experience(state.native_ids,state.native_levels,state.native_xp)
            values = jnp.where(state.copied[:,None],native,values)
            values = jnp.where((state.summon_owner >= 0)[:,None],0,values)
        return jnp.where(((self.slots < 6) | state.in_battle)[:, None], values, 0)

    def unit_traits(self, state):
        traits = self._combat_traits(state)
        traits = traits.at[:, 3].set(traits[:, 3] & ~state.wards_used)
        return jnp.where(((self.slots < 6) | state.in_battle)[:, None], traits, 0)

    def unit_sizes(self, state):
        sizes = (self.progression.sizes[self._effective_ids(state)] if self.progression_enabled
                 else self.size_table[jnp.maximum(state.enemy, 0)])
        if self.has_copies:
            sizes = jnp.where(state.copied,1,sizes)
        return jnp.where((self.slots < 6) | state.in_battle, sizes, 0)

    def _actor_mass_healer(self, state):
        if not self.basic_combat or not self.has_healers:
            return jnp.bool_(False)
        flag = (self.progression.mass_healers[self._effective_ids(state)[state.actor]] if self.progression_enabled
                else self.mass_healers[state.enemy,state.actor])
        return flag & ~state.imp[state.actor]

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
        if self.observation_version in (2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
            # No duplicate max-HP arrays or obsolete numeric battle strengths.
            # Signed queue priority contains both order and waiting/acted status.
            world = jnp.stack((self.opponent_positions[:,0] / (self.size-1),
                               self.opponent_positions[:,1] / (self.size-1), state.alive,
                               self.enemy_counts / 6, self.enemy_health / self.hero_hp),axis=1)
            active = (state.hp > 0) & ~state.escaped
            queue = jnp.where(active & (state.turn_phase == 0), state.priority,
                             jnp.where(active & (state.turn_phase == 1), -state.priority, 0.))
            hp_scale = (jnp.maximum(self.max_hp(state), 1) if self.observation_version in (3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
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
            if self.chests_enabled:
                context = jnp.concatenate((self.chest_rules.observation(state), context))
            if self.progression_enabled:
                xp = self.unit_experience(state).astype(jnp.float32)
                ids = jnp.where((self.slots < 6) | state.in_battle, state.unit_ids, 0)
                growth = jnp.stack((ids / (len(self.progression.rows)-1), xp[:, 0]/100.,
                                   xp[:, 1]/1000., xp[:, 2]/10000., xp[:, 3]/jnp.maximum(xp[:, 2], 1)), axis=1)
                context = jnp.concatenate((growth.reshape(-1), context))
            if self.observation_version in (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30):
                # Compact exact source/bitset encoding; keeps the GPU policy input small.
                context = jnp.concatenate((self.unit_traits(state).reshape(-1) /
                                           jnp.tile(jnp.array([4. if self.observation_version >= 10 else 3., 9., 511., 511.]), 12), context))
            if self.observation_version >= 10:
                context = jnp.concatenate((context, self.unit_sizes(state) / 2.))
            if self.observation_version >= 14:
                flags = state.imp.astype(jnp.int32) + 2*state.fenrir + 4*state.activation_done + 8*((self.slots == state.actor) & state.second_strike)
                flags += 16*state.paralyzed + 32*state.long_paralyzed + 64*state.weakened + 128*state.saved_weakened
                context = jnp.concatenate((context, flags / 255.))
            if self.observation_version >= 15:
                poison = jnp.stack((state.poison_turns / 6., state.poison_damage / 300., (state.poison_source+1) / 12.), axis=1)
                context = jnp.concatenate((context, poison.reshape(-1)))
            if self.observation_version >= 16:
                context = jnp.concatenate((context, state.decay_form / len(self.progression.rows)))
            if self.observation_version >= 17:
                water = jnp.stack((state.water_turns / 6., state.water_damage / 300., (state.water_source+1) / 12.),axis=1)
                context = jnp.concatenate((context,water.reshape(-1)))
            if self.observation_version >= 18:
                burn = jnp.stack((state.burn_turns/6.,state.burn_damage/300.,(state.burn_source+1)/12.),axis=1)
                context = jnp.concatenate((context,burn.reshape(-1)))
            if self.observation_version >= 21:
                credit = jnp.stack((state.enemy_initial_hp/1000.,state.enemy_damage_credit/1000.,state.enemy_killed_credit),axis=1)
                context = jnp.concatenate((context,state.feared,jnp.where(state.in_battle,credit.reshape(-1),0.)))
            if self.observation_version >= 22:
                context = jnp.concatenate((context,jnp.array([state.post_victory/2.]),state.pending_healers))
            if self.observation_version >= 23:
                context = jnp.concatenate((context,state.healer_wards.reshape(-1)/4095.,state.saved_healer_wards.reshape(-1)/4095.,
                    state.ward_native_used/511.,state.saved_ward_native_used/511.))
            if self.observation_version >= 24:
                for prefix in ('','saved_'):
                    flags = getattr(state,prefix+'powerup').astype(jnp.int32)+2*getattr(state,prefix+'powerup_layered')
                    context = jnp.concatenate((context,getattr(state,prefix+'primary_override')/1000.,getattr(state,prefix+'preweak_damage')/1000.,flags/3.))
            if self.observation_version >= 25:
                context = jnp.concatenate((context,state.battle_revived,state.revival_xp_cutoff/10000.))
            if self.observation_version >= 26:
                context = jnp.concatenate((context,state.waited,state.bonus_turns/10.))
            if self.observation_version >= 27:
                context = jnp.concatenate((context,state.initiative_override/100.,state.saved_initiative_override/100.,state.slow_original/100.))
            if self.observation_version >= 28:
                context = jnp.concatenate((context,state.copied,(state.summon_owner+1)/12.,state.preparation,
                    jnp.where(state.copied | (state.summon_owner >= 0),state.native_ids,0)/(len(self.progression.rows)-1)))
            if self.observation_version >= 29:
                context=jnp.concatenate((context,jnp.where(state.copied[:,None],state.copy_secondary,0).reshape(-1)/300.))
            if self.items_enabled:
                context=jnp.concatenate((context,self.item_rules.observation(state)))
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
        controlled = state.in_battle & (state.actor < 6) & ~state.retreating[state.actor] & ~state.paralyzed[state.actor] & ~state.long_paralyzed[state.actor]
        targets = (state.hp[6:] > 0) & ~state.escaped[6:] & controlled
        if self.basic_combat:
            if self.has_healers:
                targets = jnp.where(self._actor_is_healer(state),
                    (state.hp[:6] > 0) & ~state.escaped[:6] & controlled, targets)
            if self.has_patriarchs:
                targets = jnp.where(self._actor_is_patriarch(state),self._patriarch_targets(state)[:6] & controlled,targets)
            if self.has_alchemists:
                targets = jnp.where(self._alchemist_units(state)[state.actor],self._alchemist_targets(state)[:6] & controlled,targets)
            targets &= (self._actor_power_factor(state) == 0) | (self.slots[:6] != state.actor) | self._actor_power_cures(state)
            targets &= ~self._actor_is_melee(state) | self._melee_targets(state)
        elif self.warrior_slot >= 0:
            targets &= (state.actor != self.warrior_slot) | self._melee_targets(state)
        mask = jnp.concatenate((movement, targets, jnp.asarray([
            controlled & ~state.second_strike, controlled & ~state.second_strike & (state.turn_phase[state.actor] == 0) & ~state.waited[state.actor],
            controlled & ~state.second_strike, state.in_battle & ~controlled])))
        if self.basic_combat:
            mask = jnp.concatenate((mask, self.construction.available(state),
                                    jnp.asarray([~state.in_battle & ~state.done])))
        if self.capital_enabled:
            heal, revive = self.capital.available(state, self.max_hp(state))
            mask = jnp.concatenate((mask, heal, revive))
        if self.num_actions >= POTION_START:
            mask = jnp.concatenate((mask, jnp.zeros(FENRIR-mask.shape[0], jnp.bool_),
                                    jnp.asarray([controlled & ~state.second_strike & self._actor_is_wolf_lord(state)]),
                                    jnp.zeros(COPY_ACTIONS,bool)))
        if self.potions_enabled:
            mask = jnp.concatenate((mask, self.potion_rules.available(state, self.max_hp(state)).reshape(-1)))
        if self.has_copies:
            copy = self._actor_copies(state) & controlled
            valid_copy = self._copy_targets(state)
            mask = mask.at[SHOOT:DEFEND].set(jnp.where(copy,valid_copy[6:],mask[SHOOT:DEFEND]))
            mask = mask.at[COPY_ALLY:COPY_ALLY+COPY_ACTIONS].set(copy & valid_copy[:6])
            preparing = jnp.zeros(self.num_actions,bool).at[SHOOT:DEFEND].set(copy & valid_copy[6:])
            preparing = preparing.at[COPY_ALLY:COPY_ALLY+COPY_ACTIONS].set(copy & valid_copy[:6]).at[DEFEND].set(controlled).at[CONTINUE].set(~controlled)
            mask = jnp.where(state.in_battle & (state.round == 0),preparing,mask)
        if self.has_summons:
            mask = mask.at[SHOOT:DEFEND].set(jnp.where(self._actor_summons(state) > 0,
                controlled & self._summon_targets(state)[:6],mask[SHOOT:DEFEND]))
        if self.progression_enabled and self.has_healers:
            post = jnp.zeros(self.num_actions,jnp.bool_).at[SHOOT:DEFEND].set(
                (state.hp[:6] > 0) & ~state.escaped[:6] & (state.actor < 6))
            if self.has_patriarchs:
                post = post.at[SHOOT:DEFEND].set(jnp.where(self._actor_is_patriarch(state),self._patriarch_targets(state)[:6] & (state.actor < 6),post[SHOOT:DEFEND]))
            post = post.at[WAIT].set(state.actor < 6).at[CONTINUE].set(state.actor >= 6)
            mask = jnp.where(state.post_victory > 0,post,mask)
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
        state = state.replace(copied=jnp.zeros(12,bool),summon_owner=jnp.full(12,-1,jnp.int32),
                              preparation=jnp.zeros(12,bool),
                              primary_override=jnp.full(12,-1,jnp.int32),preweak_damage=jnp.full(12,-1,jnp.int32),
                              powerup=jnp.zeros(12,bool),powerup_layered=jnp.zeros(12,bool),
                              saved_primary_override=jnp.full(12,-1,jnp.int32),saved_preweak_damage=jnp.full(12,-1,jnp.int32),
                              saved_powerup=jnp.zeros(12,bool),saved_powerup_layered=jnp.zeros(12,bool),
                              healer_wards=jnp.zeros((12,4),jnp.uint32),ward_native_used=jnp.zeros(12,jnp.uint32),
                              saved_healer_wards=jnp.zeros((12,4),jnp.uint32),saved_ward_native_used=jnp.zeros(12,jnp.uint32),
                              fenrir=jnp.zeros(12, jnp.bool_), armor_shreds=jnp.zeros(12, jnp.int32),
                              imp=jnp.zeros(12, jnp.bool_), imp_wards=jnp.zeros(12, jnp.uint32),
                              activation_done=jnp.zeros(12, jnp.bool_),
            post_victory=jnp.int32(0),pending_healers=jnp.zeros(12,jnp.bool_),
            initiative_override=jnp.full(12,-1,jnp.int32),saved_initiative_override=jnp.full(12,-1,jnp.int32),
            slow_original=jnp.full(12,-1,jnp.int32),waited=jnp.zeros(12,bool),bonus_turns=jnp.zeros(12,jnp.int32),
            battle_revived=jnp.zeros(12,bool),revival_xp_cutoff=jnp.zeros(12,jnp.int32),
            feared=jnp.zeros(12,jnp.bool_),enemy_initial_hp=jnp.zeros(6,jnp.int32),
            enemy_damage_credit=jnp.zeros(6,jnp.int32),enemy_killed_credit=jnp.zeros(6,jnp.bool_),
            weakened=jnp.zeros(12,jnp.bool_),saved_weakened=jnp.zeros(12,jnp.bool_),
            paralyzed=jnp.zeros(12,jnp.bool_),long_paralyzed=jnp.zeros(12,jnp.bool_),
                              decay_form=jnp.zeros(12, jnp.int32), decay_shreds=jnp.zeros(12, jnp.int32),
                              water_turns=jnp.zeros(12,jnp.int32), water_damage=jnp.zeros(12,jnp.int32),
                              water_source=jnp.full(12,-1,jnp.int32), last_water_damage=jnp.zeros(12,jnp.int32),
                              burn_turns=jnp.zeros(12,jnp.int32),burn_damage=jnp.zeros(12,jnp.int32),
                              burn_source=jnp.full(12,-1,jnp.int32),last_burn_damage=jnp.zeros(12,jnp.int32))
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
        state = state.replace(
            battle_key=key, in_battle=jnp.bool_(True), hp=hp, priority=priority,
            enemy_initial_hp=enemy_hp,
            activation_done=state.activation_done.at[actor].set(True),
            wards_used=jnp.zeros(12, jnp.uint32), last_immune=jnp.uint32(0), last_ward=jnp.uint32(0),
            turn_phase=jnp.zeros(12, jnp.int32), defended=jnp.zeros(12, bool),
            retreating=jnp.zeros(12, bool), escaped=jnp.zeros(12, bool),
            actor=actor, round=jnp.int32(1), last_event=jnp.int32(ENGAGE),
            last_actor=jnp.int32(-1), last_target=jnp.int32(-1), last_damage=jnp.int32(0),
        )
        if self.has_copies or self.has_summons:
            state = state.replace(native_ids=state.unit_ids,native_levels=state.unit_levels,native_xp=state.unit_xp)
        if self.has_copies:
            pending = self.progression.doppelgangers[state.unit_ids] & (hp > 0)
            # All uncopied doppelgangers are excluded as targets independently of actor.
            targets = (hp > 0) & (self.progression.sizes[state.unit_ids] == 1) & ~self.progression.copy_forbidden[state.unit_ids] & ~pending
            pending &= jnp.any(targets)
            preparing = jnp.any(pending)
            state = state.replace(preparation=pending,round=jnp.where(preparing,0,1),
                actor=jnp.where(preparing,jnp.argmax(jnp.where(pending,priority,-1e9)),actor).astype(jnp.int32),
                activation_done=jnp.where(preparing,False,state.activation_done))
        return state

    def _world_step(self, state, action, key, random_values):
        state = state.replace(last_poison_damage=jnp.zeros(12, jnp.int32),
                              last_item_loot=jnp.zeros_like(state.last_item_loot),
                              last_water_damage=jnp.zeros(12,jnp.int32),
                              last_burn_damage=jnp.zeros(12,jnp.int32))
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
            if self.items_enabled:
                rest_healing = (rest_max*(1+self.item_rules.value(state,'regeneration',2))+9)//10
            building = jnp.clip(action - BUILD_START, 0, BUILD_SLOTS - 1)
            building_action = (action >= BUILD_START) & (action < REST)
            next_state = next_state.replace(
                gold=gold-jnp.where(building_action, self.construction.costs[building], 0),
                buildings=state.buildings | jnp.where(building_action, self.construction.bits[building], jnp.uint32(0)),
                blocked_buildings=state.blocked_buildings | jnp.where(
                    building_action, self.construction.blocks[building], jnp.uint32(0)),
                built_today=~resting & (state.built_today | building_action),
                movement_points=jnp.where(resting, self.movement_cap(state),
                                          jnp.maximum(0, state.movement_points - jnp.where(
                                              engage, (self.movement_cap(state)+1)//2, moved.astype(jnp.int32) * MOVE_COST))),
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
            next_state = self.potion_rules.expire(next_state,resting)
            potion_slot=jnp.clip(action-POTION_START,0,max(0,self.potion_rules.action_count-1))%6
            next_state = self.potion_rules.apply(next_state, action, rest_max,
                self._potion_intrinsic(state.unit_ids[potion_slot],state.unit_levels[potion_slot]))
        if self.chests_enabled:
            next_state = self.chest_rules.collect(next_state, moved)
        if self.items_enabled:
            changed=resting | jnp.any(next_state.last_item_loot>0) | (next_state.last_potion>=0) | (action>=REVIVE_START)&(action<REVIVE_START+6)
            next_state=jax.lax.cond(changed,self._refresh_equipment,lambda s:s,next_state)
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
            traits = self._combat_traits(state)
            source = jnp.left_shift(jnp.uint32(1), jnp.maximum(traits[state.actor, 1], 1)-1)
            # Python reference: ordinary attacks choose the lowest-HP reachable
            # non-immune opponent. Wards do not disqualify a target. Area spells
            # still act when everyone is immune; the attack resolves each target.
            valid &= (traits[state.actor, 0] == AREA) | ((traits[:6, 2] & source) == 0)
        elif self.has_enemy_warriors:
            warrior = self._enemy_is_warrior(state)
            valid &= ~warrior | self._melee_targets(state, enemy_side=True)
            base_damage = jnp.where(warrior, self.warrior_damage, base_damage)
        damage = jnp.where(state.defended[:6], (base_damage + 1) // 2, base_damage)
        kill = valid & (state.hp[:6] <= damage)
        candidates = valid if self.basic_combat else valid & jnp.where(jnp.any(kill), kill, True)
        # Uniform random keys break equal-HP ties without consulting hit rolls.
        score = jnp.where(candidates, state.hp[:6] + random_values[:6] * .5, 1e9)
        attack = SHOOT + jnp.argmin(score).astype(jnp.int32)
        action = jnp.where(jnp.any(valid), attack, DEFEND) if self.basic_combat or self.has_enemy_warriors else attack
        if self.has_secondary_paralysis:
            candidates = valid & ~state.paralyzed[:6] & ~state.long_paralyzed[:6]
            # Reference ranks stored damage, before the battle damage cap.
            raw = self._combat_stats(state,cap=False)[:6,DAMAGE]
            score = jnp.where(candidates,-raw+random_values[:6]*.5,1e9)
            smart = jnp.where(jnp.any(candidates),SHOOT+jnp.argmin(score),action)
            action = jnp.where((self._actor_secondary_paralysis(state) == 2) & (traits[state.actor,0] != AREA),smart,action)
        if self.has_witches or self.has_paralysis or self.has_fear:
            living = (state.hp[:6] > 0) & ~state.escaped[:6]
            weapon_immune = living & ((self._combat_traits(state)[:6, 2] & jnp.uint32(1)) != 0)
            prefer_immune = jnp.any(weapon_immune) & ~self._actor_is_fear_caster(state)
            candidates = jnp.where(prefer_immune, weapon_immune, living)
            score = jnp.where(candidates, jnp.where(prefer_immune, 0., -state.hp[:6]) + random_values[:6]*.5, 1e9)
            witch_target = jnp.where(jnp.any(living), SHOOT+jnp.argmin(score), DEFEND)
            action = jnp.where(self._actor_is_witch(state) | (self._actor_ghost_mode(state) > 0) | self._actor_is_fear_caster(state), witch_target, action)
        if self.basic_combat and self.has_healers:
            # Reference: lowest absolute HP among wounded living allies, self included.
            wounded = (state.hp[6:] > 0) & ~state.escaped[6:] & (state.hp[6:] < self._combat_stats(state)[6:, HP])
            score = jnp.where(wounded, state.hp[6:] + random_values[:6] * .5, 1e9)
            heal = jnp.where(jnp.any(wounded), SHOOT + jnp.argmin(score).astype(jnp.int32), DEFEND)
            heal = jnp.where(self._actor_mass_healer(state),SHOOT,heal)
            action = jnp.where(self._actor_is_healer(state), heal, action)
            if self.has_powerups:
                living = (state.hp[6:] > 0) & ~state.escaped[6:] & (self.slots[6:] != state.actor)
                raw = self._combat_stats(state,cap=False)[6:,DAMAGE]
                score = jnp.where(living,-raw+random_values[:6]*.5,1e9)
                buff = jnp.where(jnp.any(living),SHOOT+jnp.argmin(score),DEFEND)
                action = jnp.where(self._actor_power_factor(state) > 0,buff,action)
        if self.has_patriarchs:
            valid = self._patriarch_targets(state)[6:]
            dead = valid & (state.hp[6:] <= 0)
            wounded = valid & (state.hp[6:] > 0) & (state.hp[6:] < self._combat_stats(state)[6:,HP])
            targets = jnp.where(jnp.any(dead),dead,wounded)
            score = jnp.where(targets,jnp.where(jnp.any(dead),0.,state.hp[6:])+random_values[:6]*.5,1e9)
            support = jnp.where(jnp.any(targets),SHOOT+jnp.argmin(score),DEFEND)
            action = jnp.where(self._actor_is_patriarch(state),support,action)
        if self.has_alchemists:
            raw = self._combat_stats(state,cap=False)[6:]
            current = jnp.where(state.turn_phase[6:] == 0,state.priority[6:],0.)
            valid = self._alchemist_targets(state)[6:] & (current < raw[:,INITIATIVE]) & (raw[:,DAMAGE] > 0)
            score = jnp.where(valid,-raw[:,DAMAGE]+random_values[:6]*.5,1e9)
            support = jnp.where(jnp.any(valid),SHOOT+jnp.argmin(score),DEFEND)
            action = jnp.where(self._alchemist_units(state)[state.actor],support,action)
        if self.has_copies or self.has_summons:
            action = self._special_enemy_action(state,action,random_values)
        return action

    def _cleanse(self,state,hp,wards_used,mask):
        """Cure living recipients, restoring original forms before healing."""
        if not self.has_cures:
            return state,hp,wards_used
        state,priority = self._restore_slow(state,state.priority,state.turn_phase,mask)
        state = state.replace(priority=priority)
        transformed = mask & (state.imp | (state.decay_form > 0))
        state = self._restore_initiative_form(state,transformed)
        natural = (self._natural_stats(state) if self.progression_enabled
                   else self.stats_table[state.enemy])
        if self.has_copies:
            natural = jnp.where(state.copied[:,None],state.copy_stats,natural)
        maximum = jnp.where(state.fenrir,275.,natural[:,HP])
        if self.has_wights:
            current_max = self.progression.base_stats[state.decay_form,HP]
            restored = jnp.maximum(1,jnp.rint(hp/jnp.maximum(current_max,1)*maximum).astype(jnp.int32))
            hp = jnp.where(transformed & (state.decay_form > 0),restored,hp)
        wards_used = jnp.where(transformed,state.imp_wards,wards_used)
        state = self._restore_damage_forms(state,transformed)
        if self.has_powerups:
            weak = jnp.where(transformed,state.saved_weakened,state.weakened)
            original = jnp.where(transformed,self._original_primary(state,saved=True),self._original_primary(state))
            restored = jnp.where(state.preweak_damage >= 0,state.preweak_damage,original)
            state = state.replace(primary_override=jnp.where(mask & weak,restored,state.primary_override),
                powerup_layered=state.powerup_layered & ~mask)
        state = self._restore_ward_forms(state,transformed)
        priority = jnp.where(transformed & (state.turn_phase < 2) & (state.turn_phase != 1),
            jnp.where(state.initiative_override >= 0,state.initiative_override,jnp.where(state.fenrir,65.,natural[:,INITIATIVE])),state.priority)
        updates = dict(imp=state.imp & ~mask,decay_form=jnp.where(mask,0,state.decay_form),
            decay_shreds=jnp.where(mask,0,state.decay_shreds),armor_shreds=jnp.where(mask,0,state.armor_shreds),
            weakened=state.weakened & ~mask,saved_weakened=state.saved_weakened & ~mask,
            paralyzed=state.paralyzed & ~mask,long_paralyzed=state.long_paralyzed & ~mask,
            retreating=state.retreating & ~(mask & state.feared),feared=state.feared & ~mask,
            priority=priority)
        for kind in ('poison','water','burn'):
            updates[kind+'_turns'] = jnp.where(mask,0,getattr(state,kind+'_turns'))
            updates[kind+'_damage'] = jnp.where(mask,0,getattr(state,kind+'_damage'))
            updates[kind+'_source'] = jnp.where(mask,-1,getattr(state,kind+'_source'))
        return state.replace(**updates),hp,wards_used

    def _battle_step(self, state, action, key, random_values):
        if self.has_copies:
            return jax.lax.cond(state.round == 0,self._prepare_copies,self._battle_step_regular,
                                state,action,key,random_values)
        return self._battle_step_regular(state,action,key,random_values)

    def _battle_step_regular(self, state, action, key, random_values):
        actor = state.actor
        post_turn = state.post_victory > 0
        skipping = (state.paralyzed[actor] | state.long_paralyzed[actor]) & ~post_turn
        finite_skip = state.paralyzed[actor]
        if self.has_paralysis:
            state = state.replace(paralyzed=state.paralyzed.at[actor].set(state.paralyzed[actor] & post_turn),
                long_paralyzed=state.long_paralyzed.at[actor].set(state.long_paralyzed[actor] & (post_turn | finite_skip | (random_values[67] >= .33))))
        escaping = state.retreating[actor] & ~skipping & ~post_turn
        action = jnp.where(actor >= 6, self._enemy_action(
            state, random_values[30:36] if self.basic_combat else random_values[1:]), action)
        action = jnp.where(escaping | skipping, CONTINUE, action)
        special_event = jnp.int32(-1)
        if self.has_copies or self.has_summons:
            state,action,special_event = self._special_action(state,action,random_values)
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
        transformed_slots = jnp.zeros(6, jnp.bool_)
        decayed_slots = jnp.zeros(6, jnp.bool_)
        paralyzed_slots = jnp.zeros(6,jnp.bool_)
        feared_slots = jnp.zeros(6,jnp.bool_)
        damage_credit = jnp.zeros(6,jnp.int32)
        revived = jnp.zeros(12,bool)
        extra_turn = jnp.zeros(12,bool)
        slowed = jnp.zeros(6,bool)
        if self.basic_combat:
            effective_ids = self._effective_ids(state) if self.progression_enabled else None
            effective_levels = self._effective_levels(state) if self.progression_enabled else None
            stats = self._combat_stats(state)
            target_slots = jnp.arange(6) + jnp.where(actor < 6, 6, 0)
            traits = self._combat_traits(state)
            area_attack = traits[actor, 0] == AREA
            targets = ((area_attack | (target_slots == target))
                       & (state.hp[target_slots] > 0) & ~state.escaped[target_slots])
            accuracy = stats[actor, ACCURACY]
            if self.has_aoe_accuracy_falloff:
                falloff = (self.progression.aoe_accuracy_falloff[effective_ids[actor]]
                           if self.progression_enabled else self.aoe_accuracy_falloff[state.enemy, actor])
                index = jnp.maximum(jnp.cumsum(targets.astype(jnp.int32))-1, 0)
                accuracy = jnp.maximum(0., accuracy-jnp.where(area_attack & falloff, index*10, 0))
            hits = accuracy_hits(accuracy, random_values[12:18], random_values[18:24])
            bonuses = jnp.minimum((random_values[24:30] * 6).astype(jnp.int32), 5)
            if self.progression_enabled:
                damage = self.progression.damage(
                    state.replace(unit_ids=effective_ids, unit_levels=effective_levels), actor, target_slots, state.defended[target_slots], bonuses,
                                                 stats[actor, DAMAGE], self._effective_shreds(state) if self.has_shatterers else None,
                                                 state.imp if self.has_witches else None,
                                                 stats[target_slots,ARMOR] if self.has_potion_buffs or self.items_enabled else None)
            else:
                damage_id = self.damage_ids[state.enemy, actor]
                if self.has_wolf_lord:
                    damage_id = jnp.where(state.fenrir[actor], self.combat_info['fenrir_damage_id'], damage_id)
                if self.has_witches:
                    imp_damage = jnp.where(self.size_table[state.enemy, actor] == 2,
                                          self.combat_info['imp_damage_ids'][1], self.combat_info['imp_damage_ids'][0])
                    damage_id = jnp.where(state.imp[actor], imp_damage, damage_id)
                armor_id = (self.shatter_armor_ids[state.enemy, target_slots, jnp.minimum(state.armor_shreds[target_slots], 6)]
                            if self.has_shatterers else self.armor_ids[state.enemy, target_slots])
                if self.has_witches:
                    armor_id = jnp.where(state.imp[target_slots], 0, armor_id)
                damage = self.damage_rolls[damage_id, armor_id,
                                           state.defended[target_slots].astype(jnp.int32), bonuses]
            if self.has_weakening and not self.progression_enabled:
                amount = (stats[actor,DAMAGE]+bonuses)*(1.-jnp.minimum(stats[target_slots,ARMOR],90.)/100.)
                damage = jnp.where(stats[actor,DAMAGE] > 0,jnp.rint(amount*jnp.where(state.defended[target_slots],.5,1.)),0).astype(jnp.int32)
            connected = attack & ~healer & targets & hits
            ghost_mode = self._actor_ghost_mode(state)
            # AOE status-only attacks reject an existing matching flag before
            # checking wards; single Ghost attacks keep their primary hit path.
            already = jnp.where(ghost_mode == 2,state.long_paralyzed[target_slots],state.paralyzed[target_slots])
            connected &= ~(area_attack & (ghost_mode > 0) & already)
            effective = connected
            check_primary = connected & ~(area_attack & self._actor_is_witch(state))
            if self.has_protections:
                source_bit = jnp.left_shift(jnp.uint32(1), jnp.maximum(traits[actor, 1], 1)-1)
                immune = check_primary & ((traits[target_slots, 2] & source_bit) != 0)
                warded = check_primary & ~immune & ((traits[target_slots, 3] &
                                                  ~wards_used[target_slots] & source_bit) != 0)
                wards_used = wards_used.at[target_slots].set(
                    wards_used[target_slots] | jnp.where(warded, source_bit, jnp.uint32(0)))
                slot_bits = jnp.left_shift(jnp.uint32(1), target_slots.astype(jnp.uint32))
                immune_slots = jnp.sum(jnp.where(immune, slot_bits, jnp.uint32(0)))
                ward_slots = jnp.sum(jnp.where(warded, slot_bits, jnp.uint32(0)))
                effective &= ~immune & ~warded
            ghost_mode = self._actor_ghost_mode(state)
            damage = jnp.where(self._actor_is_witch(state) | (ghost_mode > 0) | self._actor_is_fear_caster(state), 0, damage)
            removed = jnp.where(effective, jnp.minimum(damage, state.hp[target_slots]), 0)
            if self.has_centaurs:
                centaur = (self.progression.centaurs[effective_ids[actor]] if self.progression_enabled
                           else self.centaurs[state.enemy, actor]) & ~state.imp[actor]
                # Reference: a connected hit adds 5% of effective base damage
                # only to a survivor, bypassing armour, defend and secondary
                # POWER/source checks. User: round this addition to integer HP.
                extra = jnp.rint(stats[actor, DAMAGE] / 20.).astype(jnp.int32)
                extra = jnp.minimum(extra, state.hp[target_slots]-removed)
                removed += jnp.where(centaur & effective, extra, 0)
            damage_credit = jnp.where(actor < 6,removed,0)
            if self.has_paralysis:
                # Ghost uses its primary source and primary accuracy only; those
                # immunity/ward checks were already applied to effective.
                paralyzed_slots = effective & (ghost_mode > 0) & ~jnp.where(ghost_mode == 2,state.long_paralyzed[target_slots],state.paralyzed[target_slots])
                state = state.replace(
                    paralyzed=state.paralyzed.at[target_slots].set(state.paralyzed[target_slots] | (paralyzed_slots & (ghost_mode == 1))),
                    long_paralyzed=state.long_paralyzed.at[target_slots].set(state.long_paralyzed[target_slots] | (paralyzed_slots & (ghost_mode == 2))))
            zeros = jnp.zeros(6, jnp.int32)
            hp = state.hp - jnp.where(actor < 6, jnp.concatenate((zeros, removed)),
                                     jnp.concatenate((removed, zeros)))
            applied = jnp.sum(removed)
            leech_mode = jnp.int32(0)
            if self.has_leech:
                leech_mode = (self.progression.leech_modes[effective_ids[actor]] if self.progression_enabled
                              else self.leech_modes[state.enemy,actor])
                pool = jnp.where((leech_mode > 0) & ~state.imp[actor], applied//2, 0)
                hp = vampiric_heal(hp,stats[:,HP],state.escaped,actor,pool,leech_mode == 2)
            hit = attack & jnp.any(targets & hits)
            if self.has_healers:
                # Mass healing excludes the caster. Only the two native Profit
                # cure forms cleanse; the ordinary Cleric does not remove effects.
                mass = self._actor_mass_healer(state)
                allies = (self.slots < 6) == (actor < 6)
                recipients = allies & (hp > 0) & ~state.escaped & jnp.where(mass,self.slots != actor,self.slots == target)
                cure_caster = (self.progression.mass_cures[effective_ids[actor]] if self.progression_enabled
                               else self.mass_cures[state.enemy,actor])
                cleanse = recipients & attack & healer & cure_caster
                state,hp,wards_used = self._cleanse(state,hp,wards_used,cleanse)
                # Cure can restore a Wight form with a different maximum HP.
                maximum = self._combat_stats(state)[:,HP] if self.has_cures else stats[:,HP]
                healed = jnp.where(attack & healer & recipients,
                    jnp.minimum(stats[actor,DAMAGE],jnp.maximum(maximum-hp,0)),0).astype(jnp.int32)
                hp += healed
                if self.has_alchemists:
                    extra_turn = self._alchemist_targets(state) & (self.slots == target) & attack & self._alchemist_units(state)[actor]
                    state = state.replace(turn_phase=jnp.where(extra_turn,0,state.turn_phase),
                        priority=jnp.where(extra_turn,stats[:,INITIATIVE],state.priority),
                        bonus_turns=state.bonus_turns+extra_turn.astype(jnp.int32))
                if self.has_patriarchs:
                    revived = self._patriarch_targets(state) & (state.hp <= 0) & (self.slots == target) & attack & self._actor_is_patriarch(state)
                    restored = jnp.maximum(1,jnp.rint(stats[:,HP]*.5)).astype(jnp.int32)
                    hp = jnp.where(revived,restored,hp)
                    state = state.replace(battle_revived=state.battle_revived | revived,
                        revival_xp_cutoff=jnp.where(revived,state.battle_xp[jnp.where(self.slots < 6,1,0)],state.revival_xp_cutoff),
                        turn_phase=jnp.where(revived,2,state.turn_phase),priority=jnp.where(revived,0.,state.priority),
                        retreating=state.retreating & ~revived)
                    state,hp,wards_used = self._cleanse(state,hp,wards_used,revived)
                    healed += jnp.where(revived,hp,0)
                if self.has_powerups:
                    factor = self._actor_power_factor(state)
                    buffed = recipients & attack & healer & (factor > 0)
                    power_cure = (self.progression.power_cures[effective_ids[actor]] if self.progression_enabled
                                  else self.power_cures[state.enemy,actor])
                    state,hp,wards_used = self._cleanse(state,hp,wards_used,buffed & power_cure)
                    original = self._original_primary(state)
                    buffed &= (original > 0) & (self.slots != actor)
                    amount = jnp.rint(original*factor).astype(jnp.int32)
                    state = state.replace(primary_override=jnp.where(buffed,amount,state.primary_override),
                        powerup=state.powerup | buffed,powerup_layered=state.powerup_layered & ~buffed)
                if self.has_healer_wards:
                    elements = (self.progression.healer_ward_elements[effective_ids[actor]] if self.progression_enabled
                                else self.healer_ward_elements[state.enemy,actor])
                    granted = recipients & attack & healer & (stats[actor,DAMAGE] > 0)
                    owners,sticky,wards_used = grant_wards(state.healer_wards,state.ward_native_used,wards_used,
                        self._base_combat_traits(state)[:,3],granted,elements,actor)
                    state = state.replace(healer_wards=owners,ward_native_used=sticky)
                applied = jnp.where(healer,jnp.sum(healed),applied)
                if self.has_powerups:
                    applied = jnp.where(attack & (factor > 0),amount[target],applied)
                area_attack |= healer & mass
                hit |= attack & healer
            if self.has_shatterers or self.has_witches or self.has_wights or self.has_poisoners or self.has_water or self.has_fire or self.has_secondary_paralysis or self.has_weakening or self.has_fear or self.has_hermits:
                secondary = (self.progression.secondary_sources[effective_ids[actor]] if self.progression_enabled
                             else self.secondary_sources[state.enemy, actor])
                guards = (self.progression.capital_guards[state.unit_ids] if self.progression_enabled
                          else self.capital_guards[state.enemy])
                secondary_fear = jnp.bool_(False)
                hermit = jnp.bool_(False)
                shatterer = jnp.bool_(False)
                wight = jnp.bool_(False)
                poisoner = jnp.bool_(False)
                water_caster = jnp.bool_(False)
                fire_caster = jnp.bool_(False)
                cached = jnp.bool_(False)
                paralysis_mode = self._actor_secondary_paralysis(state)
                weakener = ((self.progression.weakeners[effective_ids[actor]] if self.progression_enabled else self.weakeners[state.enemy,actor]) & ~state.imp[actor]) if self.has_weakening else jnp.bool_(False)
                if self.has_shatterers:
                    shatterer = (self.progression.shatterers[effective_ids[actor]] if self.progression_enabled
                                 else self.shatterers[state.enemy, actor]) & ~state.imp[actor]
                witch = self._actor_is_witch(state)
                fear_caster = self._actor_is_fear_caster(state)
                eligible = ((shatterer & (removed > 0) & (hp[target_slots] > 0))
                            | (witch & effective)) & ~guards[target_slots]
                eligible |= fear_caster & effective & ~state.retreating[target_slots]
                if self.has_poisoners or self.has_water or self.has_fire or self.has_wights or self.has_secondary_paralysis or self.has_weakening or self.has_hermits or self.has_secondary_fear:
                    secondary_stats = (self._secondary_stats(state,actor)
                                       if self.progression_enabled else self.secondary_values[state.enemy, actor])
                    status_hit = accuracy_hits(secondary_stats[1], random_values[37:43], random_values[43:49])
                    connected_status = effective & status_hit & (hp[target_slots] > 0)
                    eligible |= weakener & connected_status
                    if self.has_secondary_fear:
                        secondary_fear = (self.progression.secondary_fear[effective_ids[actor]] if self.progression_enabled else self.secondary_fear[state.enemy,actor]) & ~state.imp[actor]
                        eligible |= secondary_fear & (secondary_stats[1] > 0) & connected_status & ~state.retreating[target_slots]
                    if self.has_secondary_paralysis:
                        eligible |= (paralysis_mode > 0) & (secondary_stats[1] > 0) & connected_status & ~state.long_paralyzed[target_slots]
                    if self.has_wights:
                        wight = self.progression.wights[effective_ids[actor]] & ~state.imp[actor]
                        eligible |= wight & connected_status & ~self.progression.neutrals[state.unit_ids[target_slots]]
                    if self.has_poisoners:
                        poisoner = (self.progression.poisoners[effective_ids[actor]] if self.progression_enabled
                                    else self.poisoners[state.enemy, actor]) & ~state.imp[actor]
                        cached = (self.progression.cached_poisoners[effective_ids[actor]] if self.progression_enabled
                                  else self.cached_poisoners[state.enemy, actor])
                        locked = jnp.any((state.poison_source == actor) & (state.poison_turns > 0) & (hp > 0) & ~state.escaped)
                        eligible |= poisoner & (~cached | ~locked) & connected_status & (state.poison_turns[target_slots] == 0) & ((leech_mode != 1) | (removed > 0))
                    if self.has_water:
                        water_caster = (self.progression.water_casters[effective_ids[actor]] if self.progression_enabled
                                        else self.water_casters[state.enemy,actor]) & ~state.imp[actor]
                        cached_water = (self.progression.cached_water[effective_ids[actor]] if self.progression_enabled
                                        else self.cached_water[state.enemy,actor])
                        water_locked = jnp.any((state.water_source == actor) & (state.water_turns > 0) & (hp > 0) & ~state.escaped)
                        eligible |= water_caster & (~cached_water | ~water_locked) & connected_status & (state.water_turns[target_slots] == 0)
                    if self.has_fire:
                        fire_caster = (self.progression.fire_casters[effective_ids[actor]] if self.progression_enabled
                                       else self.fire_casters[state.enemy,actor]) & ~state.imp[actor]
                        cached_fire = (self.progression.cached_fire[effective_ids[actor]] if self.progression_enabled
                                       else self.cached_fire[state.enemy,actor])
                        fire_locked = jnp.any((state.burn_source == actor) & (state.burn_turns > 0) & (hp > 0) & ~state.escaped)
                        eligible |= fire_caster & (~cached_fire | ~fire_locked) & connected_status & (state.burn_turns[target_slots] == 0)
                if self.has_hermits:
                    hermit = (self.progression.hermits[effective_ids[actor]] if self.progression_enabled else self.hermits[state.enemy,actor]) & ~state.imp[actor]
                    eligible |= hermit & connected_status
                # Mutually exclusive caster types share the same source/ward
                # pipeline; only Teurg preserves an explicitly absent source.
                effect_source = jnp.where(shatterer | weakener | hermit | (paralysis_mode > 0) | (secondary > 0), secondary, traits[actor, 1])
                affected, wards_used, blocked, warded = source_protection(
                    eligible, effect_source, traits, wards_used, target_slots)
                immune_slots |= blocked
                ward_slots |= warded
                if self.has_hermits:
                    slowed = affected & hermit & (state.slow_original[target_slots] < 0)
                    base = self._combat_stats(state)[target_slots,INITIATIVE].astype(jnp.int32)
                    lowered = jnp.rint(base*.5).astype(jnp.int32)
                    state = state.replace(slow_original=state.slow_original.at[target_slots].set(jnp.where(slowed,base,state.slow_original[target_slots])),
                        initiative_override=state.initiative_override.at[target_slots].set(jnp.where(slowed,lowered,state.initiative_override[target_slots])),
                        priority=state.priority.at[target_slots].set(jnp.where(slowed,jnp.minimum(state.priority[target_slots],lowered),state.priority[target_slots])))
                if self.has_fear:
                    feared_slots = affected & (fear_caster | secondary_fear)
                    protected = self.fear_protected[target_slots]
                    paralyzed_slots |= feared_slots & protected & ~state.paralyzed[target_slots]
                    state = state.replace(
                        paralyzed=state.paralyzed.at[target_slots].set(state.paralyzed[target_slots] | (feared_slots & protected)),
                        retreating=state.retreating.at[target_slots].set(state.retreating[target_slots] | (feared_slots & ~protected)),
                        feared=state.feared.at[target_slots].set(state.feared[target_slots] | (feared_slots & ~protected)))
                if self.has_weakening:
                    if self.has_powerups:
                        newly = affected & weakener & ~state.weakened[target_slots]
                        raw = self._combat_stats(state,cap=False)[target_slots,DAMAGE].astype(jnp.int32)
                        state = state.replace(
                            primary_override=state.primary_override.at[target_slots].set(jnp.where(newly,jnp.rint(raw*.68).astype(jnp.int32),state.primary_override[target_slots])),
                            preweak_damage=state.preweak_damage.at[target_slots].set(jnp.where(newly,raw,state.preweak_damage[target_slots])),
                            powerup_layered=state.powerup_layered.at[target_slots].set(jnp.where(newly,state.powerup[target_slots],state.powerup_layered[target_slots])))
                    state = state.replace(weakened=state.weakened.at[target_slots].set(state.weakened[target_slots] | (affected & weakener)))
                if self.has_secondary_paralysis:
                    secondary_paralyzed = affected & (paralysis_mode > 0)
                    paralyzed_slots |= secondary_paralyzed
                    state = state.replace(
                        paralyzed=state.paralyzed.at[target_slots].set(state.paralyzed[target_slots] | (secondary_paralyzed & (paralysis_mode == 1))),
                        long_paralyzed=state.long_paralyzed.at[target_slots].set(state.long_paralyzed[target_slots] | (secondary_paralyzed & (paralysis_mode == 2))))
                if self.has_shatterers:
                    shreds = jnp.minimum(6, self._effective_shreds(state)[target_slots] +
                        (affected & shatterer & (stats[target_slots, ARMOR] > 0)).astype(jnp.int32))
                    lowered = state.decay_form[target_slots] > 0
                    state = state.replace(
                        armor_shreds=state.armor_shreds.at[target_slots].set(jnp.where(lowered, state.armor_shreds[target_slots], shreds)),
                        decay_shreds=state.decay_shreds.at[target_slots].set(jnp.where(lowered, shreds, state.decay_shreds[target_slots])))
                if self.has_witches:
                    transformed_slots = affected & witch
                    state,wards_used=transform_imp(self,state,wards_used,target_slots,transformed_slots)
                if self.has_wights:
                    state,hp,wards_used,decayed_slots=transform_decay(
                        self,state,hp,wards_used,target_slots,affected & wight)
                if self.has_poisoners:
                    poisoned = affected & poisoner
                    duration = jnp.minimum((random_values[49:55]*6).astype(jnp.int32), 5)+1
                    state = state.replace(
                        poison_turns=state.poison_turns.at[target_slots].set(jnp.where(poisoned, duration, state.poison_turns[target_slots])),
                        poison_damage=state.poison_damage.at[target_slots].set(jnp.where(poisoned, secondary_stats[0].astype(jnp.int32), state.poison_damage[target_slots])),
                        poison_source=state.poison_source.at[target_slots].set(jnp.where(poisoned, jnp.where(cached, actor, -1), state.poison_source[target_slots])))
                if self.has_water:
                    soaked = affected & water_caster
                    duration = jnp.minimum((random_values[49:55]*6).astype(jnp.int32),5)+1
                    state = state.replace(
                        water_turns=state.water_turns.at[target_slots].set(jnp.where(soaked,duration,state.water_turns[target_slots])),
                        water_damage=state.water_damage.at[target_slots].set(jnp.where(soaked,secondary_stats[0].astype(jnp.int32),state.water_damage[target_slots])),
                        water_source=state.water_source.at[target_slots].set(jnp.where(soaked,jnp.where(cached_water,actor,-1),state.water_source[target_slots])))
                if self.has_fire:
                    burning = affected & fire_caster
                    duration = jnp.minimum((random_values[49:55]*6).astype(jnp.int32),5)+1
                    state = state.replace(
                        burn_turns=state.burn_turns.at[target_slots].set(jnp.where(burning,duration,state.burn_turns[target_slots])),
                        burn_damage=state.burn_damage.at[target_slots].set(jnp.where(burning,secondary_stats[0].astype(jnp.int32),state.burn_damage[target_slots])),
                        burn_source=state.burn_source.at[target_slots].set(jnp.where(burning,jnp.where(cached_fire,actor,-1),state.burn_source[target_slots])))
            if self.items_enabled and (self.item_rules.status_items or self.item_rules.has_drain):
                connected=attack & ~area_attack & jnp.any(effective)
                state,hp,wards_used,item_immune,item_ward=equipment_attack(
                    self,state,hp,wards_used,target,connected,applied,random_values)
                immune_slots |= item_immune
                ward_slots |= item_ward
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
        transformed = (action == FENRIR) & self._actor_is_wolf_lord(state)
        if self.has_wolf_lord:
            new_hp = jnp.rint(state.hp[actor] * 275. / jnp.maximum(self._combat_stats(state)[actor, HP], 1)).astype(jnp.int32)
            hp = hp.at[actor].set(jnp.where(transformed, new_hp, hp[actor]))
            if self.has_hermits:
                state = state.replace(initiative_override=state.initiative_override.at[actor].set(jnp.where(transformed,65,state.initiative_override[actor])))
            if self.has_powerups:
                state = state.replace(primary_override=state.primary_override.at[actor].set(jnp.where(transformed,90,state.primary_override[actor])),
                    preweak_damage=state.preweak_damage.at[actor].set(jnp.where(transformed,-1,state.preweak_damage[actor])),
                    powerup_layered=state.powerup_layered.at[actor].set(state.powerup_layered[actor] & ~transformed))
            state = state.replace(fenrir=state.fenrir.at[actor].set(state.fenrir[actor] | transformed),
                weakened=state.weakened.at[actor].set(state.weakened[actor] & ~transformed))
        another_strike = attack & self._actor_double_strike(state) & ~state.second_strike & ~escaping
        if self.has_summons:
            hp = self._linked_summon_hp(state,hp)
        state = state.replace(waited=state.waited.at[actor].set(state.waited[actor] | (action == WAIT)))
        state = self._expire_powerups(state,(self.slots == actor) & ~another_strike)
        phases = state.turn_phase.at[actor].set(jnp.where(another_strike, state.turn_phase[actor], jnp.where(action == WAIT, 1, 2)))
        defended = state.defended.at[actor].set((action == DEFEND) & (special_event < 0))
        retreating = state.retreating.at[actor].set(jnp.where(skipping,state.retreating[actor],action == RETREAT))
        escaped = state.escaped.at[actor].set(escaping)
        if self.items_enabled:
            instant=(action==RETREAT)&(actor==self._equipment_hero(state))&jnp.any(self.item_rules.value(state,'instant_retreat')[:2])
            escaped=escaped.at[actor].set(escaped[actor]|instant)
            retreating=retreating.at[actor].set(retreating[actor]&~instant)
        active = (hp > 0) & ~escaped
        normal, waiting = active & (phases == 0), active & (phases == 1)
        new_round = ~jnp.any(normal | waiting)
        phases = jnp.where(new_round, jnp.zeros(12, jnp.int32), phases)
        priority = jnp.where(new_round, self._round_priority(random_values, state.enemy, state), state.priority)
        round_number = state.round + new_round.astype(jnp.int32)
        scores = jnp.where(active & (phases == 0), priority,
                            jnp.where(active & (phases == 1), -priority, -1e9 if self.basic_combat else -100))
        next_actor = jnp.where(another_strike, actor, jnp.argmax(scores)).astype(jnp.int32)
        defended = defended.at[next_actor].set(False)
        state = state.replace(activation_done=jnp.where(new_round, False, state.activation_done),
                              last_poison_damage=jnp.zeros(12, jnp.int32),
                              last_water_damage=jnp.zeros(12,jnp.int32),
                              last_burn_damage=jnp.zeros(12,jnp.int32))
        before_queue_activated = state.activation_done
        queue_round = round_number
        visited = jnp.zeros(12,bool).at[next_actor].set(~another_strike & jnp.any(active[:6]) & jnp.any(active[6:]))
        if self.has_summons or self.has_water or self.has_fire:
            # Effects tick in reference order: poison, fire, then water; a lethal
            # earlier effect suppresses later ticks and any following actor.
            turns = jnp.stack((state.poison_turns,state.burn_turns,state.water_turns),axis=1)
            damage = jnp.stack((state.poison_damage,state.burn_damage,state.water_damage),axis=1)
            source = jnp.stack((state.poison_source,state.burn_source,state.water_source),axis=1)
            queue = advance_linked_queue if self.has_summons else advance_periodic_queue
            linked = {'summon_owner':state.summon_owner} if self.has_summons else {}
            (hp,phases,priority,round_number,activated,turns,damage,source,next_actor,losses,visited) = queue(
                hp,escaped,phases,priority,round_number,state.activation_done,turns,damage,source,
                self._combat_traits(state)[:,2],self._combat_stats(state)[:,INITIATIVE]+random_values[55:67]*10,
                next_actor,another_strike,self.max_rounds,jnp.array([16,4,8],jnp.uint32),jnp.array([0,10,10],jnp.int32),**linked)
            state = state.replace(activation_done=activated,poison_turns=turns[:,0],poison_damage=damage[:,0],
                poison_source=source[:,0],last_poison_damage=losses[:,0],
                burn_turns=turns[:,1],burn_damage=damage[:,1],burn_source=source[:,1],last_burn_damage=losses[:,1],
                water_turns=turns[:,2],water_damage=damage[:,2],water_source=source[:,2],last_water_damage=losses[:,2])
            defended = defended.at[next_actor].set(False)
            active = (hp > 0) & ~escaped
        elif self.has_poisoners:
            (hp, phases, priority, round_number, activated, poison_turns, poison_damage,
             poison_source, next_actor, poison_losses, visited) = advance_poison_queue(
                hp, escaped, phases, priority, round_number, state.activation_done,
                state.poison_turns, state.poison_damage, state.poison_source,
                self._combat_traits(state)[:, 2], self._combat_stats(state)[:, INITIATIVE]+random_values[55:67]*10,
                next_actor, another_strike, self.max_rounds)
            state = state.replace(activation_done=activated, poison_turns=poison_turns,
                                  poison_damage=poison_damage, poison_source=poison_source, last_poison_damage=poison_losses)
            defended = defended.at[next_actor].set(False)
            active = (hp > 0) & ~escaped
        state,priority = self._restore_slow(state,priority,phases,visited & (~before_queue_activated | (round_number != queue_round)))
        if self.has_summons:
            hp = self._linked_summon_hp(state,hp)
            active = (hp > 0) & ~escaped
            next_actor = jnp.where(active[next_actor],next_actor,jnp.argmax(jnp.where(
                active & (phases == 0),priority,jnp.where(active & (phases == 1),-priority,-1e9)))).astype(jnp.int32)
        state = state.replace(waited=jnp.where(round_number != state.round,False,state.waited))
        lost = ~jnp.any(hp[:6] > 0)
        victory = ~jnp.any(active[6:]) & ~lost
        withdrawal = ~jnp.any(active[:6]) & ~lost
        timeout = ((round_number > self.max_rounds) | (state.step_count+1 >= self.max_steps)) & ~victory & ~lost & ~withdrawal
        raw_ended = victory | withdrawal | lost | timeout
        pending = jnp.zeros(12,jnp.bool_)
        post_team = jnp.int32(0)
        if self.progression_enabled and self.has_healers:
            eligible = self.progression.post_healers[state.unit_ids] & (hp > 0) & ~escaped
            if self.has_copies or self.has_summons:
                eligible &= ~state.copied & (state.summon_owner < 0)
            eligible &= jnp.where(victory,self.slots < 6,self.slots >= 6) & (victory | withdrawal | lost)
            pending = jnp.where(post_turn,state.pending_healers.at[actor].set(False),eligible)
            # The episode budget remains a hard bound even during this final phase.
            pending &= state.step_count+1 < self.max_steps
            healing = jnp.any(pending)
            post_team = jnp.where(healing,jnp.where(victory,1,2),0).astype(jnp.int32)
            victory &= ~healing
            lost &= ~healing
            withdrawal &= ~healing
            next_actor = jnp.where(healing,jnp.argmax(pending),next_actor).astype(jnp.int32)
        alive = state.alive.at[state.enemy].set(~victory)
        won = ~jnp.any(alive) & ~lost
        back = victory | withdrawal
        event = jnp.where(attack, jnp.where(hit, HIT, MISS),
                            jnp.where(action == DEFEND, GUARD,
                            jnp.where(action == WAIT, DELAY,
                            jnp.where(action == RETREAT, FLEE, ESCAPE))))
        if self.items_enabled:
            event=jnp.where(instant,ESCAPE,event)
        event = jnp.where(attack & healer, HEAL, event)
        event = jnp.where(attack & (self._actor_power_factor(state) > 0),BUFFED,event)
        event = jnp.where(jnp.any(revived),BATTLE_REVIVED,event)
        event = jnp.where(jnp.any(extra_turn),EXTRA_TURN,event)
        event = jnp.where(jnp.any(slowed),SLOWED,event)
        event = jnp.where((post_team > 0) & ~post_turn,POST_HEAL,event)
        event = jnp.where(jnp.any(feared_slots),FEARED,event)
        event = jnp.where(jnp.any(paralyzed_slots),PARALYZED,event)
        event = jnp.where(skipping,PARALYSIS_SKIP,event)
        event = jnp.where(transformed, TRANSFORMED, event)
        event = jnp.where(jnp.any(transformed_slots), IMP_TRANSFORMED, event)
        event = jnp.where(jnp.any(decayed_slots), DECAY_TRANSFORMED, event)
        event = jnp.where(special_event >= 0,special_event,event)
        event = jnp.where(attack & (applied == 0) & (immune_slots != 0), IMMUNE, event)
        event = jnp.where(attack & (applied == 0) & (ward_slots != 0), WARD, event)
        event = jnp.where(victory, VICTORY, jnp.where(lost, DEFEAT,
                            jnp.where(withdrawal, WITHDRAW, jnp.where(timeout, LIMIT, event))))
        progress = {}
        ended_status = victory | withdrawal | lost | timeout
        state = self._restore_initiative_form(state,raw_ended & (state.imp | (state.decay_form > 0)))
        if self.has_hermits:
            state = state.replace(initiative_override=jnp.where(ended_status,-1,state.initiative_override),
                saved_initiative_override=jnp.where(ended_status,-1,state.saved_initiative_override),
                slow_original=jnp.where(ended_status,-1,state.slow_original))
        state = self._restore_damage_forms(state,raw_ended & (state.imp | (state.decay_form > 0)))
        if self.has_powerups:
            updates = {}
            for prefix in ('','saved_'):
                for name,empty in (('primary_override',-1),('preweak_damage',-1),('powerup',False),('powerup_layered',False)):
                    updates[prefix+name] = jnp.where(ended_status,empty,getattr(state,prefix+name))
            state = state.replace(**updates)
        if self.has_healer_wards:
            caster_bits = jnp.sum(jnp.where(visited,jnp.left_shift(jnp.uint32(1),self.slots.astype(jnp.uint32)),jnp.uint32(0)))
            expire = caster_bits | jnp.where(ended_status | (raw_ended & state.fenrir),jnp.uint32(4095),jnp.uint32(0))[:,None]
            state,wards_used = self._expire_healer_wards(state,wards_used,expire)
            # Escaped recipients lose grants, but a caster leaving does not expire other recipients.
            state = state.replace(healer_wards=jnp.where(escaped[:,None],jnp.uint32(0),state.healer_wards),
                saved_healer_wards=jnp.where(escaped[:,None],jnp.uint32(0),state.saved_healer_wards),
                ward_native_used=jnp.where(escaped,jnp.uint32(0),state.ward_native_used),
                saved_ward_native_used=jnp.where(escaped,jnp.uint32(0),state.saved_ward_native_used))
            state = self._restore_ward_forms(state,raw_ended & (state.imp | (state.decay_form > 0)))
        natural_weakening = jnp.where(raw_ended & (state.imp | (state.decay_form > 0)),state.saved_weakened,state.weakened)
        state = state.replace(weakened=natural_weakening & ~ended_status,saved_weakened=state.saved_weakened & ~ended_status,
            feared=state.feared & ~ended_status & (hp > 0) & ~escaped)
        state = state.replace(paralyzed=state.paralyzed & ~ended_status & (hp > 0) & ~escaped,
                              long_paralyzed=state.long_paralyzed & ~ended_status & (hp > 0) & ~escaped)
        if self.has_wights:
            ended = raw_ended
            lowered = state.decay_form > 0
            natural_hp = self._natural_stats(state)[:, HP]
            if self.has_copies:
                natural_hp = jnp.where(state.copied,state.copy_stats[:,HP],natural_hp)
            natural_hp = jnp.where(state.fenrir, 275., natural_hp)
            current_hp = self.progression.base_stats[state.decay_form, HP]
            restored = jnp.rint(hp / jnp.maximum(current_hp, 1) * natural_hp).astype(jnp.int32)
            restored = jnp.where(hp > 0, jnp.maximum(restored, 1), 0)
            hp = jnp.where(ended & lowered, restored, hp)
            wards_used = jnp.where(ended & lowered, state.imp_wards, wards_used)
            state = state.replace(decay_form=jnp.where(ended, 0, state.decay_form),
                                  decay_shreds=jnp.where(ended, 0, state.decay_shreds))
        if self.has_witches:
            ended = raw_ended
            wards_used = jnp.where(ended & state.imp, state.imp_wards, wards_used)
            state = state.replace(imp=state.imp & ~ended,
                                  activation_done=jnp.where(ended, False, state.activation_done))
        if self.has_poisoners:
            ended = victory | withdrawal | lost | timeout
            state = state.replace(poison_turns=jnp.where(ended, 0, state.poison_turns),
                poison_damage=jnp.where(ended, 0, state.poison_damage),
                poison_source=jnp.where(ended, -1, state.poison_source))
        if self.has_water:
            ended = victory | withdrawal | lost | timeout
            state = state.replace(water_turns=jnp.where(ended,0,state.water_turns),
                                  water_damage=jnp.where(ended,0,state.water_damage),
                                  water_source=jnp.where(ended,-1,state.water_source))
        if self.has_fire:
            ended = victory | withdrawal | lost | timeout
            state = state.replace(burn_turns=jnp.where(ended,0,state.burn_turns),
                                  burn_damage=jnp.where(ended,0,state.burn_damage),
                                  burn_source=jnp.where(ended,-1,state.burn_source))
        if self.has_shatterers:
            state = state.replace(armor_shreds=jnp.where(victory | withdrawal | lost | timeout, 0, state.armor_shreds))
        if self.has_wolf_lord:
            # Restore every natural form, including dead/escaped units, before XP.
            ended = raw_ended
            natural = (self._natural_stats(state)[:, HP]
                       if self.progression_enabled else self.stats_table[state.enemy, :, HP])
            if self.has_copies:
                natural = jnp.where(state.copied,state.copy_stats[:,HP],natural)
            restored = jnp.rint(hp / 275. * natural).astype(jnp.int32)
            restored = jnp.where(hp > 0, jnp.maximum(restored, 1), 0)
            hp = jnp.where(ended & state.fenrir, restored, hp)
            state = state.replace(fenrir=state.fenrir & ~ended)
        if self.progression_enabled:
            killed = (state.hp > 0) & (hp == 0)
            kill_xp = self.unit_experience(state)[:,1]
            bank = state.battle_xp + jnp.sum(jnp.where(killed, kill_xp, 0).reshape(2, 6), axis=1)
            if self.has_copies or self.has_summons:
                state,hp = self._restore_roster(state,hp,raw_ended)
            ids, levels, xp, enemies, hp, gains, promoted = self.progression.finish(
                state, hp, escaped, victory, lost, withdrawal, bank,
                self.item_rules.value(state,'experience',3) if self.items_enabled else False)
            if self.has_potion_buffs:
                intrinsic = self._potion_intrinsic(ids,levels)
                changed = ((promoted >> self.slots.astype(jnp.uint32)) & 1) != 0
                doses = jnp.where(changed[:,None],state.potion_doses,0)
                rebuilt = self.potion_rules.rebuild(intrinsic,doses)
                state = state.replace(potion_bonus=jnp.where(changed[:,None],rebuilt,state.potion_bonus))
                temporary=self.potion_rules.temporary_bonus(intrinsic+state.potion_bonus,state.potion_active)
                state=state.replace(potion_temporary=jnp.where(changed[:,None],temporary,state.potion_temporary))
                hp = jnp.where(changed,(intrinsic[:,HP]+state.potion_bonus[:,HP]).astype(jnp.int32),hp)
            progress = dict(unit_ids=ids, unit_levels=levels, unit_xp=xp,
                            enemy_progress=enemies, battle_xp=bank,
                            last_xp=gains, last_promoted=promoted)
        if self.capital_enabled:
            # Count actual damage independently of healing/leech and HP rescaling.
            periodic = state.last_poison_damage[6:]+state.last_burn_damage[6:]+state.last_water_damage[6:]
            paid_damage = jnp.minimum(state.enemy_initial_hp,state.enemy_damage_credit+damage_credit+periodic)
            paid_kills = state.enemy_killed_credit | ((state.hp[6:] > 0) & (hp[6:] == 0) & (state.enemy_initial_hp > 0))
            credit = jnp.array([jnp.sum(paid_damage),jnp.sum(paid_kills)],jnp.int32)
            progress["recovery_balance"] = state.recovery_balance+jnp.where(victory,credit,0)
            progress["enemy_damage_credit"] = paid_damage
            progress["enemy_killed_credit"] = paid_kills
        recovered_hp = self.restored_hp
        if self.basic_combat:
            # Both survivors and fallen units retain their HP; revival is paid.
            recovered_hp = jnp.where(self.restored_hp > 0, hp, 0)
        next_state = state.replace(
            battle_key=key, hp=jnp.where(back, recovered_hp, hp), **progress,
            post_victory=post_team,pending_healers=pending,
            second_strike=another_strike & ~(victory | withdrawal | lost | timeout),
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
        if self.items_enabled:
            next_state=jax.lax.cond(back,self._refresh_equipment,lambda s:s,next_state)
        if self.has_witches or self.has_poisoners or self.has_water or self.has_fire or self.has_wights or self.has_hermits:
            next_state = self._start_activation(next_state, random_values[36],
                                                next_state.in_battle & ~next_state.done & (next_state.post_victory == 0))
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
        if self.potions_enabled and self.potion_rules.count:
            index = jnp.clip(action-POTION_START, 0, self.potion_rules.action_count-1)
            available = self.potion_rules.available(state, self.max_hp(state), index // 6, index % 6)
            world_valid |= (action >= POTION_START) & (action < self.num_actions) & available
        controlled = (state.actor < 6) & ~state.retreating[state.actor] & ~state.paralyzed[state.actor] & ~state.long_paralyzed[state.actor]
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
        if self.has_patriarchs:
            attack_valid = jnp.where(self._actor_is_patriarch(state),(action >= SHOOT) & (action < DEFEND) & self._patriarch_targets(state)[target],attack_valid)
        if self.has_alchemists:
            attack_valid = jnp.where(self._alchemist_units(state)[state.actor],(action >= SHOOT) & (action < DEFEND) & self._alchemist_targets(state)[target],attack_valid)
        attack_valid &= (self._actor_power_factor(state) == 0) | (target != state.actor) | self._actor_power_cures(state)
        unit_valid = attack_valid | (~state.second_strike & ((action == DEFEND) | (action == RETREAT) | ((action == WAIT) & (state.turn_phase[state.actor] == 0) & ~state.waited[state.actor])))
        unit_valid |= (action == FENRIR) & ~state.second_strike & self._actor_is_wolf_lord(state)
        if self.has_copies:
            copying = self._actor_copies(state)
            ally = (action >= COPY_ALLY) & (action < COPY_ALLY+COPY_ACTIONS)
            copy_slot = jnp.clip(jnp.where(ally,action-COPY_ALLY,action-SHOOT),0,5)
            copy_valid = ((ally | ((action >= SHOOT) & (action < DEFEND)))
                          & self._copy_targets(state)[copy_slot+jnp.where(ally,0,6)])
            unit_valid = jnp.where(copying & ((action >= SHOOT) & (action < DEFEND) | ally),copy_valid,unit_valid)
            unit_valid = jnp.where(state.round == 0,(copying & copy_valid) | (action == DEFEND),unit_valid)
        if self.has_summons:
            summoning = self._actor_summons(state) > 0
            unit_valid = jnp.where(summoning & (action >= SHOOT) & (action < DEFEND),
                                  self._summon_targets(state)[target_slot],unit_valid)
        battle_valid = jnp.where(controlled,unit_valid,action == CONTINUE)
        if self.progression_enabled and self.has_healers:
            post_valid = jnp.where(state.actor < 6,
                ((action >= SHOOT) & (action < DEFEND) & (state.hp[target_slot] > 0) & ~state.escaped[target_slot]) | (action == WAIT),
                action == CONTINUE)
            if self.has_patriarchs:
                post_valid = jnp.where((state.actor < 6) & self._actor_is_patriarch(state),attack_valid | (action == WAIT),post_valid)
            battle_valid = jnp.where(state.post_victory > 0,post_valid,battle_valid)
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
