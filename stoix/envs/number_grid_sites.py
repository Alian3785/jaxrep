"""Original 3x3 sites, merchant inventory and training. See docs/MAP_SITES.md."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from stoix.envs.number_grid_items import VALUABLE

BOUGHT, TRAINED = 38, 40
SELL_PERCENT = 20  # Installed Globals/GVars.DBF: SELL_RATIO.
# [row, column], relative to the upper-left of the 3x3 object.
APPROACH_OFFSETS = ((1, 3), (2, 3), (3, 1), (3, 2), (3, 3))
TRAINING_PRICES = json.loads((Path(__file__).parent/'data/training_prices.json').read_text())


def site_layout(game_map):
    """Validate footprints and derive entrance tiles; never accept arbitrary ranges."""
    from stoix.envs.number_grid_ruins import ruin_geometry
    ruin_cells = {p for ruin in ruin_geometry(game_map) for p in ruin['footprint']}
    sites = []
    occupied = {tuple(p) for p in game_map['opponent_positions']}
    occupied.add(tuple(game_map['agent_position']))
    occupied.update(ruin_cells)
    occupied.update(tuple(c['position']) for c in game_map.get('chests', []))
    occupied.update(tuple(c['position']) for c in game_map.get('cities', [])+game_map.get('mines', []))
    occupied.update(tuple(p) for p in game_map.get('obstacles', []))
    for kind, name in (('merchant', 'Торговец'), ('trainer', 'Тренер'), ('mercenary', 'Лагерь наёмников')):
        value = game_map.get(kind)
        if value is None:
            continue
        anchor = value.get('position') if isinstance(value, dict) else None
        if (not isinstance(anchor, (list, tuple)) or len(anchor) != 2
                or any(type(n) is not int or not 0 < n < game_map['size']-4 for n in anchor)):
            raise ValueError(kind+' needs an interior 3x3 position and space for its entrance')
        r, c = anchor
        cells = [(r+i, c+j) for i in range(3) for j in range(3)]
        if occupied.intersection(cells):
            raise ValueError('Site footprints must not overlap another object or unit')
        occupied.update(cells)
        sites.append(dict(kind=kind, name=value.get('name', name), position=list(anchor),
                          size=[3, 3], footprint=cells,
                          interaction_tiles=[(r+i, c+j) for i, j in APPROACH_OFFSETS]))
    blocked = ruin_cells | {p for site in sites for p in site['footprint']} | {tuple(p) for p in game_map.get('obstacles', [])}
    for site in sites:
        site['interaction_tiles'] = [p for p in site['interaction_tiles'] if p not in blocked]
        if not site['interaction_tiles']:
            raise ValueError('Site entrance is blocked')
    return sites


def trade_action_names(game_map):
    if game_map.get('merchant') is None and game_map.get('trainer') is None:
        return ()
    names = []
    if game_map.get('merchant') is not None:
        names += ['buy_'+key for key in game_map['merchant']['potions']]
    if game_map.get('trainer') is not None:
        names += ['train_'+str(slot) for slot in range(6)]
    return tuple(names)


class SiteRules:
    def __init__(self, game_map, potions, items, progression, action_start):
        self.sites = site_layout(game_map)
        self.items, self.potions, self.progression = items, potions, progression
        self.has_merchant = game_map.get('merchant') is not None
        self.has_trainer = game_map.get('trainer') is not None
        self.buy_keys = tuple(game_map.get('merchant', {}).get('potions', {}))
        if self.has_merchant and not self.buy_keys:
            raise ValueError('A merchant must have potion stock')
        self.buy_ids = jnp.array([potions.keys.index(k) for k in self.buy_keys], jnp.int32)
        self.initial_stock = jnp.array([game_map['merchant']['potions'][k] for k in self.buy_keys], jnp.int32)
        self.buy_base = jnp.array([potions.items[potions.keys.index(k)]['price'] for k in self.buy_keys], jnp.int32)
        self.buy_start = action_start
        self.train_start = self.buy_start+len(self.buy_keys)
        self.end = self.train_start+(6 if self.has_trainer else 0)
        self.action_count = self.end-action_start
        self.names = trade_action_names(game_map)
        self.anchors = jnp.array([s['position'] for s in self.sites], jnp.int32)
        self.footprints = jnp.array([p for s in self.sites for p in s['footprint']], jnp.int32)
        self.approaches = {s['kind']: jnp.array(s['interaction_tiles'], jnp.int32) for s in self.sites}
        self.hero_ids = jnp.array([bool(r.get('hero')) for r in progression.rows])
        prices = [[0, 0, 0, 0, 0]]
        for row in progression.rows[1:]:
            prices.append(TRAINING_PRICES[row['game_id'].lower()])
        self.training = jnp.array(prices, jnp.int32)
        self.observation_size = self.anchors.size+2*len(self.buy_keys)+(6 if self.has_trainer else 0)
        self.map_scale = game_map['size']-1

    def blocked(self, position):
        return jnp.any(jnp.all(position[..., None, :] == self.footprints, axis=-1), axis=-1)

    def at(self, state, kind):
        if kind not in self.approaches:
            return jnp.bool_(False)
        return jnp.any(jnp.all(state.position == self.approaches[kind], axis=-1))

    def leader_alive(self, state):
        return jnp.any(self.hero_ids[state.unit_ids[:6]] & (state.hp[:6] > 0))

    def common(self, state, kind):
        return self.at(state, kind) & ~state.in_battle & ~state.done & self.leader_alive(state)

    def buy_prices(self, state):
        # Reference: two copies of Lute of Charming do not stack.
        discount = jnp.max(self.items.value(state, 'discount')[:2]) if self.items else 0
        return self.buy_base*(100-discount)//100

    def training_quotes(self, state):
        ids, levels = state.unit_ids[:6], state.unit_levels[:6]
        native, threshold, base, early, late = self.training[ids].T
        first = jnp.maximum(0, jnp.minimum(levels, threshold)-native)
        last = jnp.maximum(0, levels-jnp.maximum(native, threshold))
        price = base+first*early+last*late
        cap = jnp.maximum(0, self.progression.required_xp(ids, levels)-1)
        missing = jnp.maximum(0, cap-state.unit_xp[:6])
        eligible = (ids != 0) & (state.hp[:6] > 0) & (price > 0)
        amount = jnp.where(eligible, jnp.minimum(missing, state.gold//jnp.maximum(price, 1)), 0)
        # Price/XP, cap, missing XP, affordable XP, total cost (even when away).
        return jnp.stack((price, cap, missing, amount, amount*price), axis=-1)

    def available(self, state):
        parts = []
        if self.has_merchant:
            common = self.common(state, 'merchant')
            parts.append(common & (state.merchant_stock > 0) & (state.gold >= self.buy_prices(state)))
        if self.has_trainer:
            parts.append(self.common(state, 'trainer') & (self.training_quotes(state)[:, 3] > 0))
        return jnp.concatenate(parts) if parts else jnp.zeros(0, bool)

    @staticmethod
    def compact(inventory, remove):
        # Stable compaction preserves acquisition order and leaves empty slots last.
        keep = (inventory >= 0) & ~remove
        n = inventory.size
        destination = jnp.where(keep, jnp.cumsum(keep)-1, n)
        return jnp.full_like(inventory, -1).at[destination].set(inventory, mode='drop')

    def autosell(self, state):
        if not self.has_merchant or not self.items or not self.items.capacity:
            return state
        ids = state.item_inventory
        remove = ((ids >= 0) & (self.items.tables['category'][ids+1] == VALUABLE)
                  & self.at(state, 'merchant') & ~state.in_battle & ~state.done)
        gold = jnp.sum(jnp.where(remove, self.items.tables['price'][ids+1]*SELL_PERCENT//100, 0))
        return state.replace(item_inventory=self.compact(ids, remove), gold=state.gold+gold,
                             last_sale_gold=state.last_sale_gold+gold,
                             last_sold_count=state.last_sold_count+jnp.sum(remove))

    def apply(self, state, action):
        if not self.action_count:
            return state
        index = jnp.clip(action-self.buy_start, 0, self.action_count-1)
        legal = (action >= self.buy_start) & (action < self.end) & self.available(state)[index]
        if self.has_merchant:
            def buy(s):
                i = jnp.clip(action-self.buy_start, 0, len(self.buy_keys)-1)
                cost = self.buy_prices(s)[i]
                return s.replace(gold=s.gold-cost, potions=s.potions.at[self.buy_ids[i]].add(1),
                    merchant_stock=s.merchant_stock.at[i].add(-1), last_event=jnp.int32(BOUGHT),
                    last_service_cost=cost, last_trade_item=i)
            state = jax.lax.cond(legal & (action < self.train_start), buy, lambda s: s, state)

        if self.has_trainer:
            def train(s):
                slot = jnp.clip(action-self.train_start, 0, 5)
                quote = self.training_quotes(s)[slot]
                return s.replace(gold=s.gold-quote[4], unit_xp=s.unit_xp.at[slot].add(quote[3]),
                    last_xp=s.last_xp.at[slot].set(quote[3]), last_service_cost=quote[4],
                    last_event=jnp.int32(TRAINED), last_target=slot)
            state = jax.lax.cond(legal & (action >= self.train_start), train, lambda s: s, state)
        return state

    def observation(self, state):
        parts = [self.anchors.reshape(-1)/self.map_scale,
                 state.merchant_stock/jnp.maximum(self.initial_stock, 1), self.buy_prices(state)/1000.]
        if self.has_trainer:
            parts.append(self.training_quotes(state)[:, 0]/10.)
        return jnp.concatenate(parts)

    def quotes(self, state):
        return dict(buy=self.buy_prices(state), train=self.training_quotes(state))

    @property
    def metadata(self):
        bought = [dict(self.potions.items[self.potions.keys.index(key)], action=self.buy_start+i)
                  for i, key in enumerate(self.buy_keys)]
        return dict(sites=self.sites, buy=bought,
                    train_start=self.train_start if self.has_trainer else None,
                    auto_sale='valuables', sell_percent=SELL_PERCENT)
