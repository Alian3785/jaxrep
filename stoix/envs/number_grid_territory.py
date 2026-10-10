"""Territory, mine income and settlements. Sources: docs/TERRITORY_CITIES.md.

Static BFS ranks are built once; runtime queries only gather the ranks needed.
Each source grows independently, including overlap with another owned source.
"""
from collections import deque
from functools import lru_cache
import jax
import jax.numpy as jnp

from stoix.envs.number_grid_lords import lord_settings

MANA_KINDS = ('infernal', 'life', 'death', 'runes', 'elves')
RESOURCE_KINDS = ('gold',) + MANA_KINDS
RESOURCE_NAMES = ('Золото', 'Мана преисподней', 'Мана жизни', 'Мана смерти', 'Мана рун', 'Мана эльфов')
CITY_CAPTURED, CITY_UPGRADED, OBSTACLE = 43, 44, 45
CITY_GROWTH = (0, 15, 15, 20, 20, 25)
CITY_BONUS = (0, 10, 15, 20, 25, 30)
CITY_COST = (0, 150, 250, 500, 750, 0)


@lru_cache(maxsize=32)
def growth_ranks(size, sources, blocked):
    """Reference BFS + Chebyshev/Manhattan tie break, converted from x/y to r/c."""
    blocked = set(blocked)
    rows = []
    for origin in sources:
        distance, queue = {origin: 0}, deque([origin])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
                nxt = (r+dr, c+dc)
                if (0 < nxt[0] < size-1 and 0 < nxt[1] < size-1
                        and nxt not in blocked and nxt not in distance):
                    distance[nxt] = distance[(r,c)]+1
                    queue.append(nxt)
        def order(tile, origin=origin, distance=distance):
            dy, dx = tile[0]-origin[0], tile[1]-origin[1]
            return (distance[tile], max(abs(dx),abs(dy)), abs(dx)+abs(dy),
                    abs(dy), abs(dx), tile[0], tile[1])
        ranks = [size*size]* (size*size)
        for rank, (r,c) in enumerate(sorted(distance, key=order)):
            ranks[r*size+c] = rank
        rows.append(tuple(ranks))
    return tuple(rows)


class TerritoryRules:
    def __init__(self, game_map, progression, start, site_rules=None, ruin_rules=None):
        self.size = int(game_map['size'])
        self.capital = tuple(game_map['agent_position'])
        self.cities = list(game_map.get('cities', []))
        self.count = len(self.cities)
        self.start, self.end = start, start+self.count
        self.positions = jnp.array([c['position'] for c in self.cities], jnp.int32).reshape(-1,2)
        self.initial_levels = jnp.array([c.get('level',1) for c in self.cities], jnp.int32)
        self.mines = list(game_map.get('mines', []))
        self.mine_positions = jnp.array([m['position'] for m in self.mines], jnp.int32).reshape(-1,2)
        self.mine_kinds = jnp.array([RESOURCE_KINDS.index(m['kind']) for m in self.mines], jnp.int32)
        self.mine_onehot = jax.nn.one_hot(self.mine_kinds, 6, dtype=jnp.int32)
        self.mine_cells = self.mine_positions[:,0]*self.size+self.mine_positions[:,1]
        lord = lord_settings(game_map)
        self.lord = lord['id']
        self.lord_regen = lord['regeneration']
        faction_mana = dict(legions=0, empire=1, undead_hordes=2, mountain_clans=3, elves=4)
        self.base_income = jnp.array([100]+[25 if i == faction_mana[game_map['faction']] else 0 for i in range(5)], jnp.int32)
        self.heroes = jnp.array([bool(r.get('hero')) for r in progression.rows])
        self.regen = jnp.array([int(r.get('regeneration',r.get('game_data',{}).get('REGEN',0)) or 0) for r in progression.rows],jnp.int32)
        self.growth = jnp.array(CITY_GROWTH,jnp.int32)
        self.fort_bonus = jnp.array(CITY_BONUS,jnp.int32)
        self.costs = jnp.array(CITY_COST,jnp.int32)//lord['city_cost_divisor']
        obstacles = {tuple(p) for p in game_map.get('obstacles',[])}
        reserved = {self.capital, *(tuple(c['position']) for c in self.cities),
                    *(tuple(m['position']) for m in self.mines),
                    *(tuple(p) for p in game_map['opponent_positions']),
                    *(tuple(c['position']) for c in game_map.get('chests',[]))}
        if obstacles & reserved:
            raise ValueError('Obstacles overlap an army, city, chest, capital or mine')
        points = [self.capital]+[tuple(c['position']) for c in self.cities]+[tuple(m['position']) for m in self.mines]+list(obstacles)
        if any(len(p)!=2 or any(type(v) is not int or not 0<v<self.size-1 for v in p) for p in points):
            raise ValueError('Territory objects must be interior cells')
        if len({tuple(c['position']) for c in self.cities}) != self.count or self.capital in {tuple(c['position']) for c in self.cities}:
            raise ValueError('Settlement entrances must be distinct')
        if any(type(c.get('level',1)) is not int or not 1<=c.get('level',1)<=5 for c in self.cities):
            raise ValueError('City level must be 1..5')
        # All 3x3 sites share the same impassable footprint and entrance rules.
        if site_rules is not None:
            obstacles.update(tuple(p) for site in site_rules.sites for p in site['footprint'])
        if ruin_rules is not None:
            obstacles.update(tuple(p) for ruin in ruin_rules.ruins for p in ruin['footprint'])
        self.obstacle_cells = tuple(sorted(obstacles))
        blocked = [False]*(self.size*self.size)
        for r,c in obstacles:
            blocked[r*self.size+c] = True
        self.blocked_grid = jnp.array(blocked)
        directions = ((-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1))
        self.movement_grid = jnp.array([[0<r+dr<self.size-1 and 0<c+dc<self.size-1
            and (r+dr,c+dc) not in obstacles for dr,dc in directions]
            for r in range(self.size) for c in range(self.size)],bool)
        sources = (self.capital,)+tuple(tuple(c['position']) for c in self.cities)
        ranks = growth_ranks(self.size, sources, tuple(sorted(obstacles)))
        # Water is passable to armies but cannot become faction land. Keep
        # physical reachability separate, including land accessible across water.
        water = {tuple(p) for p in game_map.get('terrain',{}).get('water',[])}
        land_ranks = growth_ranks(self.size, sources, tuple(sorted(obstacles | water))) if water else ranks
        self.ranks = jnp.array(land_ranks,jnp.int32)
        self.reachable_counts = jnp.sum(self.ranks < self.size*self.size,axis=1)
        if ruin_rules is not None:
            reserved.difference_update(tuple(r['entrance']) for r in ruin_rules.ruins)
            for ruin in ruin_rules.ruins:
                if not any(ranks[0][r*self.size+c] < self.size*self.size for r,c in ruin['interaction_tiles']):
                    raise ValueError('Ruin has no reachable approach: '+ruin['name'])
        for p in reserved:
            if ranks[0][p[0]*self.size+p[1]] == self.size*self.size:
                raise ValueError('Map object is unreachable from capital: '+str(p))
        self.initial_claims = jnp.array([1]+[0]*self.count,jnp.int32)
        enemy_positions = game_map['opponent_positions']
        self.enemy_count = len(enemy_positions)
        armies = [[] for _ in range(self.size*self.size)]
        for i,(r,c) in enumerate(enemy_positions):
            armies[r*self.size+c].append(i)
        width = max(1,max(map(len,armies)))
        self.armies_by_cell = jnp.array([row+[self.enemy_count]*(width-len(row)) for row in armies],jnp.int32)
        self.city_defenders = self.armies_by_cell[self.positions[:,0]*self.size+self.positions[:,1]]
        self.city_enemies = jnp.array([[tuple(p)==tuple(c['position']) for p in enemy_positions] for c in self.cities],bool).reshape(self.count,len(enemy_positions))
        self.goal_cities = jnp.any(self.city_enemies,axis=1)
        self.enemy_city = jnp.array([next((i for i,c in enumerate(self.cities) if tuple(p)==tuple(c['position'])),-1) for p in enemy_positions],jnp.int32)
        offsets = [(r,c) for r in range(-2,3) for c in range(-2,3)]
        self.local_offsets = jnp.array(offsets,jnp.int32)
        # Mana, six income values, independent source growth, city and mine rows,
        # local 5x5 ownership/obstacle windows, fort/lord context.
        self.observation_size = 5+6+(1+self.count)+7*self.count+5*len(self.mines)+50+3
        self.metadata = dict(cities=[dict(c,action=start+i) for i,c in enumerate(self.cities)],
            mines=self.mines, resource_kinds=RESOURCE_KINDS, resource_names=RESOURCE_NAMES,
            mine_income=50,base_income=self.base_income.tolist(),capital_growth=10,
            city_growth=CITY_GROWTH,city_bonus=CITY_BONUS,city_costs=self.costs.tolist(),
            capital_regeneration=35,capital_armor=50,own_land_regeneration=10,
            lord_type=self.lord,lord_regeneration=self.lord_regen,obstacles=game_map.get('obstacles',[]))

    def _cells(self, positions):
        return positions[...,0]*self.size+positions[...,1]

    def owned(self, state, positions):
        cells = self._cells(jnp.clip(positions,0,self.size-1))
        return jnp.any(self.ranks[:,cells] < state.territory_claims.reshape((-1,)+(1,)*cells.ndim),axis=0)

    def blocked(self, positions):
        outside = jnp.any((positions<=0)|(positions>=self.size-1),axis=-1)
        return outside | self.blocked_grid[self._cells(jnp.clip(positions,0,self.size-1))]

    def movement_mask(self, state):
        return self.movement_grid[self._cells(jnp.clip(state.position,0,self.size-1))]

    def _living(self, state, indices):
        return (indices<self.enemy_count) & state.alive[jnp.minimum(indices,self.enemy_count-1)]

    def enemy_at(self, state, positions):
        indices = self.armies_by_cell[self._cells(jnp.clip(positions,0,self.size-1))]
        first = jnp.min(jnp.where(self._living(state,indices),indices,self.enemy_count),axis=-1)
        return jnp.where(first<self.enemy_count,first,-1)

    def remaining_defenders(self, state):
        return jnp.sum(self._living(state,self.city_defenders),axis=-1)

    def city_at(self, state):
        return jnp.all(self.positions==state.position,axis=1)

    def at_owned_city(self, state):
        return jnp.any(self.city_at(state)&state.city_owned)

    def service_at(self, state):
        return jnp.all(state.position==jnp.array(self.capital)) | self.at_owned_city(state)

    def fort_regeneration(self,state):
        city = jnp.max(jnp.where(self.city_at(state)&state.city_owned,self.fort_bonus[state.city_levels],0),initial=0)
        return jnp.where(jnp.all(state.position==jnp.array(self.capital)),35,city)

    def regeneration_percent(self,state,banner=0):
        place = jnp.where(self.service_at(state),self.fort_regeneration(state),10*self.owned(state,state.position))
        return jnp.where(state.unit_ids!=0,jnp.clip(self.regen[state.unit_ids]+self.lord_regen+place+10*banner,0,100),0)

    def mine_ownership(self,state):
        return self.owned(state,self.mine_positions)

    def income(self,state):
        return self.base_income+50*jnp.sum(self.mine_onehot*self.mine_ownership(state)[:,None],axis=0)

    def next_turn(self,state):
        rates = jnp.concatenate((jnp.array([10],jnp.int32),jnp.where(state.city_owned,self.growth[state.city_levels],0)))
        state = state.replace(territory_claims=jnp.minimum(state.territory_claims+rates,self.reachable_counts))
        income = self.income(state)
        return state.replace(gold=state.gold+income[0],mana=state.mana+income[1:])

    def capture(self,state,entered):
        hero = jnp.any(self.heroes[state.unit_ids[:6]] & (state.hp[:6]>0) & (state.summon_owner[:6]<0) & ~state.copied[:6])
        cleared = self.remaining_defenders(state)==0
        captured = entered & ~state.city_owned & self.city_at(state) & cleared & hero
        claims = state.territory_claims.at[1:].set(jnp.where(captured,1,state.territory_claims[1:]))
        return state.replace(city_owned=state.city_owned|captured,territory_claims=claims,last_captured_cities=captured,
            last_event=jnp.where(jnp.any(captured),CITY_CAPTURED,state.last_event))

    def goals_captured(self,state):
        return jnp.all(state.city_owned | ~self.goal_cities)

    def available(self,state):
        return ~state.in_battle & ~state.done & state.city_owned & (state.city_levels<5) & (state.gold>=self.costs[state.city_levels])

    def upgrade(self,state,action):
        if not self.count:
            return state
        index = jnp.clip(action-self.start,0,self.count-1)
        valid = (action>=self.start)&(action<self.end)&self.available(state)[index]
        cost = jnp.where(valid,self.costs[state.city_levels[index]],0)
        return state.replace(city_levels=state.city_levels.at[index].add(valid.astype(jnp.int32)),gold=state.gold-cost,
            last_upgraded_city=jnp.where(valid,index,-1),last_service_cost=jnp.where(valid,cost,state.last_service_cost),
            last_event=jnp.where(valid,CITY_UPGRADED,state.last_event))

    def defender_armor(self,state):
        if not self.count:
            return jnp.int32(0)
        index = self.enemy_city[jnp.maximum(state.enemy,0)]
        return jnp.where((state.enemy>=0)&(index>=0)&~state.city_owned[jnp.maximum(index,0)],self.fort_bonus[state.city_levels[jnp.maximum(index,0)]],0)

    def observation(self,state):
        cities = jnp.concatenate((self.positions/(self.size-1),state.city_levels[:,None]/5.,state.city_owned[:,None],
            state.territory_claims[1:,None]/(self.size*self.size),
            self.remaining_defenders(state)[:,None]/2.,
            self.costs[state.city_levels,None]/1000.),axis=1).reshape(-1)
        mines = jnp.concatenate((self.mine_positions/(self.size-1),self.mine_kinds[:,None]/5.,
            self.mine_ownership(state)[:,None],jnp.full((len(self.mines),1),.05)),axis=1).reshape(-1)
        local = state.position+self.local_offsets
        return jnp.concatenate((state.mana/1000.,self.income(state)/100.,state.territory_claims/(self.size*self.size),
            cities,mines,self.owned(state,local),self.blocked(local),
            jnp.array([self.service_at(state),self.fort_regeneration(state)/100.,self.lord_regen/100.])))

    def quotes(self,state,banner=0):
        return dict(income=self.income(state),mine_owned=self.mine_ownership(state),
            city_available=self.available(state),city_costs=self.costs[state.city_levels],
            at_owned_fort=self.service_at(state),regeneration=self.regeneration_percent(state,banner)[:6],
            territory=jnp.any(self.ranks<state.territory_claims[:,None],axis=0).reshape(self.size,self.size))
