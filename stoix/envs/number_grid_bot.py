"""Mirror match's capital bot. Sources and scope: docs/MIRROR_MATCH.md.

Fixed-shape FIFO BFS and the ordinary tactical engine execute entirely in JAX.
The scratch battle state is private: only enemy armies, bot progress and PRNG
are committed, never the player's inventory, rewards, HP or campaign counters.
"""
import jax
import jax.numpy as jnp

from stoix.envs.number_grid_combat import HP

HUNTING, RETURNING, RESTING, DEFEATED = range(4)
MODE_NAMES = ('hunting', 'returning', 'resting', 'defeated')


class ScriptedBotRules:
    def __init__(self, game_map, territory, progression):
        self.enabled = bool(game_map.get('scripted_bot'))
        self.observation_size = 0
        if not self.enabled:
            return
        spec = game_map['scripted_bot']
        self.enemy = spec['enemy']
        self.size = game_map['size']
        self.cells = self.size*self.size
        self.home = jnp.array(spec['home'], jnp.int32)
        self.budget = spec.get('movement', 20)
        self.respawn_turns = spec.get('respawn_turns', 2)
        self.battle_decisions = spec.get('battle_decisions', 300)
        if (type(self.enemy) is not int or not 0 <= self.enemy < len(game_map['enemy_rosters'])
                or any(type(v) is not int or v <= 0
                       for v in (self.budget, self.respawn_turns, self.battle_decisions))):
            raise ValueError('Invalid scripted bot index or positive integer budgets')
        if (list(spec['home']) != list(game_map['opponent_positions'][self.enemy])
                or tuple(spec['home']) in territory.obstacle_cells):
            raise ValueError('Scripted bot must start on its passable home tile')
        if game_map.get('cities'):
            raise ValueError('This scripted bot supports city-free maps such as Mirror match')
        self.territory = territory
        self.progression = progression
        self.initial_ids = progression.enemy_ids[self.enemy]
        self.initial_progress = progression.initial_enemies[self.enemy]
        source_ids = game_map.get('reference_enemy_ids', list(range(len(game_map['enemy_rosters']))))
        self.source_ids = jnp.array(source_ids, jnp.int32)
        excluded = set(spec.get('protected_enemies', [])) | {self.enemy}
        self.eligible = jnp.array([i not in excluded for i in range(len(source_ids))])
        # Reference offsets in (x,y), converted to row/column.
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                   (1, -1), (1, 0), (1, 1)]
        neighbors, valid = [], []
        blocked = set(territory.obstacle_cells)
        for r in range(self.size):
            for c in range(self.size):
                neighbors.append([max(0, min(self.cells-1, (r+dr)*self.size+c+dc)) for dr, dc in offsets])
                valid.append([0 < r+dr < self.size-1 and 0 < c+dc < self.size-1
                              and (r+dr, c+dc) not in blocked for dr, dc in offsets])
        self.neighbors = jnp.array(neighbors, jnp.int32)
        self.valid_neighbors = jnp.array(valid, bool)
        self.observation_size = 5+6*4

    def _cell(self, position):
        return position[..., 0]*self.size+position[..., 1]

    def ids(self, state):
        return self.progression.enemy_ids.at[self.enemy].set(state.bot_ids)

    def maximum(self, state):
        return self.progression.stats(state.bot_ids, state.enemy_progress[self.enemy, :, 0])[:, HP].astype(jnp.int32)

    def mark_defeated(self, state):
        return state.replace(bot_mode=jnp.int32(DEFEATED),
            bot_timer=jnp.where(state.bot_mode == DEFEATED, state.bot_timer, self.respawn_turns),
            alive=state.alive.at[self.enemy].set(False),
            pending_encounters=state.pending_encounters.at[self.enemy].set(False),
            enemy_spell_effects=state.enemy_spell_effects.at[self.enemy].set(jnp.uint32(0)))

    def sync_defeat(self, state):
        return jax.lax.cond(~state.alive[self.enemy] & (state.bot_mode != DEFEATED),
                            self.mark_defeated, lambda s: s, state)

    def _search(self, state, allow_hero):
        """FIFO BFS: occupied enemy cells are reachable terminal nodes."""
        start = self._cell(state.enemy_positions[self.enemy])
        hero = self._cell(state.position)
        occupied = jnp.zeros(self.cells, jnp.int32).at[self._cell(state.enemy_positions)].max(
            state.alive.astype(jnp.int32)) > 0
        occupied = occupied.at[start].set(False)
        queue = jnp.zeros(self.cells, jnp.int32).at[0].set(start)
        parent = jnp.full(self.cells, -1, jnp.int32).at[start].set(start)
        distance = jnp.full(self.cells, self.cells, jnp.int32).at[start].set(0)

        def visit(carry):
            head, tail, queue, parent, distance = carry
            current = queue[head]
            neighbors = self.neighbors[current]
            fresh = self.valid_neighbors[current] & (parent[neighbors] < 0)
            fresh &= (neighbors != hero) | allow_hero
            fresh &= ~occupied[current] | (current == start)
            # Distinct valid neighbours; invalid entries use an out-of-bounds
            # scatter dropped by JAX rather than overwriting a real cell.
            indices = jnp.where(fresh, neighbors, self.cells)
            parent = parent.at[indices].set(current, mode='drop')
            distance = distance.at[indices].set(distance[current]+1, mode='drop')
            slots = jnp.where(fresh, tail+jnp.cumsum(fresh.astype(jnp.int32))-1, self.cells)
            queue = queue.at[slots].set(neighbors, mode='drop')
            return head+1, tail+jnp.sum(fresh), queue, parent, distance

        _, _, _, parent, distance = jax.lax.while_loop(
            lambda carry: carry[0] < carry[1], visit,
            (jnp.int32(0), jnp.int32(1), queue, parent, distance))
        return parent, distance

    def navigate(self, env, state):
        position = state.enemy_positions[self.enemy]
        returning = state.bot_mode == RETURNING
        targets = self.eligible & state.alive
        has_target = jnp.any(targets)
        pursuing = ~returning & ~has_target
        home_blocked = returning & jnp.all(state.position == self.home)
        parent, distance = self._search(state, pursuing | home_blocked)
        costs = distance[self._cell(state.enemy_positions)]
        fallback = 10000+jnp.max(jnp.abs(state.enemy_positions-position), axis=1)
        costs = jnp.where(costs < self.cells, costs, fallback)
        nearest = jnp.min(jnp.where(targets, costs, 100000))
        target_index = jnp.argmin(jnp.where(targets & (costs == nearest), self.source_ids, 2**30))
        target_position = jnp.where(returning, self.home,
                                   jnp.where(pursuing, state.position, state.enemy_positions[target_index]))
        target_cell = self._cell(target_position)
        reachable = parent[target_cell] >= 0
        start = self._cell(position)
        # Reverse path [target, ..., start], built with a fixed-shape buffer.
        path = jnp.full(self.cells, start, jnp.int32).at[0].set(target_cell)

        def trace(carry):
            count, path, current = carry
            previous = parent[current]
            return count+1, path.at[count].set(previous), previous

        length, path, _ = jax.lax.while_loop(
            lambda c: reachable & (c[2] != start) & (c[0] < self.cells), trace,
            (jnp.int32(1), path, target_cell))
        stop_adjacent = ~returning | home_blocked
        steps = jnp.maximum(0, length-1-stop_adjacent.astype(jnp.int32))
        bot_hp = jnp.maximum(0, self.maximum(state)-state.enemy_wounds[self.enemy])
        travel = state.replace(unit_ids=state.unit_ids.at[:6].set(state.bot_ids),
                               hp=state.hp.at[:6].set(bot_hp), in_battle=jnp.bool_(False),
                               copied=jnp.zeros(12, bool), summon_owner=jnp.full(12, -1, jnp.int32))
        terrain_costs = env.terrain.costs(travel)
        capital = jnp.array(self.territory.capital)

        def move(i, carry):
            pos, remaining, stopped = carry
            cell = path[jnp.maximum(0, length-2-i)]
            next_pos = jnp.array([cell//self.size, cell % self.size], jnp.int32)
            # Historical greedy fallback for unreachable neutrals/return home.
            candidates = pos+jnp.array([(dy, dx) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                        if dx or dy], jnp.int32)
            occupied = jnp.any(jnp.all(candidates[:, None] == state.enemy_positions, axis=-1)
                               & state.alive[None, :]
                               & (jnp.arange(len(state.alive)) != self.enemy)[None, :], axis=1)
            valid = ~self.territory.blocked(candidates) & ~occupied & jnp.any(candidates != state.position, axis=1)
            delta = jnp.abs(candidates-target_position)
            score = jnp.max(delta, axis=1)*self.cells+jnp.sum(delta, axis=1)*self.size
            greedy = candidates[jnp.argmin(jnp.where(valid, score, 2**30))]
            partial = ~reachable & ~pursuing & jnp.any(valid)
            partial &= ~((jnp.max(jnp.abs(pos-target_position)) <= 1) & stop_adjacent)
            partial &= jnp.any(pos != target_position)
            next_pos = jnp.where(partial, greedy, next_pos)
            active = (remaining > 0) & ~stopped & ((reachable & (i < steps)) | partial)
            nominal = terrain_costs[env.terrain.kinds(next_pos)]
            pos = jnp.where(active, next_pos, pos)
            remaining -= jnp.where(active, jnp.minimum(remaining, nominal), 0)
            stopped |= active & jnp.all(pos == capital)
            return pos, remaining, stopped

        position, remaining, _ = jax.lax.fori_loop(0, self.budget, move,
            (position, jnp.int32(self.budget), jnp.bool_(False)))
        contact = jnp.max(jnp.abs(position-target_position)) <= 1
        pending = contact & (pursuing | home_blocked)
        arrived = returning & ~home_blocked & jnp.all(position == self.home)
        state = state.replace(enemy_positions=state.enemy_positions.at[self.enemy].set(position),
            bot_mode=jnp.where(arrived, RESTING, state.bot_mode), bot_timer=jnp.where(arrived, 1, state.bot_timer),
            bot_target=jnp.where(returning, -2, jnp.where(pursuing, -1, target_index)),
            bot_moves_spent=self.budget-remaining,
            pending_encounters=state.pending_encounters.at[self.enemy].set(pending))
        fight = ~returning & has_target & contact
        return state, target_index, fight

    def _autobattle(self, env, state, target):
        # Player-only layers are cleared in a private copy. No host callbacks.
        ids = state.bot_ids
        progress = state.enemy_progress[self.enemy]
        hp = jnp.maximum(0, self.maximum(state)-state.enemy_wounds[self.enemy])
        scratch = state.replace(bot_autobattle=jnp.bool_(True),
            position=state.enemy_positions[self.enemy], origin=state.enemy_positions[self.enemy],
            enemy=target, unit_ids=state.unit_ids.at[:6].set(ids),
            unit_levels=state.unit_levels.at[:6].set(progress[:, 0]),
            unit_xp=state.unit_xp.at[:6].set(progress[:, 1]), hp=state.hp.at[:6].set(hp),
            done=jnp.bool_(False), won=jnp.bool_(False), lost=jnp.bool_(False),
            summon_battle=jnp.bool_(False), spell_effects=jnp.uint32(0),
            potion_bonus=jnp.zeros_like(state.potion_bonus), potion_doses=jnp.zeros_like(state.potion_doses),
            potion_temporary=jnp.zeros_like(state.potion_temporary), potion_active=jnp.zeros_like(state.potion_active),
            potion_wards=jnp.zeros_like(state.potion_wards),
            equipped=jnp.full_like(state.equipped, -1),
            equipment_bonus=jnp.zeros_like(state.equipment_bonus),
            item_inventory=jnp.full_like(state.item_inventory, -1),
            item_charges_spent=jnp.zeros_like(state.item_charges_spent),
            battle_items=jnp.full_like(state.battle_items, -1),
            battle_hero_fort_armor=jnp.int32(0))
        scratch = env._begin_battle(scratch)

        def activation(carry):
            decisions, current = carry
            automatic = env.automatic_battle_turn(current)
            action = jnp.where(automatic, env._automatic_battle_action(current),
                               self.blue_action(env, current))
            key, draw = jax.random.split(current.battle_key)
            following, _ = env._battle_step(current, action, key,
                                            jax.random.uniform(draw, (env.random_size,)))
            return decisions+(~automatic).astype(jnp.int32), following

        # One tactical graph for initial red activations, blue decisions and
        # automatic responses. Complete the last response/healing sequence even
        # when the blue-decision limit has just been reached.
        _, result = jax.lax.while_loop(
            lambda c: c[1].in_battle & ~c[1].done &
                      ((c[0] < self.battle_decisions) | env.automatic_battle_turn(c[1])),
            activation, (jnp.int32(0), scratch))
        victory = ~result.alive[target] & ~result.lost
        bot_max = self.progression.stats(result.unit_ids[:6], result.unit_levels[:6])[:, HP].astype(jnp.int32)
        enemy_max = self.progression.stats(result.unit_ids[6:], result.unit_levels[6:])[:, HP].astype(jnp.int32)
        # On loss/timeout the surviving neutral retains wounds and progression.
        surviving_wounds = jnp.where(result.in_battle, jnp.maximum(0, enemy_max-result.hp[6:]),
                                     result.enemy_wounds[target])
        wounds = state.enemy_wounds.at[target].set(jnp.where(victory, 0, surviving_wounds))
        wounds = wounds.at[self.enemy].set(jnp.where(victory, jnp.maximum(0, bot_max-result.hp[:6]), state.enemy_wounds[self.enemy]))
        enemy_progress = state.enemy_progress.at[target].set(result.enemy_progress[target])
        saved = jnp.stack((result.unit_levels[:6], result.unit_xp[:6]), axis=-1)
        enemy_progress = enemy_progress.at[self.enemy].set(jnp.where(victory, saved, progress))
        updated = state.replace(battle_key=result.battle_key, enemy_wounds=wounds,
            enemy_progress=enemy_progress, bot_ids=jnp.where(victory, result.unit_ids[:6], ids),
            alive=state.alive.at[target].set(state.alive[target] & ~victory),
            enemy_spell_effects=state.enemy_spell_effects.at[target].set(jnp.where(victory, jnp.uint32(0), state.enemy_spell_effects[target])),
            bot_mode=jnp.where(victory, RETURNING, state.bot_mode),
            bot_defeated_enemies=state.bot_defeated_enemies+victory.astype(jnp.int32))
        return jax.lax.cond(victory, lambda s: s, self.mark_defeated, updated)

    def blue_action(self, env, state):
        """Reference scripted blue: lowest HP target, wounded ally for healers."""
        from stoix.envs.number_grid import SHOOT, DEFEND, WAIT
        mask = env.action_mask(state)
        support = env._actor_is_healer(state)
        hp = jnp.where(support, state.hp[:6], state.hp[6:])
        maximum = env._combat_stats(state)[:6, HP]
        valid = mask[SHOOT:DEFEND] & (~support | (state.hp[:6] < maximum))
        target = jnp.argmin(jnp.where(valid, hp, 2**30))
        fallback = jnp.where(mask[DEFEND], DEFEND, jnp.where(mask[WAIT], WAIT, jnp.argmax(mask)))
        # With no wounded recipient the reference falls back to its first
        # allowed target; mass healers also use the first target selector.
        first = jnp.argmax(mask[SHOOT:DEFEND])
        fallback = jnp.where(jnp.any(mask[SHOOT:DEFEND]), SHOOT+first, fallback)
        action = jnp.where(jnp.any(valid), SHOOT+target, fallback)
        return jnp.where(env._actor_mass_healer(state), SHOOT+first, action).astype(jnp.int32)

    def advance(self, env, state):
        state = state.replace(bot_moves_spent=jnp.int32(0), bot_target=jnp.int32(-1))

        def respawn(s):
            timer = jnp.maximum(0, s.bot_timer-1)
            due = timer == 0
            return s.replace(bot_timer=timer, bot_mode=jnp.where(due, HUNTING, DEFEATED),
                bot_ids=jnp.where(due, self.initial_ids, s.bot_ids),
                enemy_progress=s.enemy_progress.at[self.enemy].set(jnp.where(due, self.initial_progress, s.enemy_progress[self.enemy])),
                enemy_wounds=s.enemy_wounds.at[self.enemy].set(jnp.where(due, 0, s.enemy_wounds[self.enemy])),
                enemy_positions=s.enemy_positions.at[self.enemy].set(jnp.where(due, self.home, s.enemy_positions[self.enemy])),
                alive=s.alive.at[self.enemy].set(due),
                pending_encounters=s.pending_encounters.at[self.enemy].set(due & jnp.all(s.position == self.home)))

        def rest(s):
            return s.replace(bot_mode=jnp.int32(HUNTING), bot_timer=jnp.int32(0),
                             enemy_wounds=s.enemy_wounds.at[self.enemy].set(0))

        def travel(s):
            s, target, fight = self.navigate(env, s)
            return jax.lax.cond(fight, lambda s: self._autobattle(env, s, target), lambda s: s, s)

        # Hunting and returning share one travel program. Repeating `travel`
        # in switch's branch tuple creates two copies of the entire battle
        # graph during lowering, despite both entries calling the same Python
        # function. Keep the original mode in state for navigate's target rule.
        branch = jnp.maximum(state.bot_mode-1, 0)
        return jax.lax.switch(branch, (travel, rest, respawn), state)

    def after_battle(self, before, after, victory, ended):
        fighting_bot = (before.enemy == self.enemy) & ~before.bot_autobattle
        won = fighting_bot & victory
        real_victory = won & ~before.summon_battle
        after = after.replace(bot_hero_victories=after.bot_hero_victories+real_victory.astype(jnp.int32),
            bot_ids=jnp.where(fighting_bot & ended, after.unit_ids[6:], after.bot_ids),
            bot_mode=jnp.where(fighting_bot & ended & ~victory, RETURNING, after.bot_mode))
        return jax.lax.cond(won, self.mark_defeated, lambda s: s, after)

    def observation(self, state):
        progress = state.enemy_progress[self.enemy]
        maximum = self.maximum(state)
        hp = jnp.maximum(0, maximum-state.enemy_wounds[self.enemy])
        hp = jnp.where(state.alive[self.enemy], hp, 0)
        return jnp.concatenate((jnp.array([state.bot_mode/3., state.bot_timer/2.,
            state.bot_defeated_enemies/len(state.alive), state.bot_hero_victories,
            state.bot_moves_spent/self.budget]),
            state.bot_ids/len(self.progression.rows), progress[:, 0]/99.,
            progress[:, 1]/1000., hp/jnp.maximum(1, maximum)))
