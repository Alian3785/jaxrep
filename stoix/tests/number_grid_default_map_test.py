"""CUDA contracts: the Python reference Default map, its spell shop and capital guard."""
import jax
import jax.numpy as jnp
import pytest

from stoix.envs.number_grid import NumberGrid, REST, action_names, load_map
from stoix.envs.number_grid_casting import SPELL_CAST, SPELL_KILL, SUMMON_CAST
from stoix.envs.number_grid_sites import SPELL_BOUGHT
from stoix.envs.number_grid_territory import CAPITAL_ARMOR
from stoix.tests.number_grid_fixtures import compiled_method

DEFAULT = load_map('default')
SHOP = (4, 46)  # spell shop at [3, 43], approach offset (1, 3)
ARMAGEDDON, VERDANT = 'g000ss0019', 'g000ss0117'
GUARD, DRAGON = 74, 30  # reference enemy ids 75 (Myzrael) and 31 (Green Dragon)
DIRECTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


@pytest.fixture(scope='module')
def default_game():
    env = NumberGrid(map_config=DEFAULT)
    state, _ = compiled_method(env, 'reset')(jax.random.PRNGKey(42))
    return env, state, compiled_method(env, 'step'), compiled_method(env, 'action_mask')


def test_default_map_layout_and_actions(default_game):
    env, state, _, _ = default_game
    names = action_names(DEFAULT)
    assert DEFAULT['size'] == 50 and len(DEFAULT['opponent_positions']) == 75
    assert env.num_actions == len(names) == 229 and env.observation_size == 2432
    assert [s['kind'] for s in env.site_rules.sites] == [
        'merchant', 'merchant', 'trainer', 'mercenary', 'mercenary', 'spell_shop']
    assert env.territory.count == 5 and len(env.territory.mines) == 12 and env.ruins.count == 5
    assert len(DEFAULT['chests']) == 18 and env.item_rules.capacity == 15
    shop = env.site_rules
    assert names[shop.spell_start:shop.end] == ('buy_spell_'+ARMAGEDDON, 'buy_spell_'+VERDANT)
    assert names[env.casting.start:][-1] == 'cast_'+VERDANT and env.casting.count == 23
    # Alar sells the reference merchant stock; Tralar only buys valuables.
    assert names[shop.buy_start:shop.train_start] == tuple('buy_'+k for k in DEFAULT['merchants'][0]['potions'])
    assert [int(i) for i in env.recruitment.offer_sites] == [3, 4, 4]
    assert DEFAULT['enemy_rosters'][GUARD] == [None]*4+['g000uu3001', None]
    assert DEFAULT['enemy_capitals'][0]['position'] == DEFAULT['opponent_positions'][GUARD] == [11, 27]
    assert [int(i) for i in jnp.flatnonzero(env.territory.capital_guards)] == [GUARD]
    ids = env.progression.ids
    assert state.unit_ids[:6].tolist() == [ids['squire'], ids['g000uu0019'], ids['squire'], 0, ids['acolyte'], 0]
    assert state.position.tolist() == [28, 6]
    assert int(state.gold) == 0 and state.spell_stock.tolist() == [1, 1]


def test_spell_shop_sells_armageddon_and_verdant(default_game):
    env, initial, step, mask = default_game
    shop, casting = env.site_rules, env.casting
    ids = [row['id'] for row in casting.rows]
    armageddon, verdant = ids.index(ARMAGEDDON), ids.index(VERDANT)
    at_shop = initial.replace(position=jnp.array(SHOP, jnp.int32), gold=jnp.int32(1500))
    assert mask(at_shop)[shop.spell_start] and mask(at_shop)[shop.spell_start+1]
    assert not mask(initial.replace(gold=jnp.int32(1500)))[shop.spell_start]
    bought, _ = step(at_shop, jnp.int32(shop.spell_start))
    assert bought.last_event == SPELL_BOUGHT and bought.gold == 500
    assert bought.spell_stock.tolist() == [0, 1] and bought.learned_spells == 1 << armageddon
    assert not jnp.any(mask(bought)[shop.spell_start:shop.end])  # sold out / 500 < 1000
    # Level V needs a mage to research; a bought copy is cast by any lord at full price.
    cast = casting.start+armageddon
    assert not mask(bought)[cast]
    armed = bought.replace(mana=jnp.array([70, 250, 80, 100, 0], jnp.int32))
    assert mask(armed)[cast]
    struck, _ = step(armed, jnp.int32(cast))
    assert struck.last_event in (SPELL_CAST, SPELL_KILL) and not jnp.any(struck.mana)
    assert struck.last_spell_target >= 0 and casting.field[struck.last_spell_target]
    # Verdant: a foreign shop spell gets its own cast action, one cast per day.
    second, _ = step(at_shop.replace(gold=jnp.int32(1000)), jnp.int32(shop.spell_start+1))
    assert second.learned_spells == 1 << verdant and second.gold == 0
    cast = casting.start+verdant
    assert not mask(second.replace(mana=jnp.array([5000]*4+[0], jnp.int32)))[cast]
    ready = second.replace(mana=jnp.array([0, 110, 80, 50, 250], jnp.int32))
    assert mask(ready)[cast]
    summoned, _ = step(ready, jnp.int32(cast))
    assert summoned.last_event == SUMMON_CAST and summoned.in_battle and not jnp.any(summoned.mana)
    assert [env.progression.rows[int(i)]['game_id'] for i in summoned.unit_ids[:6] if i] == ['g000uu8039']


def test_capital_guard_armor_regeneration_and_goal(default_game):
    env, initial, step, _ = default_game
    guard = DEFAULT['opponent_positions'][GUARD]
    assert not env.casting.field[GUARD]  # damage and weakening never target the capital
    blocked = env.territory.blocked_grid.reshape(50, 50)
    direction, (dr, dc) = next((i, d) for i, d in enumerate(DIRECTIONS)
                               if not blocked[guard[0]-d[0], guard[1]-d[1]])
    near = initial.replace(position=jnp.array([guard[0]-dr, guard[1]-dc], jnp.int32))
    battle, _ = step(near, jnp.int32(direction))
    assert battle.in_battle and battle.enemy == GUARD and battle.battle_fort_armor == CAPITAL_ARMOR
    # Rest: Myzrael +15% field +35% capital of 900; the Green Dragon heals fully.
    wounds = initial.enemy_wounds.at[GUARD, 4].set(800).at[DRAGON, 1].set(500)
    rested, _ = step(initial.replace(enemy_wounds=wounds), jnp.int32(REST))
    assert rested.enemy_wounds[GUARD, 4] == 350 and rested.enemy_wounds[DRAGON, 1] == 0
    # Victory: every stack but the capital guard, and every defended city.
    owned = jnp.ones_like(initial.city_owned)
    only_guard = jnp.zeros_like(initial.alive).at[GUARD].set(True)
    won, ts = step(initial.replace(alive=only_guard, city_owned=owned), jnp.int32(REST))
    assert won.won and won.done and float(ts.reward) > 2
    one_left = only_guard.at[0].set(True)
    open_, _ = step(initial.replace(alive=one_left, city_owned=owned), jnp.int32(REST))
    assert not open_.won and not open_.done
