"""Pin the native Empire-Alliance War scenario for its event-free Capital port."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct

from disciples_scenario import read_scenario, records
from inspect_unit_sources import dbf

ROOT = Path(__file__).resolve().parents[1]
NULL = 'G000000000'


def inspect(game):
    data, source = read_scenario(game, 'exports/empire-alliance war.sg')
    rows = records(data)
    source['sha256'] = hashlib.sha256(data).hexdigest()
    types = defaultdict(list)
    for row in rows:
        types[row.kind].append(row)
    files = {p.stem.lower(): p for p in (game / 'Globals').iterdir() if p.suffix.lower() == '.dbf'}
    globals_data = {name: dbf(files[name]) for name in ('glmark', 'gitem', 'gspells', 'gmodif', 'gmodifl', 'lmodife')}
    source['globals_sha256'] = {name: hashlib.sha256(files[name].read_bytes()).hexdigest() for name in globals_data}
    text_ids = {r[key].lower() for kind in ('gitem', 'gspells') for r in globals_data[kind] for key in ('NAME_TXT', 'DESC_TXT')}
    globals_data['texts'] = {r['TXT_ID'].lower(): r['TEXT'] for r in dbf(files['tglobal']) if r['TXT_ID'].lower() in text_ids}
    source['globals_sha256']['tglobal'] = hashlib.sha256(files['tglobal'].read_bytes()).hexdigest()
    info = types['.?AVCScenarioInfo@@'][0]
    size = info.integer('MAP_SIZE')
    terrain = [[None] * size for _ in range(size)]
    for block in types['.?AVCMidgardMapBlock@@']:
        by, bx = int(block.id[-4:-2], 16), int(block.id[-2:], 16)
        words = struct.unpack('<32I', block.blob('BLOCKDATA'))
        for i, word in enumerate(words):
            y, x = by + i // 8, bx + i % 8
            if terrain[y][x] is not None:
                raise ValueError('Overlapping terrain blocks')
            terrain[y][x] = word & 255
    if any(value is None for row in terrain for value in row):
        raise ValueError('Missing terrain cells')
    instances = {r.text('ITEM_ID').upper(): r.text('ITEM_TYPE').upper() for r in types['.?AVCMidItem@@']}

    def inventory(row):
        return dict(Counter(instances[key] for key in row.ids('ITEM_ID')))

    units = {r.id.upper(): dict(source_id=r.id, game_id=r.text('TYPE').upper(), level=r.integer('LEVEL'),
                               hp=r.integer('HP'), xp=r.integer('XP'), name=r.text('NAME_TXT'),
                               modifiers=r.ids('MODIF_ID')) for r in types['.?AVCMidUnit@@']}
    forts = []
    for kind, capital in (('.?AVCCapital@@', True), ('.?AVCMidVillage@@', False)):
        for r in types[kind]:
            x, y = r.position
            forts.append(dict(source_id=r.id, name=r.text('NAME_TXT').strip(), position=r.position,
                              marker_position=[x + (4 if capital else 3), y + (2 if capital else 3)],
                              size=5 if capital else 4, kind='capital' if capital else 'city',
                              level=5 if capital else r.integer('SIZE'), owner=r.text('OWNER'),
                              visitor=r.text('STACK'), inventory=inventory(r)))
    ruins = []
    for r in types['.?AVCMidRuin@@']:
        x, y = r.position
        ruins.append(dict(source_id=r.id, name=r.text('TITLE').strip(), position=r.position,
                          marker_position=[x + 2, y + 2], size=3, kind='ruin',
                          gold=int(r.text('CASH')[1:5]), reward=r.text('ITEM').upper()))
    buildings = {r['source_id']: r for r in forts + ruins}
    armies = []
    for r in rows:
        if r.kind not in ('.?AVCCapital@@', '.?AVCMidVillage@@', '.?AVCMidRuin@@', '.?AVCMidStack@@'):
            continue
        declared = [r.text(f'UNIT_{i}').upper() for i in range(6)]
        formation = [r.integer(f'POS_{i}') for i in range(6)]
        unit_cells = defaultdict(list)
        for cell, index in enumerate(formation):
            if index >= 0:
                if declared[index] == NULL:
                    raise ValueError('Formation refers to an empty unit')
                unit_cells[declared[index]].append(cell)
        if set(declared) - {NULL} != set(unit_cells):
            raise ValueError('Formation and declared units disagree')
        if not unit_cells:
            continue
        field = r.kind == '.?AVCMidStack@@'
        building = buildings.get(r.text('INSIDE') if field else r.id)
        armies.append(dict(source_id=r.id, position=r.position,
                           marker_position=building['marker_position'] if building else r.position,
                           kind=building['kind'] if building else 'field',
                           building=building['source_id'] if building else None,
                           owner=r.text('OWNER') if field else building.get('owner', NULL),
                           leader=r.text('LEADER_ID') if field else None,
                           inventory=inventory(r) if field or r.kind != '.?AVCMidRuin@@' else {},
                           units=[dict(units[key], cells=cells) for key, cells in unit_cells.items()]))
    sites = []
    for kind, label in (('.?AVCMidSiteMerchant@@', 'merchant'), ('.?AVCMidSiteMage@@', 'mage'),
                        ('.?AVCMidSiteMercs@@', 'mercenary'), ('.?AVCMidSiteTrainer@@', 'trainer')):
        for r in types[kind]:
            site = dict(source_id=r.id, name=r.text('TXT_TITLE').strip(), position=r.position, size=3, kind=label)
            if label == 'merchant':
                entries = re.findall(rb'ITEM_ID\x0b\0\0\0(G[0-9A-Za-z]{9})\0ITEM_COUNT(.{4})', r.data, re.DOTALL)
                if len(entries) != r.integer('QTY_ITEM'):
                    raise ValueError('Merchant stock mismatch')
                site['stock'] = {key.decode().upper(): struct.unpack('<i', count)[0] for key, count in entries}
            elif label == 'mage':
                site['spells'] = r.ids('SPELL_ID')
            elif label == 'mercenary':
                site['units'] = [dict(game_id=key.decode().upper(), level=struct.unpack('<i', level)[0], unique=bool(unique[0]))
                                 for key, level, unique in re.findall(rb'UNIT_ID\x0b\0\0\0(G[0-9A-Za-z]{9})\0UNIT_LEVEL(.{4})UNIT_UNIQ(.)', r.data, re.DOTALL)]
            sites.append(site)
    landmarks = {r['LMARK_ID'].upper(): r for r in globals_data['glmark']}
    players = [dict(source_id=r.id, race=r.text('RACE_ID'), bank=r.text('BANK'), lord=r.text('LORD_ID'))
               for r in types['.?AVCMidPlayer@@']]
    return dict(schema_version=1, source=source, size=size, title=info.text('NAME'),
                original_objective=info.text('BRIEFING'), limits={key: info.integer(key) for key in ('MAX_UNIT', 'MAX_SPELL', 'MAX_LEADER', 'MAX_CITY')},
                terrain=terrain, roads=[r.position for r in types['.?AVCMidRoad@@']],
                landmarks=[dict(source_id=r.id, position=r.position, size=[int(landmarks[r.text('TYPE')][key]) for key in ('CX', 'CY')])
                           for r in types['.?AVCMidLandmark@@']],
                armies=armies, forts=forts, ruins=ruins, sites=sites, players=players,
                chests=[dict(source_id=r.id, position=r.position, inventory=inventory(r)) for r in types['.?AVCMidBag@@']],
                resources=[dict(source_id=r.id, position=r.position, resource=r.integer('RESOURCE')) for r in types['.?AVCMidCrystal@@']],
                events=[dict(source_id=r.id, name=r.text('NAME_TXT')) for r in types['.?AVCMidEvent@@']],
                initial_buildings={r.id: r.ids('BUILD_ID') for r in types['.?AVCPlayerBuildings@@']},
                initial_spells={r.id: r.ids('SPELL_ID') for r in types['.?AVCPlayerKnownSpells@@']},
                globals=globals_data, object_counts={key: len(value) for key, value in types.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/maps/empire_alliance_war/source.json')
    args = parser.parse_args()
    snapshot = inspect(args.game)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(size=snapshot['size'], armies=len(snapshot['armies']),
                         own=[a for a in snapshot['armies'] if a['owner'] == 'S143PL0001'],
                         spells=snapshot['initial_spells'], buildings=snapshot['initial_buildings']), ensure_ascii=False))
