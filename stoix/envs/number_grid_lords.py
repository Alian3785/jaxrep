"""Ruler bonuses from the installed manual and Python reference."""
LORDS = {
    'warrior': dict(name='Воин', regeneration=15, research_divisor=1, max_spell_level=4,
        city_cost_divisor=1, starting_building=None,
        description='Регенерация отряда +15% в день. Изучение заклинаний I–IV уровней.'),
    'mage': dict(name='Маг', regeneration=0, research_divisor=2, max_spell_level=5,
        city_cost_divisor=1, starting_building='Башня магии',
        description='Изучение заклинаний I–V уровней за половину маны. Башня магии построена с начала игры.'),
    'guildmaster': dict(name='Глава гильдии', regeneration=0, research_divisor=1, max_spell_level=4,
        city_cost_divisor=2, starting_building='Гильдия',
        description='Улучшение городов за половину золота. Гильдия построена с начала игры. Изучение заклинаний I–IV уровней. Особые действия воров пока недоступны.'),
}


def lord_settings(game_map):
    kind = game_map.get('lord_type', 'warrior')
    if not isinstance(kind, str) or kind not in LORDS:
        raise ValueError('Unknown lord_type')
    return dict(id=kind, **LORDS[kind])
