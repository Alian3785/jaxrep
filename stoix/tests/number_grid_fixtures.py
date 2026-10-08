"""Fixed six-unit formation for independent combat regression cases."""
import json
from pathlib import Path

MAP = json.loads((Path(__file__).parent / 'fixtures/basic_combat_map.json').read_text())
