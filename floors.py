"""Floor rules, kept separate from the solver.

All tile rules live in data/maps.json. To add a new floor type, add one entry
there (reward, movement, ...). Neither gridworld.py nor value_iteration.py
needs to change.
"""
import json
from dataclasses import dataclass
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "maps.json"


@dataclass(frozen=True)
class Floor:
    symbol: str
    reward: float = 0.0          # reward when a move ENDS on this tile
    traversable: bool = True     # False for walls
    terminal: bool = False       # True for goal / hole
    slippery: bool = False       # True for ice
    intended: float = 1.0        # probability of the chosen direction
    perpendicular_each: float = 0.0  # probability of each perpendicular direction


def load_config(path=DATA_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_floors(config=None):
    config = config or load_config()
    floors = {}
    for sym, t in config["tiles"].items():
        slippery = t.get("movement") == "slippery"
        floors[sym] = Floor(
            symbol=sym,
            reward=t.get("reward", 0.0),
            traversable=t.get("traversable", True),
            terminal=t.get("terminal", False),
            slippery=slippery,
            intended=t.get("intended", 1.0),
            perpendicular_each=t.get("perpendicular_each", 0.0),
        )
    return floors
