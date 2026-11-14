"""Grid World MDP with a known transition model.

Coordinates are 1-indexed (row, column) in everything printed or stored,
matching the assignment. Internally states are (row, col) tuples, also 1-indexed.
"""
from floors import load_config, load_floors

ACTIONS = ["up", "right", "down", "left"]          # also the tie-breaking order
DELTAS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
PERPENDICULAR = {
    "up": ("left", "right"), "down": ("left", "right"),
    "left": ("up", "down"), "right": ("up", "down"),
}
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}


class GridWorld:
    def __init__(self, rows, floors, discount):
        self.rows = [list(r) for r in rows]
        self.floors = floors
        self.discount = discount
        self.n_rows = len(self.rows)
        self.n_cols = len(self.rows[0])
        self.states = [
            (r, c)
            for r in range(1, self.n_rows + 1)
            for c in range(1, self.n_cols + 1)
            if self.floor(r, c).traversable
        ]
        self.start = next(
            (r, c) for r in range(1, self.n_rows + 1)
            for c in range(1, self.n_cols + 1) if self.rows[r - 1][c - 1] == "S"
        )

    def symbol(self, state):
        return self.rows[state[0] - 1][state[1] - 1]

    def floor(self, r, c):
        return self.floors[self.rows[r - 1][c - 1]]

    def floor_of(self, state):
        return self.floor(*state)

    def is_terminal(self, state):
        return self.floor_of(state).terminal

    def _move(self, state, direction):
        """Deterministic result of trying to go in `direction`; walls/edges -> stay."""
        dr, dc = DELTAS[direction]
        r, c = state[0] + dr, state[1] + dc
        if 1 <= r <= self.n_rows and 1 <= c <= self.n_cols and self.floor(r, c).traversable:
            return (r, c)
        return state

    def direction_probabilities(self, state, action):
        """{direction: probability}, determined by the tile the move STARTS on."""
        f = self.floor_of(state)
        if not f.slippery:
            return {action: 1.0}
        p1, p2 = PERPENDICULAR[action]
        return {action: f.intended, p1: f.perpendicular_each, p2: f.perpendicular_each}

    def transitions(self, state, action):
        """List of (probability, next_state, reward). Empty for terminal states.

        Reward = reward of the tile where the move ENDS (also when blocked and
        the agent stays in place). Each slip direction is resolved separately.
        """
        if self.is_terminal(state):
            return []
        out = []
        for direction, p in self.direction_probabilities(state, action).items():
            nxt = self._move(state, direction)
            out.append((p, nxt, self.floor_of(nxt).reward))
        return out


def build_world(map_key, config=None):
    config = config or load_config()
    return GridWorld(config[map_key], load_floors(config), config["discount"])
