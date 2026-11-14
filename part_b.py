"""Part B: random vs fixed policy on Gymnasium FrozenLake-v1 (4x4), 4 conditions x 3 batches x 50 episodes.

Run:  python part_b.py
Writes to outputs/: part_b_episodes.csv, part_b_summary.md, part_b_traces.txt, part_b_versions.txt
"""
import csv
import random
import sys
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

MAP = ["SFFF",
       "FHFH",
       "FFFH",
       "HFFG"]
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
NAMES = {LEFT: "Left", DOWN: "Down", RIGHT: "Right", UP: "Up"}

# ---------------------------------------------------------------------------
# FIXED POLICY: one action for every nonterminal state. Decided BEFORE running
# any experiment and never changed. State index = 4 * row + col (0-indexed).
# Idea: shortest path avoiding holes, 0 -> 4 -> 8 -> 9 -> 13 -> 14 -> 15
# (down, down, right, down, right, right); off-path states steer back toward it.
# Path tiles 9 and 13 use 'down'/'right' because their perpendicular slips are
# not holes. Hole/goal states (5, 7, 11, 12, 15) need no action.
# TODO (yours): review this table and the reasoning; change it NOW if you want
# a different design, but never after looking at results.
# ---------------------------------------------------------------------------
FIXED_POLICY = {
    0: DOWN,    # S: go down the left column
    1: LEFT,    # back toward start (below is hole 5)
    2: LEFT,    # avoid the dangerous tile 6 (holes left and right of it)
    3: LEFT,    # below is hole 7
    4: DOWN,    # left column
    6: DOWN,    # toward row 3, then right
    8: RIGHT,   # leave column 0 (below is hole 12)
    9: DOWN,    # slips go to 8 / 10, never a hole
    10: DOWN,   # toward 14
    13: RIGHT,  # slips go to 9 / stay
    14: RIGHT,  # goal; slips go to 10 / stay
}

CONDITIONS = [(False, "random"), (False, "fixed"), (True, "random"), (True, "fixed")]
BATCH_SEEDS = [11, 22, 33]
EPISODES = 50
MAX_STEPS = 100


def make_env(slippery):
    return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery,
                    render_mode="ansi", max_episode_steps=MAX_STEPS)


def run_episode(env, policy_name, episode_seed, record_trace=False):
    """Episode runner built on reset()/step(). Stops on terminated or truncated."""
    rng = random.Random(episode_seed)                       # seeded action generator
    state, _ = env.reset(seed=episode_seed)
    total_reward, steps = 0.0, 0
    trace = [f"start (episode seed {episode_seed})\n{env.render()}"] if record_trace else None
    terminated = truncated = False
    while not (terminated or truncated):
        action = rng.randrange(4) if policy_name == "random" else FIXED_POLICY[state]
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if record_trace:
            trace.append(f"step {steps}: action {NAMES[action]} -> state {state}, reward {reward}\n{env.render()}")
    # Final state decides goal vs hole; terminated wins over truncated on the last step.
    letter = MAP[state // 4][state % 4]
    if terminated:
        outcome = "goal" if letter == "G" else "hole"
    else:
        outcome = "time_limit"
    return steps, total_reward, outcome, trace


def main():
    rows, examples = [], {}
    for slippery, policy_name in CONDITIONS:
        cond = f"slippery={slippery},policy={policy_name}"
        env = make_env(slippery)
        for b in BATCH_SEEDS:
            for i in range(EPISODES):
                sd = 1000 * b + i
                steps, total, outcome, trace = run_episode(env, policy_name, sd, record_trace=True)
                rows.append([cond, b, sd, policy_name, steps, total, outcome])
                examples.setdefault((cond, outcome), (sd, trace))   # keep first of each outcome
        env.close()

    with open(OUT / "part_b_episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "batch_seed", "episode_seed", "policy", "steps", "total_reward", "outcome"])
        w.writerows(rows)

    md = ["| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for slippery, policy_name in CONDITIONS:
        cond = f"slippery={slippery},policy={policy_name}"
        for b in BATCH_SEEDS:
            rs = [r for r in rows if r[0] == cond and r[1] == b]
            g = sum(r[6] == "goal" for r in rs)
            h = sum(r[6] == "hole" for r in rs)
            t = sum(r[6] == "time_limit" for r in rs)
            ms = sum(r[4] for r in rs) / len(rs)
            md.append(f"| {'yes' if slippery else 'no'} | {policy_name} | {b} | {g} | {h} | {t} "
                      f"| {100 * g / len(rs):.0f}% | {ms:.2f} |")
    (OUT / "part_b_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    with open(OUT / "part_b_traces.txt", "w", encoding="utf-8") as f:
        for slippery, policy_name in CONDITIONS:
            cond = f"slippery={slippery},policy={policy_name}"
            for outcome in ("goal", "hole", "time_limit"):
                f.write("=" * 70 + f"\n{cond} | first observed outcome: {outcome}\n")
                if (cond, outcome) in examples:
                    sd, trace = examples[(cond, outcome)]
                    f.write(f"episode seed {sd}\n\n" + "\n\n".join(trace) + "\n\n")
                else:
                    f.write("never occurred in these 150 episodes\n\n")

    (OUT / "part_b_versions.txt").write_text(
        f"python {sys.version.split()[0]}\ngymnasium {version('gymnasium')}\nnumpy {version('numpy')}\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
