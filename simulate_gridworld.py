"""Simulate a fixed policy on a GridWorld with the Part B seed scheme."""
import random

BATCH_SEEDS = [11, 22, 33]
EPISODES = 50
MAX_STEPS = 100


def episode_seed(batch_seed, i):
    return 1000 * batch_seed + i


def run_episode(world, policy, seed, max_steps=MAX_STEPS):
    rng = random.Random(seed)
    s = world.start
    disc_return, factor = 0.0, 1.0
    trace = [s]
    for step in range(1, max_steps + 1):
        outcomes = world.transitions(s, policy[s])
        x, acc = rng.random(), 0.0
        chosen = outcomes[-1]
        for o in outcomes:
            acc += o[0]
            if x < acc:
                chosen = o
                break
        _, s, r = chosen
        disc_return += factor * r
        factor *= world.discount
        trace.append(s)
        if world.is_terminal(s):          # terminal on the last allowed step still counts
            outcome = "goal" if world.symbol(s) == "G" else "hole"
            return dict(steps=step, discounted_return=disc_return, outcome=outcome, trace=trace)
    return dict(steps=max_steps, discounted_return=disc_return, outcome="time_limit", trace=trace)


def run_batches(world, policy, batch_seeds=BATCH_SEEDS, episodes=EPISODES):
    rows = []
    for b in batch_seeds:
        for i in range(episodes):
            sd = episode_seed(b, i)
            res = run_episode(world, policy, sd)
            rows.append(dict(batch_seed=b, episode_seed=sd, **res))
    return rows


def summarize(rows):
    out = []
    for b in sorted({r["batch_seed"] for r in rows}):
        rs = [r for r in rows if r["batch_seed"] == b]
        n = len(rs)
        out.append(dict(
            batch_seed=b,
            goals=sum(r["outcome"] == "goal" for r in rs),
            holes=sum(r["outcome"] == "hole" for r in rs),
            time_limits=sum(r["outcome"] == "time_limit" for r in rs),
            mean_steps=sum(r["steps"] for r in rs) / n,
            mean_return=sum(r["discounted_return"] for r in rs) / n,
        ))
    return out
