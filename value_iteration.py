"""Value iteration written from scratch for a GridWorld with a known model."""
from gridworld import ACTIONS, ARROWS

TIE_EPS = 1e-9  # actions whose Q differs by less than this count as tied


def q_value(world, V, state, action):
    return sum(p * (r + world.discount * V[s2])
               for p, s2, r in world.transitions(state, action))


def one_sweep(world, V):
    """One full synchronous update: uses ONLY the previous round's values."""
    new = {}
    for s in world.states:
        if world.is_terminal(s):
            new[s] = 0.0
        else:
            new[s] = max(q_value(world, V, s, a) for a in ACTIONS)
    return new


def value_iteration(world, tol=1e-6, max_iter=10000):
    V = {s: 0.0 for s in world.states}            # V0 = 0 everywhere
    history = [dict(V)]                            # history[k] = V_k
    delta, converged, iterations = float("inf"), False, 0
    while iterations < max_iter:
        new = one_sweep(world, V)
        delta = max(abs(new[s] - V[s]) for s in world.states)
        V = new
        iterations += 1
        history.append(dict(V))
        if delta < tol:
            converged = True
            break
    return {"V": V, "history": history, "iterations": iterations,
            "final_delta": delta, "converged": converged}


def extract_policy(world, V):
    """Greedy policy w.r.t. V. Ties are broken by ACTIONS order: up, right, down, left."""
    policy, ties = {}, {}
    for s in world.states:
        if world.is_terminal(s):
            continue
        qs = {a: q_value(world, V, s, a) for a in ACTIONS}
        best = max(qs.values())
        tied = [a for a in ACTIONS if best - qs[a] <= TIE_EPS]
        policy[s] = tied[0]
        ties[s] = tied
    return policy, ties


def format_values(world, V, decimals=3):
    width = decimals + 5
    lines = []
    for r in range(1, world.n_rows + 1):
        cells = []
        for c in range(1, world.n_cols + 1):
            s = (r, c)
            sym = world.rows[r - 1][c - 1]
            if not world.floor(r, c).traversable:
                cells.append("#".center(width))
            elif world.is_terminal(s):
                cells.append(f"{sym}".center(width))
            else:
                cells.append(f"{V[s]:.{decimals}f}".rjust(width))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def format_policy(world, policy):
    lines = []
    for r in range(1, world.n_rows + 1):
        cells = []
        for c in range(1, world.n_cols + 1):
            s = (r, c)
            if not world.floor(r, c).traversable:
                cells.append("#")
            elif world.is_terminal(s):
                cells.append(world.rows[r - 1][c - 1])
            else:
                cells.append(ARROWS[policy[s]])
        lines.append("  ".join(cells))
    return "\n".join(lines)
