"""Demonstrate the rules before trusting the solver. Output: outputs/rule_checks.txt"""
import math
from pathlib import Path
from gridworld import build_world, ACTIONS
from value_iteration import value_iteration, extract_policy
from simulate_gridworld import run_episode

OUT = Path(__file__).parent / "outputs"
lines = []


def log(msg=""):
    print(msg)
    lines.append(msg)


def check(cond, msg):
    log(("PASS  " if cond else "FAIL  ") + msg)
    assert cond, msg


paper = build_world("paper_map")
ext = build_world("extended_map")

log("== 1. Wall / boundary behaviour (paper map) ==")
t = paper.transitions((1, 1), "up")
log(f"(1,1) up   -> {t}")
check(t == [(1.0, (1, 1), -1)], "outside the map: stay at (1,1), reward of end tile '.' = -1")
t = paper.transitions((2, 1), "right")
log(f"(2,1) right-> {t}")
check(t == [(1.0, (2, 1), -1)], "wall at (2,2): stay at (2,1), reward -1")

log("\n== 2. Ice probabilities ==")
for name, w in (("paper", paper), ("extended", ext)):
    for s in w.states:
        if w.is_terminal(s):
            continue
        for a in ACTIONS:
            total = sum(p for p, _, _ in w.transitions(s, a))
            assert math.isclose(total, 1.0), (name, s, a, total)
    log(f"{name}: probabilities sum to 1 for every nonterminal (state, action)")
t = paper.transitions((2, 3), "up")
log(f"(2,3) ice, up -> {t}")
check(sorted(p for p, _, _ in t) == [0.1, 0.1, 0.8], "ice: 0.8 intended, 0.1 + 0.1 perpendicular")
check((0.8, (1, 3), -1) in t, "intended 'up' -> (1,3), reward -1")
check((0.1, (2, 3), -1) in t, "slip left hits wall (2,2): stay on ice, reward -1")
check((0.1, (2, 4), -10) in t, "slip right -> hole (2,4), reward -10")
t = paper.transitions((3, 1), "up")
check(t == [(1.0, (2, 1), -1)], "ordinary floor is deterministic")
check(ext.transitions((3, 2), "right") == [(1.0, (3, 3), -1)], "sand (3,2) is deterministic")
check(ext.transitions((3, 3), "right") == [(1.0, (3, 4), -5)], "water (3,4) is deterministic to enter")

log("\n== 3. Floor rewards (reward of the tile where the move ends) ==")
check(paper.transitions((1, 1), "right") == [(1.0, (1, 2), -3)], "entering mud M = -3")
check(ext.transitions((3, 1), "right") == [(1.0, (3, 2), -2)], "entering sand D = -2")
check(ext.transitions((3, 3), "right") == [(1.0, (3, 4), -5)], "entering water W = -5")
check(paper.transitions((1, 3), "right") == [(1.0, (1, 4), 10)], "entering goal G = +10 (no extra floor cost)")
check(ext.transitions((2, 3), "right")[-1][2] == -10 or any(r == -10 for _, _, r in ext.transitions((2, 3), "right")),
      "entering hole H = -10")
check(ext.transitions((3, 4), "up") == [(1.0, (2, 4), -10)], "moving from water W north into hole H = -10, deterministic")

log("\n== 4. Terminal handling ==")
for w, nm in ((paper, "paper"), (ext, "extended")):
    check(w.transitions((1, 4), "left") == [] and w.transitions((2, 4), "left") == [],
          f"{nm}: no transitions (hence no rewards) out of G or H")
    res = value_iteration(w)
    check(res["V"][(1, 4)] == 0.0 and res["V"][(2, 4)] == 0.0, f"{nm}: V(G) = V(H) = 0 after convergence")
    policy, _ = extract_policy(w, res["V"])
    ep = run_episode(w, policy, seed=1234)
    check(ep["trace"][-1] in {(1, 4), (2, 4)} or ep["outcome"] == "time_limit",
          f"{nm}: episode stops at the first terminal state (outcome={ep['outcome']}, steps={ep['steps']})")
    # discounted return must equal the rewards up to termination only
    tr, g, f = ep["trace"], 0.0, 1.0
    for a, b in zip(tr, tr[1:]):
        g += f * w.floor_of(b).reward
        f *= w.discount
    check(math.isclose(g, ep["discounted_return"]), f"{nm}: discounted return counts rewards only up to termination")
    check(sum(w.is_terminal(s) for s in tr[:-1]) == 0, f"{nm}: no terminal state appears before the last trace position")

OUT.mkdir(exist_ok=True)
(OUT / "rule_checks.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\nAll rule checks passed.")
