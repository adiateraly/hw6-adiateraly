"""Part C: value iteration on the paper map and the extended map, plus simulations."""
import csv
from pathlib import Path
from gridworld import build_world, ACTIONS, ARROWS
from value_iteration import (value_iteration, extract_policy, q_value,
                             format_values, format_policy)
from simulate_gridworld import run_batches, summarize

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
MAPS = {"paper": "paper_map", "extended": "extended_map"}
V2_POSITIONS = [(1, 2), (1, 3), (2, 3)]


def main():
    results, policies, worlds = {}, {}, {}
    summary_rows = []
    for name, key in MAPS.items():
        w = build_world(key)
        worlds[name] = w
        vi = value_iteration(w, tol=1e-6, max_iter=10000)
        policy, ties = extract_policy(w, vi["V"])
        results[name], policies[name] = vi, policy

        text = [f"Map: {name}", "\n".join("".join(r) for r in w.rows), ""]
        text += ["V1 (every nonterminal tile):", format_values(w, vi["history"][1]), ""]
        if name == "paper":
            text.append("V2 at the requested positions, computed from V1 (Q for each action):")
            V1 = vi["history"][1]
            for s in V2_POSITIONS:
                qs = {a: q_value(w, V1, s, a) for a in ACTIONS}
                text.append(f"  {s}: " + ", ".join(f"{a}={qs[a]:.4f}" for a in ACTIONS)
                            + f"  -> V2 = {max(qs.values()):.4f}")
            text.append("")
        text += [
            f"Converged: {vi['converged']}  iterations: {vi['iterations']}  "
            f"final max change: {vi['final_delta']:.3e}",
            "", "Final values V*:", format_values(w, vi["V"]), "",
            "Greedy policy (ties broken by order up, right, down, left):",
            format_policy(w, policy), "", "Tied best actions (within 1e-9):",
        ]
        for s in sorted(ties):
            if len(ties[s]) > 1:
                text.append(f"  {s}: " + ", ".join(ties[s]))
        if not any(len(t) > 1 for t in ties.values()):
            text.append("  none")
        (OUT / f"part_c_{name}_map.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
        print("\n".join(text), "\n" + "=" * 60)

        rows = run_batches(w, policy)
        with open(OUT / f"part_c_{name}_episodes.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["map", "batch_seed", "episode_seed", "steps", "discounted_return", "outcome"])
            for r in rows:
                wr.writerow([name, r["batch_seed"], r["episode_seed"], r["steps"],
                             f"{r['discounted_return']:.6f}", r["outcome"]])
        for s in summarize(rows):
            summary_rows.append((name, s))

    md = ["| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in summary_rows:
        md.append(f"| {name} | {s['batch_seed']} | {s['goals']} | {s['holes']} | {s['time_limits']} "
                  f"| {s['mean_steps']:.2f} | {s['mean_return']:.4f} |")
    (OUT / "part_c_simulation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    # Comparison of the two maps (tiles that exist in both with the same symbol)
    p, e = worlds["paper"], worlds["extended"]
    cmp = ["Comparison paper map -> extended map (tiles whose letter changed: (3,2) . -> D, (3,4) . -> W)", ""]
    cmp.append("State | paper V* | extended V* | change | paper action | extended action")
    for s in p.states:
        if p.is_terminal(s):
            continue
        dv = results["extended"]["V"][s] - results["paper"]["V"][s]
        pa, ea = policies["paper"][s], policies["extended"][s]
        flag = "  <- action changed" if pa != ea else ""
        cmp.append(f"{s} | {results['paper']['V'][s]:.4f} | {results['extended']['V'][s]:.4f} | "
                   f"{dv:+.4f} | {ARROWS[pa]} {pa} | {ARROWS[ea]} {ea}{flag}")
    (OUT / "part_c_comparison.txt").write_text("\n".join(cmp) + "\n", encoding="utf-8")
    print("\n".join(cmp))


if __name__ == "__main__":
    main()
