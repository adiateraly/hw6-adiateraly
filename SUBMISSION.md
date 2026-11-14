# HW6 submission

## AI-use declaration

I used **Claude (Anthropic)** for these things:

- **Code.** Claude wrote the Python code for Part B and Part C.

I ran all programs myself on my computer. All numbers in `outputs/` come from my runs.

## Setup and reproduction

Versions: Python 3.13.9, gymnasium 1.4.0, numpy 2.5.4 (see `outputs/part_b_versions.txt`). Part C uses only standard Python.

Commands (Windows PowerShell, in the repository folder):

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python rule_checks.py
python part_c.py
python part_b.py
```

## Part A — Paper calculations

Photos of my paper work (committed before the code):
[page 1](part_a/part_a_1.jpg), [page 2](part_a/part_a_2.jpg), [page 3](part_a/part_a_3.jpg), [page 4](part_a/part_a_4.jpg)

**1. The problem in my words.**
- **States:** the 11 tiles that are not a wall. 9 are normal tiles. 2 are end tiles (G and H).
- **Actions:** up, right, down, left.
- **Moves:** on normal tiles and mud, the move always works. On ice, 80% of the time I go where I want. 10% I slip to the left side, 10% to the right side. If I hit a wall or the edge, I stay in the same tile.
- **Rewards:** I get the reward of the tile where I end: `.` and S = −1, M = −3, I = −1, G = +10, H = −10.
- **The agent wants:** the biggest total reward over time. Reward now is worth more than reward later (we multiply by 0.9 each step).

**2. V1** (at the start V0 = 0, so V1 is the best reward after one move):

```
 -1 | -1 | 10 |  G
 -1 |  # | -1 |  H
 -1 | -1 | -1 | -1
```

At the ice tile (2,3) the four actions give: up −1.9, right −8.2, down −1.9, left −1. The best is left, so V1(2,3) = −1.

**3. V2 from V1** (all four actions at each position):

| Position | up | right | down | left | V2 | Best |
| --- | --- | --- | --- | --- | --- | --- |
| (1,2) mud | −3.9 | 8 | −3.9 | −1.9 | **8** | → |
| (1,3) | 8 | 10 | −1.9 | −3.9 | **10** | → |
| (2,3) ice | 5.21 | −7.39 | −2.71 | −0.91 | **5.21** | ↑ |

Example, ice tile, action "up":
0.8·(−1 + 0.9·10) + 0.1·(−1 + 0.9·(−1)) + 0.1·(−10) = 6.4 − 0.19 − 1.0 = 5.21.
The other calculations are in the photos.

**4. Arrows and ties.** (1,2) → , (1,3) → , (2,3) ↑. There are no ties.

**Why are two rounds not enough?** V2 looks only two moves ahead. Tiles far from the goal do not know about the goal yet. Their values and best actions can still change. We can trust the policy only when the values stop changing.

## Part B — FrozenLake

### Fixed policy, written before the experiments

Tile number = 4 × row + column (start from 0). Holes (5, 7, 11, 12) and the goal (15) need no action.

| Tile | Action | Reason |
| --- | --- | --- |
| 0 (start) | Down | go down the left side |
| 4 | Down | keep going down |
| 8 | Right | leave the left side (below is hole 12) |
| 9 | Down | side slips go to 8 or 10, not to a hole |
| 13 | Right | side slips are safe |
| 14 | Right | go to the goal |
| 1 | Left | go back to the start (below is hole 5) |
| 2 | Left | avoid tile 6 (holes on both sides) |
| 3 | Left | below is hole 7 |
| 6 | Down | go to the bottom row |
| 10 | Down | go to tile 14 |

Main route: 0 → 4 → 8 → 9 → 13 → 14 → 15 (6 steps). This policy was made for the **not slippery** lake. I did not change it for the slippery lake.

### Results

One row = one batch of 50 episodes. Episode seed = 1000 × batch seed + i.

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no | random | 11 | 0 | 50 | 0 | 0% | 8.14 |
| no | random | 22 | 1 | 49 | 0 | 2% | 7.56 |
| no | random | 33 | 0 | 50 | 0 | 0% | 6.88 |
| no | fixed | 11 | 50 | 0 | 0 | 100% | 6.00 |
| no | fixed | 22 | 50 | 0 | 0 | 100% | 6.00 |
| no | fixed | 33 | 50 | 0 | 0 | 100% | 6.00 |
| yes | random | 11 | 0 | 50 | 0 | 0% | 8.06 |
| yes | random | 22 | 0 | 50 | 0 | 0% | 7.60 |
| yes | random | 33 | 0 | 50 | 0 | 0% | 7.98 |
| yes | fixed | 11 | 1 | 49 | 0 | 2% | 5.22 |
| yes | fixed | 22 | 2 | 48 | 0 | 4% | 6.00 |
| yes | fixed | 33 | 1 | 49 | 0 | 2% | 5.14 |

Files in `outputs/`:
- [all 600 episodes](outputs/part_b_episodes.csv)
- [summary table](outputs/part_b_summary.md)
- [example traces](outputs/part_b_traces.txt) (the `[41m` signs are terminal colors that show where the agent is)
- [versions](outputs/part_b_versions.txt)

### What the results show

**What slipping changed.**
- The fixed policy had **150 goals out of 150** without slipping. With slipping it had only **4 goals out of 150** (about 3%). The policy was made for the not slippery lake. On the slippery lake the agent often goes to the side. Some tiles on my route are next to a hole. For example, at tile 4 "Down" can slip right into hole 5 (1 chance in 3). At tile 8 "Right" can slip down into hole 12 (1 chance in 3).
- For the **random** policy, slipping changed almost nothing (1 goal and 0 goals out of 150). This is normal: a random action plus a random slip is still a random direction. The small difference is luck.

**Are the batches the same?**
- Without slipping the fixed policy always does the same thing, so all 3 batches are the same (50 goals, 6.00 steps).
- With slipping the fixed policy got 1, 2 and 1 goals. The numbers are close, but they are very small, so luck matters. The random policy with slipping got 0 goals in every batch.

**Why is one successful episode not enough?** Even the random policy reached the goal one time without slipping (seed 22006, 17 steps). That was luck. The slippery fixed policy won only 4 times in 150. One success can be luck, so we need many episodes in many batches.

**Example traces** (in `outputs/part_b_traces.txt`):
- *Success, no slipping:* fixed policy, seed 11000. Down, Down, Right, Down, Right, Right. It reached the goal in 6 steps.
- *Success, slippery:* fixed policy, seed 11020, 14 steps. The agent slipped many times (for example, at tile 4 it slipped into the wall and stayed). It was lucky and did not fall in a hole.
- *Failure, slippery:* fixed policy, seed 11000. The agent went back to the start, then at tile 4 it slipped right into hole 5 after 8 steps.
- *Failure, no slipping:* random policy, seed 11000. It fell into hole 5 after 6 steps.
- *Time limit:* this never happened in 600 episodes, so I have no trace for it.

**Why is this not training?** The fixed policy is chosen one time and never changes. The agent does not learn from rewards. Nothing is updated between episodes. We only measure how good the policy is. The random policy does not learn too.

## Part C — My solver

### Compare code with paper

The program output ([paper map](outputs/part_c_paper_map.txt)) is the same as my paper work:

| Item | Paper | Code |
| --- | --- | --- |
| V1, all normal tiles | −1, only V1(1,3) = 10 | same |
| V2(1,2) | 8 | 8.0000 |
| V2(1,3) | 10 | 10.0000 |
| V2(2,3) | 5.21 | 5.2100 |
| (2,3) up / right / down / left | 5.21 / −7.39 / −2.71 / −0.91 | same |

There are no differences, so I did not need to fix anything.

### Values, policies and convergence

Files: [paper map](outputs/part_c_paper_map.txt), [extended map](outputs/part_c_extended_map.txt), [comparison](outputs/part_c_comparison.txt).

| Map | Rounds (iterations) | Last biggest change | Finished? |
| --- | --- | --- | --- |
| Paper map | 12 | 2.165e-07 | yes |
| Extended map | 11 | 2.406e-07 | yes |

I stop when the biggest change is smaller than 0.000001. The safety limit is 10,000 rounds. We never reached it.

Final values:

```
Paper map                      Extended map
 4.200 | 8.000 | 10.000 | G    4.200 | 8.000 | 10.000 | G
 2.780 |   #   |  5.824 | H    2.780 |   #   |  5.824 | H
 1.536 | 2.818 |  4.242 | 2.818  1.502 | 2.818 |  4.242 | 2.818
```

Best actions (policy):

```
Paper map     Extended map
→ → → G       → → → G
↑ # ↑ H       ↑ # ↑ H
→ → ↑ ←       ↑ → ↑ ←
```

**Tie rule:** I take the policy from the final values. If two actions have almost the same value (difference smaller than 0.000000001), I take the first one in this order: up, right, down, left. There were no ties on either map.

### Simulation results

I ran the policy for 3 batches of 50 episodes (seeds 11, 22, 33; episode seed = 1000 × batch seed + i). The limit is 100 steps. Discounted return = r0 + 0.9·r1 + 0.9²·r2 + …

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper | 11 | 42 | 8 | 0 | 4.98 | 0.8913 |
| paper | 22 | 46 | 4 | 0 | 5.04 | 1.9148 |
| paper | 33 | 48 | 2 | 0 | 5.04 | 2.4921 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |

Episode files: [paper map](outputs/part_c_paper_episodes.csv), [extended map](outputs/part_c_extended_episodes.csv). Table file: [summary](outputs/part_c_simulation_summary.md).

### Explain your results

**What changed with sand and water?**
- Only one value and one action changed. At tile (3,1) the value went from 1.5358 to 1.5020, and the best action went from → to ↑. All other values are the same.
- *Sand (3,2), cost −2:* on the paper map the best way went right along the bottom row and then up through the ice tile (2,3). Sand makes the first step more expensive. So now it is better to go up the left side and through the mud (1.502 for the mud way, 0.536 for the sand way).
- *Water (3,4), cost −5:* water changed nothing, because the best policy never goes into tile (3,4). We pay the reward of the tile where the move ends, so leaving the water tile costs nothing. Its value is the same (2.8176).

**Where did cost and risk matter?**
- On the paper map the way through the ice is a little better on average (1.536 against 1.502), but it is risky. Each time on ice there is a 10% chance to slip into the hole. In the simulation 14 episodes out of 150 ended in a hole. The mean return changed from 0.89 to 2.49 between batches.
- On the extended map the policy takes the mud way. There is no ice on this way. All 150 episodes took the same path (5 steps) and got exactly 1.502, the same as the value of the start tile.

**How does the policy handle ice?** At the ice tile (2,3) the policy always goes up, toward the goal. It never goes right or down, because these moves can end in the hole. On the extended map the policy does not touch the ice at all.

**How do simulations connect to values?**
- The value V* is the *average* total reward if we could play forever. A simulation is only a small sample: at most 100 steps and 50 episodes per batch. (0.9 to the power 100 is almost zero, so the 100-step limit does not matter.)
- On the paper map the batch averages (0.89, 1.91, 2.49) are around the value 1.536 because holes are rare but cost a lot (−10). On the extended map there is no luck, so every episode gives exactly the value of the start tile.
- So simulations only estimate the value. They are not the same thing. Also, my solver plans with the known rules. It does not learn from data.

### Rule checks

All checks passed. Full result: [outputs/rule_checks.txt](outputs/rule_checks.txt).

- **Walls and edge:** from (1,1) "up" stays in (1,1), reward −1. From (2,1) "right" hits the wall and stays in (2,1), reward −1.
- **Ice chances:** for every tile and action, the chances add up to 1. On ice, "up" at (2,3) gives 0.8 to (1,3), 0.1 stay (wall), 0.1 hole (−10).
- **Floor rewards:** mud −3, sand −2, water −5, goal +10, hole −10. Sand and water moves always work (no slipping).
- **End tiles:** there are no moves and no rewards after G or H. Their value is 0. The episode stops at the first end tile. The discounted return counts only rewards until the end.