# PhemBots

**A stigmergy-inspired multi-robot coordination system for emergent coverage and dynamic event response.**

PhemBots coordinates a team of three physical robots that must keep an area evenly covered while responding to localised events ("mess") of different priorities. The robots never communicate with each other and follow no hard-coded response rules. Instead they all read a shared **digital pheromone heatmap**, so coordination emerges from the environment, in the same way ants and termites coordinate. A more severe event draws more robots to it, and a minor one draws only the nearest.

This repository contains the code from my MSc Robotics dissertation (University of Bristol & UWE Bristol, 2026), including the two baseline systems it was compared against and the notebook used to analyse the trial data.

## How it works

**Shared heatmap.** A 30x30 grid represents the arena. It is the sum of two layers:

- *Accumulation layer*: cells a robot has just visited drop to a minimum value and then slowly recover to a cap, like evaporating pheromones. This pulls robots toward unvisited or long-untouched areas, giving even coverage without explicit rules.
- *Mess layer*: an event adds a large value (red 1500, green 700, blue 80), which is scaled by 3 when the layers are combined, so urgent events dominate the landscape.

`H_final = (3 x M_mess) + M_accum`

**Reward-weighted path planning.** Each robot runs a modified Dijkstra search on its own M5Stack. Instead of minimising cost, it routes through high-value cells, using an edge cost of `step x 50 / (1 + 0.8 x reward)`. It then drives the first four cells of the best path, recalculates, and repeats.

**Theory-of-mind discount.** Each robot simulates the likely path of every teammate using the same heatmap and the same planner. Cells a teammate would reach first have their value multiplied by 0.4 within a 4-cell radius, then the robot replans. No messages are sent between robots. This produces emergent spacing and collision avoidance, and it is why a red event pulls all three robots while a green event pulls only the two nearest, because the third predicts that it will lose the race and carries on covering.

## System architecture

Three programs run at the same time:

| Layer | Hardware | Code | Role |
|---|---|---|---|
| Laptop | PC with overhead camera tracking | `PhemBotsVS/PhemBotsVS.py` | Builds the heatmap, reads robot poses from the tracker, paints and schedules mess events, broadcasts the heatmap and robot positions over WiFi (UDP), logs each trial to CSV |
| Decision-making | M5Stack Core2 on each robot | `PhemBotsM5/PhemBotsM5.ino` | Receives the heatmap, runs the planner and theory-of-mind discount, converts the path into turn and drive commands |
| Motor control | Pololu 3pi+ 32U4 | 3pi_code_used_for_all_systems/ | Executes turn and drive commands over I2C using wheel encoders, and reports whether it has finished when the M5Stack polls it |

Robot position and heading come from an overhead camera tracking system that reads an ArUco marker shown on each M5Stack's screen. Each robot's marker ID is taken from the last number of its IP address.

## Baselines

Both baselines use the same 3pi+ firmware and the same tracking, and differ only in how targets are chosen:

- **Greedy auction** (`auction_system_A/auction_system_A.py`): the laptop assigns each robot a target cell based on value minus distance, with a minimum separation between targets.
- **Boustrophedon sweep** (`boustrophedon_waypoints_B/boustrophedon_waypoints_B.py`): the arena is split into three vertical strips, and each robot sweeps its strip in a fixed pattern and ignores events.

Both send a single target cell to the robots, which `Recieve_target_cell_used_for_A_and_B/` turns into motion.

## Results

Each system and condition was run for 10 trials of 180 seconds on the physical robots (70 trials in total). The conditions were no mess, a fixed-schedule mess, and a random mess. The boustrophedon baseline was tested only under the random condition.

- PhemBots matched the auction baseline on coverage speed and mess response time.
- It was significantly better on boundary adherence, inter-robot spacing, collision avoidance and reward efficiency (Mann-Whitney U, p < 0.05 for all comparisons except near collisions against the boustrophedon baseline under random mess).
- Reward efficiency per millimetre travelled was about 131-135 for PhemBots, compared with about 118-122 for the auction and about 103 for boustrophedon.
- Without mess, the mean near-collision count was about 4 for PhemBots and about 45 for the auction baseline.
- The number of responding robots scaled with event severity without any explicit rule. Neither baseline can do this.

Figures from the dissertation are in the `figures/` folder.

## Repository layout

```
PhemBotsVS/                          Laptop program (Python): heatmap, receives tracking data, broadcast, logging
PhemBotsM5/                          M5Stack sketch: path planning and theory of mind
3pi_code_used_for_all_systems/       Pololu 3pi+ motor control sketch
auction_system_A/                    Baseline A: greedy auction (laptop program)
boustrophedon_waypoints_B/           Baseline B: boustrophedon sweep (laptop program)
Recieve_target_cell_used_for_A_and_B/   M5Stack sketch used by both baselines
data_processing/                     Notebook with the statistics and plots for the trial data
figures/                             Figures from the dissertation
```

## Running it

This system depends on the lab's physical setup, so it cannot be run without the robots, but this is how it fitted together.

1. Flash `3pi_code_used_for_all_systems` to each 3pi+ (Arduino IDE with the Pololu 3pi+ 32U4 library).
2. Flash the M5Stack sketch (`PhemBotsM5` or the baseline receiver) with the M5Unified library. Set your WiFi name and password at the top of `setup()`.
3. Start the overhead tracking server (see credit below) and set the tracker IP in the Python program and the sketches.
4. Set the robots' IP addresses in `ROBOT_IPS`, and set `RESULTS_FOLDER` to where trial logs should be saved.
5. Install Python dependencies (`pip install opencv-python numpy`) and run the laptop program for the system under test.

Laptop controls: `s` starts or pauses the system, `r` starts a recorded trial with manual mess, `m` starts one with the scheduled mess events, `n` starts one with random mess events, `e` ends a trial, `v` toggles video recording of heatmap, `1`/`2`/`3` choose the red/green/blue pen, left-click paints mess, right-click erases, and `q` quits.

## Limitations

- The tracking system is centralised. If it fails, the whole system fails, even though each robot makes its decisions independently.
- The scalability of the theory-of-mind step has only been tested with three robots.
- Movement is stop-start, one segment at a time, which limits speed and smoothness.

## Credits

- Overhead tracking uses the SwarmTracker system by Dr Paul O'Dowd ([ProcessingTrackingVisualiser](https://github.com/paulodowd/ProcessingTrackingVisualiser)). He also designed the PCB that mounts the M5Stack on the 3pi+.
- Supervisor: Dr Chanelle Lee.

## Author

Ted Sandland, MSc Robotics, University of Bristol & UWE Bristol.
