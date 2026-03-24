"""
LQIM — Lightweight Quantum-Inspired Metaheuristic
Core scheduling algorithm for heterogeneous edge environments.

Department: AI & Data Science
College   : Karpagam Institute of Technology
Team      : Dharanya AP, GandhimathiNathan T, Nareshkanna S, Sobika M
Guide     : Mr. Vignesh M
"""

import numpy as np
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── Fixed random seed for reproducibility ──────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ══════════════════════════════════════════════════════════════
#  DATA MODELS
# ══════════════════════════════════════════════════════════════

@dataclass
class EdgeNode:
    """Represents one edge computing node."""
    id: int
    cpu_ghz: float          # CPU frequency in GHz
    ram_gb: float           # Available RAM in GB
    current_load: float     # Current utilisation 0–100 %
    energy_budget: float    # Remaining energy budget in Joules

    def available_cpu(self) -> float:
        """Effective CPU available after current load."""
        return self.cpu_ghz * (1.0 - self.current_load / 100.0)


@dataclass
class Task:
    """Represents one incoming compute task."""
    id: int
    cpu_demand: float           # GHz required
    memory_mb: float            # MB required
    exec_time_ms: float         # Estimated execution time ms
    latency_sensitivity: float  # Priority weight 0.0–1.0
    task_type: str = "IoT"      # IoT / DB / Camera / Health / Vehicle


@dataclass
class ScheduleResult:
    """Output of one LQIM scheduling decision."""
    task_id: int
    assigned_node: int
    latency_ms: float
    energy_j: float
    utilization: float
    fitness: float
    iterations: int
    solve_time_ms: float
    method: str = "LQIM"


# ══════════════════════════════════════════════════════════════
#  Q-BIT
# ══════════════════════════════════════════════════════════════

class QBit:
    """
    A single quantum bit with amplitude coefficients alpha and beta.
    |ψ⟩ = alpha|0⟩ + beta|1⟩,  alpha² + beta² = 1
    """

    def __init__(self):
        v = 1.0 / np.sqrt(2)
        self.alpha = v   # amplitude of |0⟩
        self.beta  = v   # amplitude of |1⟩

    def measure(self) -> int:
        """Collapse to 0 or 1 by probability β²."""
        return 1 if random.random() < self.beta ** 2 else 0

    def rotate(self, delta_theta: float):
        """Apply quantum rotation gate by angle delta_theta."""
        cos_t = np.cos(delta_theta)
        sin_t = np.sin(delta_theta)
        new_a = cos_t * self.alpha - sin_t * self.beta
        new_b = sin_t * self.alpha + cos_t * self.beta
        # Normalise to avoid floating-point drift
        norm = np.sqrt(new_a ** 2 + new_b ** 2)
        self.alpha = new_a / norm
        self.beta  = new_b / norm

    @property
    def prob_one(self) -> float:
        """Probability of collapsing to |1⟩."""
        return self.beta ** 2


# ══════════════════════════════════════════════════════════════
#  FITNESS FUNCTION
# ══════════════════════════════════════════════════════════════

def compute_fitness(
    node: EdgeNode,
    task: Task,
    live_load: float,
) -> Tuple[float, float, float, float]:
    """
    Composite fitness:  F = 0.4·Lat + 0.35·Energy + 0.25·(1 − Util)
    Returns (score, latency_ms, energy_j, utilization_fraction)
    """
    avail = max(node.cpu_ghz * (1.0 - live_load / 100.0) / task.cpu_demand, 0.1)
    latency  = task.exec_time_ms * (1.0 + live_load / 100.0) / avail
    energy   = task.cpu_demand * 2.5 * (latency / 1000.0)
    util     = min(live_load / 100.0 + task.cpu_demand / node.cpu_ghz, 1.0)

    lat_norm = latency / 200.0
    eng_norm = energy  / 5.0
    score    = 0.4 * lat_norm + 0.35 * eng_norm + 0.25 * (1.0 - util)

    return score, latency, energy, util


# ══════════════════════════════════════════════════════════════
#  LQIM SCHEDULER
# ══════════════════════════════════════════════════════════════

class LQIMScheduler:
    """
    Lightweight Quantum-Inspired Metaheuristic scheduler.

    Algorithm phases:
      1. Initialise Q-bit population
      2. Measure → candidate schedules
      3. Compute fitness F for all candidates
      4. Update global best
      5. Rotation gate update Δθ
      6. Convergence check (β² > 0.92)
      7. Execute schedule
    """

    ROT           = 0.05 * np.pi   # Rotation gate step size
    CONV_THRESH   = 0.92           # β² convergence threshold
    MIN_ITER      = 8              # Minimum iterations before convergence check
    POP_SIZE      = 30             # Q-bit population size
    MAX_ITER      = 80             # Hard iteration limit

    def __init__(self, nodes: List[EdgeNode]):
        self.nodes      = nodes
        self.live_loads = [n.current_load for n in nodes]

    def schedule(self, task: Task) -> ScheduleResult:
        """Run LQIM and return the best node assignment."""
        start = time.time()
        N = len(self.nodes)

        # Phase 1 — Initialise Q-bit population
        population = [[QBit() for _ in range(N)] for _ in range(self.POP_SIZE)]

        best_fit   = float('inf')
        best_node  = 0
        best_lat   = 0.0
        best_eng   = 0.0
        best_util  = 0.0
        iters      = 0

        for it in range(self.MAX_ITER):
            iters = it + 1

            # Phase 2 — Measure candidates
            for individual in population:
                probs = [q.prob_one for q in individual]
                total = sum(probs)
                r, chosen = random.random() * total, 0
                for n_idx, p in enumerate(probs):
                    r -= p
                    if r <= 0:
                        chosen = n_idx
                        break

                # Phase 3 — Evaluate fitness
                score, lat, eng, util = compute_fitness(
                    self.nodes[chosen], task, self.live_loads[chosen]
                )

                # Phase 4 — Update global best
                if score < best_fit:
                    best_fit, best_node = score, chosen
                    best_lat, best_eng, best_util = lat, eng, util

            # Phase 5 — Rotation gate update
            for individual in population:
                for n_idx, qbit in enumerate(individual):
                    cur  = 1 if qbit.prob_one > 0.5 else 0
                    best = 1 if n_idx == best_node else 0
                    if cur == 0 and best == 1:
                        qbit.rotate(self.ROT)
                    elif cur == 1 and best == 0:
                        qbit.rotate(-self.ROT)

            # Phase 6 — Convergence check
            if it >= self.MIN_ITER:
                avg_beta_sq = sum(
                    ind[best_node].prob_one for ind in population
                ) / self.POP_SIZE
                if avg_beta_sq > self.CONV_THRESH:
                    break

        # Phase 7 — Execute schedule
        self.live_loads[best_node] = min(
            self.live_loads[best_node] + random.uniform(1, 5), 95.0
        )

        solve_ms = (time.time() - start) * 1000.0

        return ScheduleResult(
            task_id       = task.id,
            assigned_node = best_node,
            latency_ms    = round(best_lat, 2),
            energy_j      = round(best_eng, 4),
            utilization   = round(best_util * 100, 2),
            fitness       = round(best_fit, 4),
            iterations    = iters,
            solve_time_ms = round(solve_ms, 2),
            method        = "LQIM",
        )


# ══════════════════════════════════════════════════════════════
#  BASELINE: GENETIC ALGORITHM
# ══════════════════════════════════════════════════════════════

class GAScheduler:
    """Genetic Algorithm baseline for benchmarking."""

    POP_SIZE  = 30
    MAX_ITER  = 80
    MUT_RATE  = 0.1

    def __init__(self, nodes: List[EdgeNode]):
        self.nodes      = nodes
        self.live_loads = [n.current_load for n in nodes]

    def schedule(self, task: Task) -> ScheduleResult:
        start = time.time()
        N = len(self.nodes)

        # Initialise population as random node indices
        pop = [random.randint(0, N - 1) for _ in range(self.POP_SIZE)]
        best_fit, best_node = float('inf'), 0
        best_lat = best_eng = best_util = 0.0

        for _ in range(self.MAX_ITER):
            # Evaluate
            fits = []
            for idx in pop:
                s, lat, eng, util = compute_fitness(
                    self.nodes[idx], task, self.live_loads[idx]
                )
                fits.append(s)
                if s < best_fit:
                    best_fit, best_node = s, idx
                    best_lat, best_eng, best_util = lat, eng, util

            # Selection (tournament)
            new_pop = []
            for _ in range(self.POP_SIZE):
                a, b = random.sample(range(self.POP_SIZE), 2)
                new_pop.append(pop[a] if fits[a] < fits[b] else pop[b])

            # Crossover + mutation
            for i in range(0, self.POP_SIZE - 1, 2):
                if random.random() < 0.8:
                    new_pop[i], new_pop[i + 1] = new_pop[i + 1], new_pop[i]
            pop = [
                random.randint(0, N - 1) if random.random() < self.MUT_RATE else g
                for g in new_pop
            ]

        solve_ms = (time.time() - start) * 1000.0
        return ScheduleResult(
            task_id=task.id, assigned_node=best_node,
            latency_ms=round(best_lat, 2), energy_j=round(best_eng, 4),
            utilization=round(best_util * 100, 2), fitness=round(best_fit, 4),
            iterations=self.MAX_ITER, solve_time_ms=round(solve_ms, 2),
            method="GA",
        )


# ══════════════════════════════════════════════════════════════
#  BASELINE: PARTICLE SWARM OPTIMIZATION
# ══════════════════════════════════════════════════════════════

class PSOScheduler:
    """Particle Swarm Optimization baseline for benchmarking."""

    N_PARTICLES = 30
    MAX_ITER    = 80
    W, C1, C2  = 0.7, 1.5, 1.5

    def __init__(self, nodes: List[EdgeNode]):
        self.nodes      = nodes
        self.live_loads = [n.current_load for n in nodes]

    def schedule(self, task: Task) -> ScheduleResult:
        start = time.time()
        N = len(self.nodes)

        pos = [random.uniform(0, N - 1) for _ in range(self.N_PARTICLES)]
        vel = [random.uniform(-1, 1)    for _ in range(self.N_PARTICLES)]
        pbest = pos[:]
        pbest_fit = [float('inf')] * self.N_PARTICLES
        gbest, gbest_fit = 0.0, float('inf')
        best_lat = best_eng = best_util = 0.0

        for _ in range(self.MAX_ITER):
            for i in range(self.N_PARTICLES):
                idx = int(np.clip(round(pos[i]), 0, N - 1))
                s, lat, eng, util = compute_fitness(
                    self.nodes[idx], task, self.live_loads[idx]
                )
                if s < pbest_fit[i]:
                    pbest_fit[i], pbest[i] = s, pos[i]
                if s < gbest_fit:
                    gbest_fit, gbest = s, pos[i]
                    best_lat, best_eng, best_util = lat, eng, util
            for i in range(self.N_PARTICLES):
                r1, r2 = random.random(), random.random()
                vel[i] = (self.W * vel[i]
                          + self.C1 * r1 * (pbest[i] - pos[i])
                          + self.C2 * r2 * (gbest    - pos[i]))
                pos[i] = np.clip(pos[i] + vel[i], 0, N - 1)

        best_node = int(np.clip(round(gbest), 0, N - 1))
        solve_ms  = (time.time() - start) * 1000.0
        return ScheduleResult(
            task_id=task.id, assigned_node=best_node,
            latency_ms=round(best_lat, 2), energy_j=round(best_eng, 4),
            utilization=round(best_util * 100, 2), fitness=round(gbest_fit, 4),
            iterations=self.MAX_ITER, solve_time_ms=round(solve_ms, 2),
            method="PSO",
        )


# ══════════════════════════════════════════════════════════════
#  SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════

def build_nodes() -> List[EdgeNode]:
    """Return the 10 heterogeneous edge nodes used in the simulation."""
    return [
        EdgeNode(0, 1.80, 1, 34,  192.6),
        EdgeNode(1, 2.16, 1, 11,   73.4),
        EdgeNode(2, 1.29, 4, 29,  179.9),
        EdgeNode(3, 2.33, 2, 44,   53.1),
        EdgeNode(4, 2.53, 2, 12,   81.9),
        EdgeNode(5, 1.49, 2, 26,   95.6),
        EdgeNode(6, 1.89, 1, 29,   93.7),
        EdgeNode(7, 1.42, 1, 20,   93.8),
        EdgeNode(8, 1.93, 8, 13,  167.8),
        EdgeNode(9, 2.02, 1,  7,  138.9),
    ]


def generate_tasks(n: int = 50) -> List[Task]:
    """Generate n tasks with Poisson-like random parameters."""
    types = ["IoT", "DB", "Camera", "Health", "Vehicle"]
    return [
        Task(
            id                  = i + 1,
            cpu_demand          = round(random.uniform(0.2, 2.0), 2),
            memory_mb           = round(random.uniform(50, 600), 1),
            exec_time_ms        = round(random.uniform(10, 200), 1),
            latency_sensitivity = round(random.uniform(0.1, 1.0), 2),
            task_type           = random.choice(types),
        )
        for i in range(n)
    ]


def run_simulation(n_tasks: int = 50):
    """Run full simulation and print comparison table."""
    nodes  = build_nodes()
    tasks  = generate_tasks(n_tasks)

    lqim = LQIMScheduler([EdgeNode(*vars(n).values()) for n in build_nodes()])
    ga   = GAScheduler  ([EdgeNode(*vars(n).values()) for n in build_nodes()])
    pso  = PSOScheduler ([EdgeNode(*vars(n).values()) for n in build_nodes()])

    results = {"LQIM": [], "GA": [], "PSO": []}

    print(f"\n{'='*65}")
    print(f"  LQIM Edge Scheduler — Simulation ({n_tasks} tasks, 10 nodes)")
    print(f"{'='*65}")
    print(f"{'Task':>4}  {'LQIM Lat':>9}  {'GA Lat':>8}  {'PSO Lat':>8}  {'LQIM Iters':>11}")
    print(f"{'-'*65}")

    for task in tasks:
        r_lqim = lqim.schedule(task)
        r_ga   = ga.schedule(task)
        r_pso  = pso.schedule(task)
        results["LQIM"].append(r_lqim)
        results["GA"].append(r_ga)
        results["PSO"].append(r_pso)
        print(
            f"T-{task.id:>3}  "
            f"{r_lqim.latency_ms:>7.1f}ms  "
            f"{r_ga.latency_ms:>6.1f}ms  "
            f"{r_pso.latency_ms:>6.1f}ms  "
            f"{r_lqim.iterations:>8} iters"
        )

    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    for method, res in results.items():
        avg_lat  = sum(r.latency_ms    for r in res) / len(res)
        avg_eng  = sum(r.energy_j      for r in res) / len(res)
        avg_util = sum(r.utilization   for r in res) / len(res)
        avg_iter = sum(r.iterations    for r in res) / len(res)
        print(
            f"  {method:<6}  "
            f"Latency: {avg_lat:6.1f}ms  "
            f"Energy: {avg_eng:.4f}J  "
            f"Util: {avg_util:5.1f}%  "
            f"Iters: {avg_iter:5.1f}"
        )
    print(f"{'='*65}\n")
    return results


if __name__ == "__main__":
    run_simulation(50)
