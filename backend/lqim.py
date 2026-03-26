"""
LQIM — Lightweight Quantum-Inspired Metaheuristic
Core scheduling algorithm for heterogeneous edge environments.

Department: AI & Data Science
College   : Karpagam Institute of Technology
Team      : Dharanya AP, GandhimathiNathan T, Nareshkanna S, Sobika M
Guide     : Mr. Vignesh M

Key Finding:
  LQIM achieves comparable scheduling quality to GA and PSO while
  converging in ~40-50% fewer iterations. In real-time edge environments,
  this translates to faster scheduling decisions — critical when the
  scheduling delay itself contributes to overall task latency.
"""

import numpy as np
import random
import time
import copy
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict


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
        return self.cpu_ghz * (1.0 - self.current_load / 100.0)


@dataclass
class Task:
    """Represents one incoming compute task."""
    id: int
    cpu_demand: float           # GHz required
    memory_mb: float            # MB required
    exec_time_ms: float         # Estimated execution time ms
    latency_sensitivity: float  # Priority weight 0.0–1.0
    task_type: str = "IoT"


@dataclass
class ScheduleResult:
    """Output of one scheduling decision."""
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
    Quantum bit: |ψ⟩ = α|0⟩ + β|1⟩, where α² + β² = 1.
    Initialized in equal superposition (α = β = 1/√2).
    """

    def __init__(self):
        v = 1.0 / np.sqrt(2)
        self.alpha = v
        self.beta  = v

    def measure(self) -> int:
        return 1 if random.random() < self.beta ** 2 else 0

    def rotate(self, delta_theta: float):
        cos_t = np.cos(delta_theta)
        sin_t = np.sin(delta_theta)
        new_a = cos_t * self.alpha - sin_t * self.beta
        new_b = sin_t * self.alpha + cos_t * self.beta
        norm = np.sqrt(new_a ** 2 + new_b ** 2)
        if norm > 1e-10:
            self.alpha = new_a / norm
            self.beta  = new_b / norm

    @property
    def prob_one(self) -> float:
        return self.beta ** 2


# ══════════════════════════════════════════════════════════════
#  FITNESS FUNCTION
# ══════════════════════════════════════════════════════════════

def compute_fitness(
    node: EdgeNode,
    task: Task,
    live_load: float,
    w1: float = 0.4,
    w2: float = 0.35,
    w3: float = 0.25,
) -> Tuple[float, float, float, float]:
    """
    Multi-objective fitness: F = w1·Lat + w2·Energy + w3·(1 − Util)

    Models:
    - Latency: execution time scaled by CPU availability and load
    - Energy: power draw proportional to CPU demand × time
    - Utilization: capacity usage (higher = better resource efficiency)
    - Memory pressure: penalty when RAM demand exceeds 80% of node capacity
    """
    load_frac = min(live_load / 100.0, 0.99)
    avail_cpu = max(node.cpu_ghz * (1.0 - load_frac), 0.01)
    cpu_ratio = max(avail_cpu / task.cpu_demand, 0.1)

    latency = task.exec_time_ms / cpu_ratio * (1.0 + load_frac * 0.5)
    latency *= (1.0 + task.latency_sensitivity * 0.3)

    mem_ratio = task.memory_mb / (node.ram_gb * 1024)
    if mem_ratio > 0.8:
        latency *= (1.0 + (mem_ratio - 0.8) * 2.0)

    energy = task.cpu_demand * 2.5 * (latency / 1000.0)
    util = min(load_frac + task.cpu_demand / node.cpu_ghz, 1.0)

    score = w1 * (latency / 300.0) + w2 * (energy / 5.0) + w3 * (1.0 - util)
    return score, latency, energy, util


# ══════════════════════════════════════════════════════════════
#  LQIM SCHEDULER
# ══════════════════════════════════════════════════════════════

class LQIMScheduler:
    """
    Lightweight Quantum-Inspired Metaheuristic scheduler.

    Phases: Init Q-bits → Measure → Fitness → Best → Rotate → Converge → Execute

    Key properties:
    - Adaptive rotation gate (step size decays with iterations)
    - Early convergence detection (β² threshold)
    - Probabilistic exploration via Q-bit superposition
    """

    ROT         = 0.04 * np.pi
    CONV_THRESH = 0.92
    MIN_ITER    = 12
    POP_SIZE    = 30
    MAX_ITER    = 80

    def __init__(self, nodes: List[EdgeNode]):
        self.nodes = nodes
        self.live_loads = [n.current_load for n in nodes]

    def schedule(self, task: Task) -> ScheduleResult:
        start = time.time()
        N = len(self.nodes)

        population = [[QBit() for _ in range(N)] for _ in range(self.POP_SIZE)]
        best_fit, best_node = float('inf'), 0
        best_lat = best_eng = best_util = 0.0
        iters = 0

        for it in range(self.MAX_ITER):
            iters = it + 1

            for individual in population:
                probs = [q.prob_one for q in individual]
                total = sum(probs)
                if total < 1e-10:
                    chosen = random.randint(0, N - 1)
                else:
                    r = random.random() * total
                    chosen = N - 1
                    for n_idx, p in enumerate(probs):
                        r -= p
                        if r <= 0:
                            chosen = n_idx
                            break

                score, lat, eng, util = compute_fitness(
                    self.nodes[chosen], task, self.live_loads[chosen]
                )
                if score < best_fit:
                    best_fit, best_node = score, chosen
                    best_lat, best_eng, best_util = lat, eng, util

            # Adaptive rotation: step decays as population converges
            rot_step = self.ROT * max(0.4, 1.0 - it / self.MAX_ITER)
            for individual in population:
                for n_idx, qbit in enumerate(individual):
                    if n_idx == best_node:
                        qbit.rotate(rot_step)
                    else:
                        qbit.rotate(-rot_step * 0.4)

            # Convergence check
            if it >= self.MIN_ITER:
                avg_beta_sq = sum(
                    ind[best_node].prob_one for ind in population
                ) / self.POP_SIZE
                if avg_beta_sq > self.CONV_THRESH:
                    break

        self.live_loads[best_node] = min(
            self.live_loads[best_node] + random.uniform(2, 6), 95.0
        )

        return ScheduleResult(
            task_id=task.id, assigned_node=best_node,
            latency_ms=round(best_lat, 2), energy_j=round(best_eng, 4),
            utilization=round(best_util * 100, 2), fitness=round(best_fit, 4),
            iterations=iters,
            solve_time_ms=round((time.time() - start) * 1000, 2),
            method="LQIM",
        )


# ══════════════════════════════════════════════════════════════
#  BASELINE: GENETIC ALGORITHM
# ══════════════════════════════════════════════════════════════

class GAScheduler:
    """
    Standard GA — tournament selection, crossover, mutation.
    Runs all 80 iterations (no early stopping).
    """

    POP_SIZE = 30
    MAX_ITER = 80
    MUT_RATE = 0.18

    def __init__(self, nodes: List[EdgeNode]):
        self.nodes = nodes
        self.live_loads = [n.current_load for n in nodes]

    def schedule(self, task: Task) -> ScheduleResult:
        start = time.time()
        N = len(self.nodes)

        pop = [random.randint(0, N - 1) for _ in range(self.POP_SIZE)]
        best_fit, best_node = float('inf'), 0
        best_lat = best_eng = best_util = 0.0

        for gen in range(self.MAX_ITER):
            fits = []
            for idx in pop:
                s, lat, eng, util = compute_fitness(
                    self.nodes[idx], task, self.live_loads[idx]
                )
                fits.append(s)
                if s < best_fit:
                    best_fit, best_node = s, idx
                    best_lat, best_eng, best_util = lat, eng, util

            new_pop = []
            for _ in range(self.POP_SIZE):
                a, b = random.sample(range(self.POP_SIZE), 2)
                new_pop.append(pop[a] if fits[a] < fits[b] else pop[b])

            for i in range(0, self.POP_SIZE - 1, 2):
                if random.random() < 0.8:
                    new_pop[i], new_pop[i + 1] = new_pop[i + 1], new_pop[i]

            pop = [
                random.randint(0, N - 1) if random.random() < self.MUT_RATE else g
                for g in new_pop
            ]

        self.live_loads[best_node] = min(
            self.live_loads[best_node] + random.uniform(2, 6), 95.0
        )
        return ScheduleResult(
            task_id=task.id, assigned_node=best_node,
            latency_ms=round(best_lat, 2), energy_j=round(best_eng, 4),
            utilization=round(best_util * 100, 2), fitness=round(best_fit, 4),
            iterations=self.MAX_ITER,
            solve_time_ms=round((time.time() - start) * 1000, 2),
            method="GA",
        )


# ══════════════════════════════════════════════════════════════
#  BASELINE: PARTICLE SWARM OPTIMIZATION
# ══════════════════════════════════════════════════════════════

class PSOScheduler:
    """
    Continuous PSO mapped to discrete nodes. Runs all 80 iterations.
    """

    N_PARTICLES = 30
    MAX_ITER    = 80
    W, C1, C2   = 0.7, 1.5, 1.5

    def __init__(self, nodes: List[EdgeNode]):
        self.nodes = nodes
        self.live_loads = [n.current_load for n in nodes]

    def schedule(self, task: Task) -> ScheduleResult:
        start = time.time()
        N = len(self.nodes)

        pos = [random.uniform(0, N - 1) for _ in range(self.N_PARTICLES)]
        vel = [random.uniform(-2, 2) for _ in range(self.N_PARTICLES)]
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
                          + self.C2 * r2 * (gbest - pos[i]))
                vel[i] = np.clip(vel[i], -3, 3)
                pos[i] = np.clip(pos[i] + vel[i], 0, N - 1)

        best_node = int(np.clip(round(gbest), 0, N - 1))
        self.live_loads[best_node] = min(
            self.live_loads[best_node] + random.uniform(2, 6), 95.0
        )
        return ScheduleResult(
            task_id=task.id, assigned_node=best_node,
            latency_ms=round(best_lat, 2), energy_j=round(best_eng, 4),
            utilization=round(best_util * 100, 2), fitness=round(gbest_fit, 4),
            iterations=self.MAX_ITER,
            solve_time_ms=round((time.time() - start) * 1000, 2),
            method="PSO",
        )


# ══════════════════════════════════════════════════════════════
#  NODE AND TASK GENERATORS
# ══════════════════════════════════════════════════════════════

def build_nodes() -> List[EdgeNode]:
    """10 heterogeneous edge nodes (1.2–2.5 GHz, 1–8 GB RAM)."""
    return [
        EdgeNode(0, 1.80, 2, 34, 192.6),
        EdgeNode(1, 2.16, 4, 11,  73.4),
        EdgeNode(2, 1.29, 4, 29, 179.9),
        EdgeNode(3, 2.33, 2, 44,  53.1),
        EdgeNode(4, 2.53, 4, 12,  81.9),
        EdgeNode(5, 1.49, 2, 26,  95.6),
        EdgeNode(6, 1.89, 2, 29,  93.7),
        EdgeNode(7, 1.42, 1, 20,  93.8),
        EdgeNode(8, 1.93, 8, 13, 167.8),
        EdgeNode(9, 2.02, 2,  7, 138.9),
    ]


def generate_tasks(n: int = 50) -> List[Task]:
    """Generate tasks with type-specific parameter distributions."""
    types = ["IoT", "DB", "Camera", "Health", "Vehicle"]
    tasks = []
    for i in range(n):
        t_type = random.choice(types)
        if t_type == "IoT":
            cpu, mem = round(random.uniform(0.2, 0.8), 2), round(random.uniform(50, 200), 1)
            exec_t, sens = round(random.uniform(10, 60), 1), round(random.uniform(0.6, 1.0), 2)
        elif t_type == "DB":
            cpu, mem = round(random.uniform(0.5, 1.5), 2), round(random.uniform(200, 600), 1)
            exec_t, sens = round(random.uniform(30, 150), 1), round(random.uniform(0.3, 0.7), 2)
        elif t_type == "Camera":
            cpu, mem = round(random.uniform(0.8, 2.0), 2), round(random.uniform(300, 600), 1)
            exec_t, sens = round(random.uniform(20, 80), 1), round(random.uniform(0.7, 1.0), 2)
        elif t_type == "Health":
            cpu, mem = round(random.uniform(0.3, 1.0), 2), round(random.uniform(100, 400), 1)
            exec_t, sens = round(random.uniform(15, 50), 1), round(random.uniform(0.8, 1.0), 2)
        else:
            cpu, mem = round(random.uniform(1.0, 2.0), 2), round(random.uniform(400, 600), 1)
            exec_t, sens = round(random.uniform(10, 40), 1), round(random.uniform(0.9, 1.0), 2)
        tasks.append(Task(i + 1, cpu, mem, exec_t, sens, t_type))
    return tasks


# ══════════════════════════════════════════════════════════════
#  SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════

def run_simulation(n_tasks: int = 50, seed: int = 42) -> Dict:
    """Run comparative simulation with independent scheduler states."""
    random.seed(seed)
    np.random.seed(seed)
    tasks = generate_tasks(n_tasks)

    lqim = LQIMScheduler(copy.deepcopy(build_nodes()))
    ga   = GAScheduler(copy.deepcopy(build_nodes()))
    pso  = PSOScheduler(copy.deepcopy(build_nodes()))

    results = {"LQIM": [], "GA": [], "PSO": []}

    print(f"\n{'='*72}")
    print(f"  LQIM Edge Scheduler — Simulation ({n_tasks} tasks, 10 nodes)")
    print(f"{'='*72}")
    print(f"{'Task':>4}  {'Type':<8} {'LQIM':>9}  {'GA':>8}  {'PSO':>8}  {'Iters':>6}")
    print(f"{'-'*72}")

    for task in tasks:
        r_lqim = lqim.schedule(task)
        r_ga   = ga.schedule(task)
        r_pso  = pso.schedule(task)
        results["LQIM"].append(r_lqim)
        results["GA"].append(r_ga)
        results["PSO"].append(r_pso)
        print(
            f"T-{task.id:>3}  {task.task_type:<8}"
            f"{r_lqim.latency_ms:>7.1f}ms  "
            f"{r_ga.latency_ms:>6.1f}ms  "
            f"{r_pso.latency_ms:>6.1f}ms  "
            f"{r_lqim.iterations:>5}"
        )

    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    for method, res in results.items():
        avg_lat  = sum(r.latency_ms  for r in res) / len(res)
        avg_eng  = sum(r.energy_j    for r in res) / len(res)
        avg_util = sum(r.utilization for r in res) / len(res)
        avg_iter = sum(r.iterations  for r in res) / len(res)
        avg_time = sum(r.solve_time_ms for r in res) / len(res)
        print(
            f"  {method:<6}  "
            f"Lat: {avg_lat:5.1f}ms  "
            f"Eng: {avg_eng:.4f}J  "
            f"Util: {avg_util:4.1f}%  "
            f"Iters: {avg_iter:4.0f}  "
            f"Solve: {avg_time:.1f}ms"
        )

    lq = results["LQIM"]
    ga_r = results["GA"]
    ps = results["PSO"]
    avg = lambda r, f: sum(getattr(x, f) for x in r) / len(r)

    print(f"\n  Iteration savings: {(1 - avg(lq,'iterations')/80)*100:.0f}% fewer iterations than GA/PSO")
    lat_diff = (1 - avg(lq,'latency_ms')/avg(ga_r,'latency_ms'))*100
    eng_diff = (1 - avg(lq,'energy_j')/avg(ga_r,'energy_j'))*100
    print(f"  vs GA:  Latency {'↓' if lat_diff > 0 else '↑'}{abs(lat_diff):.1f}%  Energy {'↓' if eng_diff > 0 else '↑'}{abs(eng_diff):.1f}%")
    print(f"{'='*72}\n")
    return results


if __name__ == "__main__":
    run_simulation(50, seed=42)
