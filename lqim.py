import numpy as np
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional  


# ──────────────────────────────────────────────
# DATA MODELS
# ──────────────────────────────────────────────

@dataclass
class EdgeNode:
    """Represents a heterogeneous edge compute node."""
    node_id: int
    cpu_ghz: float          # CPU frequency (GHz)
    ram_gb: float           # RAM (GB)
    energy_budget: float    # Energy budget (Joules)
    current_load: float = 0.0   # 0.0 – 1.0
    tasks_assigned: int = 0

    @property
    def available_cpu(self):
        return self.cpu_ghz * (1.0 - self.current_load)

    @property
    def utilization(self):
        return self.current_load


@dataclass
class Task:
    """Represents an IoT/edge task to be scheduled."""
    task_id: int
    cpu_demand: float       # GHz required
    memory_mb: float        # MB required
    exec_time_ms: float     # Expected execution time (ms)
    latency_sensitivity: float  # 0–1 (1 = highly latency-sensitive)
    arrival_time: float = 0.0


@dataclass
class ScheduleResult:
    """Result of one scheduling decision."""
    task_id: int
    node_id: int
    latency_ms: float
    energy_j: float
    utilization: float
    fitness: float
    method: str
    iterations: int
    solve_time_ms: float


# ──────────────────────────────────────────────
# Q-BIT REPRESENTATION
# ──────────────────────────────────────────────

class QBit:
    """
    Quantum Bit (Q-bit) encoding:
        Qi = [αi, βi]  where |αi|² + |βi|² = 1
    α² → probability of state |0⟩ (task NOT assigned)
    β² → probability of state |1⟩ (task IS assigned)
    """

    def __init__(self, alpha: float = None, beta: float = None):
        if alpha is None:
            # Initialise in equal superposition
            angle = np.random.uniform(0, np.pi / 2)
            self.alpha = np.cos(angle)
            self.beta = np.sin(angle)
        else:
            self.alpha = alpha
            self.beta = beta
        self._normalise()

    def _normalise(self):
        norm = np.sqrt(self.alpha**2 + self.beta**2)
        self.alpha /= norm
        self.beta /= norm

    def measure(self) -> int:
        """Collapse Q-bit: returns 1 (assigned) with probability β²."""
        return 1 if np.random.random() < self.beta**2 else 0

    def rotate(self, delta_theta: float):
        """
        Rotation gate update:
            [α'] = [cos θ   sin θ] [α]
            [β']   [-sin θ  cos θ] [β]
        """
        cos_t = np.cos(delta_theta)
        sin_t = np.sin(delta_theta)
        new_alpha = cos_t * self.alpha + sin_t * self.beta
        new_beta = -sin_t * self.alpha + cos_t * self.beta
        self.alpha, self.beta = new_alpha, new_beta
        self._normalise()

    @property
    def prob_assigned(self):
        return self.beta**2


# ──────────────────────────────────────────────
# LQIM SCHEDULER
# ──────────────────────────────────────────────

class LQIM:
    """
    Lightweight Quantum-Inspired Metaheuristic Scheduler.

    Procedure:
      1. Initialise Q-bit population
      2. Measure → candidate schedules
      3. Compute multi-objective fitness
      4. Update global best
      5. Apply rotation gate updates
      6. Repeat until convergence or time limit
      7. Execute scheduling decision
    """

    def __init__(
        self,
        pop_size: int = 30,
        max_iterations: int = 100,
        rotation_step: float = 0.05 * np.pi,
        w1: float = 0.4,   # latency weight
        w2: float = 0.35,  # energy weight
        w3: float = 0.25,  # utilisation weight
        time_limit_ms: float = 50.0,
    ):
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.rotation_step = rotation_step
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.time_limit_ms = time_limit_ms

        self.history: List[Dict] = []

    # ── Fitness ────────────────────────────────

    def _compute_latency(self, task: Task, node: EdgeNode) -> float:
        """Latency = exec_time scaled by node speed and current load."""
        speed_factor = max(node.available_cpu / task.cpu_demand, 0.1)
        load_penalty = 1.0 + node.current_load
        return task.exec_time_ms * load_penalty / speed_factor

    def _compute_energy(self, task: Task, node: EdgeNode, latency_ms: float) -> float:
        """Energy ∝ cpu_demand × latency (simplified power model)."""
        base_power = task.cpu_demand * 2.5   # Watts per GHz
        return base_power * (latency_ms / 1000.0)

    def _fitness(
        self,
        task: Task,
        node: EdgeNode,
        max_latency: float = 200.0,
        max_energy: float = 5.0,
    ) -> Tuple[float, float, float, float]:
        latency = self._compute_latency(task, node)
        energy = self._compute_energy(task, node, latency)
        util = node.current_load + (task.cpu_demand / node.cpu_ghz)
        util = min(util, 1.0)

        # Normalise each metric to [0, 1]
        norm_lat = latency / max_latency
        norm_eng = energy / max_energy
        norm_util = 1.0 - util   # lower util → worse (we want high util)

        score = self.w1 * norm_lat + self.w2 * norm_eng + self.w3 * norm_util
        return score, latency, energy, util

    # ── Core LQIM Loop ─────────────────────────

    def schedule(
        self,
        task: Task,
        nodes: List[EdgeNode],
    ) -> ScheduleResult:
        """Run LQIM to find the best node assignment for a task."""
        n_nodes = len(nodes)
        start_time = time.perf_counter()

        # Step 1: Initialise Q-bit population
        # Each individual = n_nodes Q-bits (one per node)
        population = [
            [QBit() for _ in range(n_nodes)]
            for _ in range(self.pop_size)
        ]

        best_fitness = float("inf")
        best_node_idx = 0
        best_latency = best_energy = best_util = 0.0
        iteration_count = 0

        for iteration in range(self.max_iterations):
            elapsed = (time.perf_counter() - start_time) * 1000
            if elapsed > self.time_limit_ms:
                break

            iteration_count += 1
            current_best_fitness = float("inf")
            current_best_idx = 0

            # Step 2 & 3: Measure candidates → compute fitness
            for individual in population:
                # Sample assignment: pick node with highest β²
                probs = np.array([q.prob_assigned for q in individual])
                probs_sum = probs.sum()
                if probs_sum > 0:
                    probs = probs / probs_sum
                else:
                    probs = np.ones(n_nodes) / n_nodes

                chosen_idx = np.random.choice(n_nodes, p=probs)
                node = nodes[chosen_idx]

                score, lat, eng, util = self._fitness(task, node)

                if score < current_best_fitness:
                    current_best_fitness = score
                    current_best_idx = chosen_idx
                    current_lat, current_eng, current_util = lat, eng, util

            # Step 4: Update global best
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_node_idx = current_best_idx
                best_latency = current_lat
                best_energy = current_eng
                best_util = current_util

            # Step 5: Rotation gate update
            for individual in population:
                for q_idx, qbit in enumerate(individual):
                    # Direction: rotate toward best node
                    current_state = 1 if qbit.prob_assigned > 0.5 else 0
                    best_state = 1 if q_idx == best_node_idx else 0

                    if current_state == 0 and best_state == 1:
                        # Increase β (rotate upward)
                        delta = self.rotation_step
                    elif current_state == 1 and best_state == 0:
                        # Increase α (rotate downward)
                        delta = -self.rotation_step
                    else:
                        delta = 0.0

                    if delta != 0.0:
                        qbit.rotate(delta)

            # Convergence check: if β² of best node > 0.95 across population
            avg_prob = np.mean([
                individual[best_node_idx].prob_assigned
                for individual in population
            ])
            if avg_prob > 0.95 and iteration > 10:
                break

        solve_time = (time.perf_counter() - start_time) * 1000

        return ScheduleResult(
            task_id=task.task_id,
            node_id=nodes[best_node_idx].node_id,
            latency_ms=round(best_latency, 2),
            energy_j=round(best_energy, 4),
            utilization=round(best_util, 4),
            fitness=round(best_fitness, 6),
            method="LQIM",
            iterations=iteration_count,
            solve_time_ms=round(solve_time, 3),
        )


# ──────────────────────────────────────────────
# BASELINE SCHEDULERS (GA & PSO)
# ──────────────────────────────────────────────

class GeneticAlgorithmScheduler:
    """Baseline GA scheduler for comparison."""

    def __init__(self, pop_size=30, generations=100, mutation_rate=0.1,
                 w1=0.4, w2=0.35, w3=0.25):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.w1, self.w2, self.w3 = w1, w2, w3

    def _fitness(self, task, node):
        speed_factor = max((node.cpu_ghz * (1.0 - node.current_load)) / task.cpu_demand, 0.1)
        load_penalty = 1.0 + node.current_load
        latency = task.exec_time_ms * load_penalty / speed_factor
        energy = task.cpu_demand * 2.5 * (latency / 1000.0)
        util = min(node.current_load + (task.cpu_demand / node.cpu_ghz), 1.0)
        norm_lat = latency / 200.0
        norm_eng = energy / 5.0
        norm_util = 1.0 - util
        score = self.w1 * norm_lat + self.w2 * norm_eng + self.w3 * norm_util
        return score, latency, energy, util

    def schedule(self, task, nodes):
        start = time.perf_counter()
        n = len(nodes)
        population = [random.randint(0, n - 1) for _ in range(self.pop_size)]

        best_fitness = float("inf")
        best_idx = 0
        best_lat = best_eng = best_util = 0.0

        for gen in range(self.generations):
            scored = []
            for idx in population:
                score, lat, eng, util = self._fitness(task, nodes[idx])
                scored.append((score, idx, lat, eng, util))
            scored.sort(key=lambda x: x[0])

            if scored[0][0] < best_fitness:
                best_fitness, best_idx, best_lat, best_eng, best_util = (
                    scored[0][0], scored[0][1], scored[0][2], scored[0][3], scored[0][4]
                )

            # Selection + crossover + mutation
            survivors = [s[1] for s in scored[:self.pop_size // 2]]
            children = []
            while len(children) < self.pop_size:
                p1, p2 = random.sample(survivors, 2)
                child = p1 if random.random() < 0.5 else p2
                if random.random() < self.mutation_rate:
                    child = random.randint(0, n - 1)
                children.append(child)
            population = children

        solve_time = (time.perf_counter() - start) * 1000
        return ScheduleResult(
            task_id=task.task_id,
            node_id=nodes[best_idx].node_id,
            latency_ms=round(best_lat, 2),
            energy_j=round(best_eng, 4),
            utilization=round(best_util, 4),
            fitness=round(best_fitness, 6),
            method="GA",
            iterations=self.generations,
            solve_time_ms=round(solve_time, 3),
        )


class PSOScheduler:
    """Baseline PSO scheduler for comparison."""

    def __init__(self, n_particles=30, iterations=100, w=0.5, c1=1.5, c2=1.5,
                 w1=0.4, w2=0.35, w3=0.25):
        self.n_particles = n_particles
        self.iterations = iterations
        self.w, self.c1, self.c2 = w, c1, c2
        self.w1, self.w2, self.w3 = w1, w2, w3

    def _fitness(self, task, node):
        speed_factor = max((node.cpu_ghz * (1.0 - node.current_load)) / task.cpu_demand, 0.1)
        load_penalty = 1.0 + node.current_load
        latency = task.exec_time_ms * load_penalty / speed_factor
        energy = task.cpu_demand * 2.5 * (latency / 1000.0)
        util = min(node.current_load + (task.cpu_demand / node.cpu_ghz), 1.0)
        norm_lat = latency / 200.0
        norm_eng = energy / 5.0
        norm_util = 1.0 - util
        score = self.w1 * norm_lat + self.w2 * norm_eng + self.w3 * norm_util
        return score, latency, energy, util

    def schedule(self, task, nodes):
        start = time.perf_counter()
        n = len(nodes)
        positions = np.random.uniform(0, n - 1, self.n_particles)
        velocities = np.zeros(self.n_particles)
        pbest = positions.copy()
        pbest_scores = [float("inf")] * self.n_particles
        gbest_pos = positions[0]
        gbest_score = float("inf")
        gbest_lat = gbest_eng = gbest_util = 0.0

        for _ in range(self.iterations):
            for i in range(self.n_particles):
                idx = int(np.clip(round(positions[i]), 0, n - 1))
                score, lat, eng, util = self._fitness(task, nodes[idx])
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest[i] = positions[i]
                if score < gbest_score:
                    gbest_score = score
                    gbest_pos = positions[i]
                    gbest_lat, gbest_eng, gbest_util = lat, eng, util

            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i]
                                 + self.c1 * r1 * (pbest[i] - positions[i])
                                 + self.c2 * r2 * (gbest_pos - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], 0, n - 1)

        solve_time = (time.perf_counter() - start) * 1000
        best_idx = int(np.clip(round(gbest_pos), 0, n - 1))
        return ScheduleResult(
            task_id=task.task_id,
            node_id=nodes[best_idx].node_id,
            latency_ms=round(gbest_lat, 2),
            energy_j=round(gbest_eng, 4),
            utilization=round(gbest_util, 4),
            fitness=round(gbest_score, 6),
            method="PSO",
            iterations=self.iterations,
            solve_time_ms=round(solve_time, 3),
        )


# ──────────────────────────────────────────────
# SIMULATION ENVIRONMENT
# ──────────────────────────────────────────────

class EdgeSimulator:
    """
    Simulates a heterogeneous edge environment with:
    - N heterogeneous nodes
    - Poisson task arrivals
    - Dynamic load changes
    """

    def __init__(self, n_nodes: int = 10, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.nodes = self._create_nodes(n_nodes)
        self.task_counter = 0

    def _create_nodes(self, n: int) -> List[EdgeNode]:
        nodes = []
        for i in range(n):
            nodes.append(EdgeNode(
                node_id=i,
                cpu_ghz=round(np.random.uniform(1.2, 2.8), 2),
                ram_gb=random.choice([1, 2, 4, 8]),
                energy_budget=round(np.random.uniform(50, 200), 1),
                current_load=round(np.random.uniform(0.05, 0.45), 2),
            ))
        return nodes

    def generate_task(self) -> Task:
        self.task_counter += 1
        return Task(
            task_id=self.task_counter,
            cpu_demand=round(np.random.uniform(0.2, 1.5), 2),
            memory_mb=round(np.random.uniform(50, 500), 0),
            exec_time_ms=round(np.random.uniform(20, 150), 1),
            latency_sensitivity=round(np.random.uniform(0.3, 1.0), 2),
            arrival_time=time.time(),
        )

    def update_loads(self):
        """Simulate dynamic load fluctuations."""
        for node in self.nodes:
            delta = np.random.uniform(-0.08, 0.08)
            node.current_load = float(np.clip(node.current_load + delta, 0.0, 0.95))

    def apply_result(self, result: ScheduleResult):
        """Update node state after a task is assigned."""
        node = next(n for n in self.nodes if n.node_id == result.node_id)
        node.current_load = min(node.current_load + 0.03, 0.95)
        node.tasks_assigned += 1


def run_simulation(n_tasks: int = 50, n_nodes: int = 10, seed: int = 42) -> Dict:
    """
    Run full comparative simulation: LQIM vs GA vs PSO.
    Returns aggregated metrics for all three methods.
    """
    results = {"LQIM": [], "GA": [], "PSO": []}

    lqim = LQIM(pop_size=30, max_iterations=80)
    ga = GeneticAlgorithmScheduler(pop_size=30, generations=80)
    pso = PSOScheduler(n_particles=30, iterations=80)

    for method_name, scheduler, sim_seed in [
        ("LQIM", lqim, seed),
        ("GA", ga, seed + 1),
        ("PSO", pso, seed + 2),
    ]:
        sim = EdgeSimulator(n_nodes=n_nodes, seed=sim_seed)
        for i in range(n_tasks):
            task = sim.generate_task()
            if i % 5 == 0:
                sim.update_loads()
            result = scheduler.schedule(task, sim.nodes)
            results[method_name].append(result)
            sim.apply_result(result)

    # Aggregate metrics
    summary = {}
    for method, res_list in results.items():
        lats = [r.latency_ms for r in res_list]
        engs = [r.energy_j for r in res_list]
        utils = [r.utilization for r in res_list]
        times = [r.solve_time_ms for r in res_list]
        iters = [r.iterations for r in res_list]
        summary[method] = {
            "avg_latency_ms": round(np.mean(lats), 2),
            "avg_energy_j": round(np.mean(engs), 4),
            "avg_utilization": round(np.mean(utils), 4),
            "avg_solve_time_ms": round(np.mean(times), 3),
            "avg_iterations": round(np.mean(iters), 1),
            "results": res_list,
        }

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("  LQIM – Lightweight Quantum-Inspired Metaheuristic")
    print("  Edge Resource Scheduling Simulation")
    print("=" * 60)

    summary = run_simulation(n_tasks=50, n_nodes=10)

    print(f"\n{'Method':<8} {'Avg Latency':>14} {'Avg Energy':>12} {'Avg Util':>10} {'Solve Time':>12}")
    print("-" * 60)
    for method, stats in summary.items():
        print(
            f"{method:<8} "
            f"{stats['avg_latency_ms']:>12.1f}ms "
            f"{stats['avg_energy_j']:>10.4f}J "
            f"{stats['avg_utilization']*100:>9.1f}% "
            f"{stats['avg_solve_time_ms']:>10.1f}ms"
        )

    ga_lat = summary["GA"]["avg_latency_ms"]
    lqim_lat = summary["LQIM"]["avg_latency_ms"]
    ga_eng = summary["GA"]["avg_energy_j"]
    lqim_eng = summary["LQIM"]["avg_energy_j"]
    ga_util = summary["GA"]["avg_utilization"]
    lqim_util = summary["LQIM"]["avg_utilization"]

    print(f"\n  Latency improvement vs GA:     {(ga_lat - lqim_lat) / ga_lat * 100:.1f}%")
    print(f"  Energy reduction vs GA:        {(ga_eng - lqim_eng) / ga_eng * 100:.1f}%")
    print(f"  Utilisation improvement vs GA: {(lqim_util - ga_util) / ga_util * 100:.1f}%")
    print()
