"""Unit tests for LQIM algorithm components."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from lqim import (
    QBit, LQIM, GeneticAlgorithmScheduler, PSOScheduler,
    EdgeNode, Task, EdgeSimulator, run_simulation
)


class TestQBit:
    def test_normalisation(self):
        q = QBit()
        assert abs(q.alpha**2 + q.beta**2 - 1.0) < 1e-9

    def test_measure_binary(self):
        q = QBit()
        for _ in range(20):
            assert q.measure() in (0, 1)

    def test_rotation_preserves_norm(self):
        q = QBit()
        for delta in [0.1, -0.1, 0.5, np.pi/4]:
            q.rotate(delta)
            assert abs(q.alpha**2 + q.beta**2 - 1.0) < 1e-8

    def test_rotate_toward_one(self):
        """Rotating upward should increase β²."""
        q = QBit(alpha=np.sqrt(0.9), beta=np.sqrt(0.1))
        initial_prob = q.prob_assigned
        for _ in range(10):
            q.rotate(0.1)
        assert q.prob_assigned > initial_prob

    def test_superposition_init(self):
        """Fresh Q-bit should be in superposition (neither extreme)."""
        probs = [QBit().prob_assigned for _ in range(100)]
        assert 0.05 < np.mean(probs) < 0.95


class TestEdgeNode:
    def test_available_cpu(self):
        node = EdgeNode(0, cpu_ghz=2.0, ram_gb=4, energy_budget=100, current_load=0.5)
        assert abs(node.available_cpu - 1.0) < 1e-9

    def test_utilisation(self):
        node = EdgeNode(0, cpu_ghz=2.0, ram_gb=4, energy_budget=100, current_load=0.3)
        assert node.utilization == 0.3


class TestLQIM:
    def _make_env(self):
        nodes = [
            EdgeNode(i, cpu_ghz=1.5+i*0.1, ram_gb=4, energy_budget=100, current_load=0.2+i*0.03)
            for i in range(5)
        ]
        task = Task(task_id=1, cpu_demand=0.8, memory_mb=200,
                    exec_time_ms=60, latency_sensitivity=0.7)
        return task, nodes

    def test_returns_valid_result(self):
        scheduler = LQIM(pop_size=10, max_iterations=20)
        task, nodes = self._make_env()
        result = scheduler.schedule(task, nodes)
        assert result.node_id in [n.node_id for n in nodes]
        assert result.latency_ms > 0
        assert result.energy_j > 0
        assert 0 <= result.utilization <= 1.0
        assert result.method == "LQIM"

    def test_respects_time_limit(self):
        import time
        scheduler = LQIM(pop_size=30, max_iterations=1000, time_limit_ms=60.0)
        task, nodes = self._make_env()
        start = time.perf_counter()
        scheduler.schedule(task, nodes)
        elapsed = (time.perf_counter() - start) * 1000
        # Should finish within 2x the time limit
        assert elapsed < 130

    def test_selects_best_node(self):
        """When one node is clearly better, LQIM should prefer it."""
        nodes = [
            EdgeNode(0, cpu_ghz=2.8, ram_gb=8, energy_budget=200, current_load=0.05),  # best
            EdgeNode(1, cpu_ghz=1.2, ram_gb=1, energy_budget=50, current_load=0.90),   # worst
            EdgeNode(2, cpu_ghz=1.5, ram_gb=2, energy_budget=80, current_load=0.70),
        ]
        task = Task(1, cpu_demand=1.0, memory_mb=100, exec_time_ms=50, latency_sensitivity=1.0)
        scheduler = LQIM(pop_size=20, max_iterations=60)

        wins = 0
        for _ in range(10):
            result = scheduler.schedule(task, nodes)
            if result.node_id == 0:
                wins += 1
        # Should prefer node 0 in majority of runs
        assert wins >= 6, f"LQIM should prefer best node, got {wins}/10"


class TestBaselines:
    def _make_env(self):
        nodes = [
            EdgeNode(i, cpu_ghz=1.5+i*0.2, ram_gb=4, energy_budget=100, current_load=0.2)
            for i in range(5)
        ]
        task = Task(1, cpu_demand=0.8, memory_mb=200, exec_time_ms=60, latency_sensitivity=0.7)
        return task, nodes

    def test_ga_returns_valid(self):
        ga = GeneticAlgorithmScheduler(pop_size=10, generations=20)
        task, nodes = self._make_env()
        result = ga.schedule(task, nodes)
        assert result.method == "GA"
        assert result.node_id in range(5)

    def test_pso_returns_valid(self):
        pso = PSOScheduler(n_particles=10, iterations=20)
        task, nodes = self._make_env()
        result = pso.schedule(task, nodes)
        assert result.method == "PSO"
        assert result.node_id in range(5)


class TestSimulation:
    def test_run_simulation(self):
        summary = run_simulation(n_tasks=10, n_nodes=5, seed=99)
        for method in ["LQIM", "GA", "PSO"]:
            assert method in summary
            assert summary[method]["avg_latency_ms"] > 0
            assert len(summary[method]["results"]) == 10

    def test_lqim_competitive(self):
        """LQIM should achieve lower or comparable latency to GA in simulation."""
        summary = run_simulation(n_tasks=30, n_nodes=8, seed=42)
        lqim_lat = summary["LQIM"]["avg_latency_ms"]
        ga_lat = summary["GA"]["avg_latency_ms"]
        # LQIM should not be more than 20% worse than GA on avg latency
        assert lqim_lat <= ga_lat * 1.20, f"LQIM latency {lqim_lat} too high vs GA {ga_lat}"


class TestEdgeSimulator:
    def test_generates_tasks(self):
        sim = EdgeSimulator(n_nodes=5, seed=1)
        tasks = [sim.generate_task() for _ in range(10)]
        ids = [t.task_id for t in tasks]
        assert ids == list(range(1, 11))
        for t in tasks:
            assert t.cpu_demand > 0
            assert t.exec_time_ms > 0

    def test_node_count(self):
        sim = EdgeSimulator(n_nodes=7, seed=2)
        assert len(sim.nodes) == 7

    def test_update_loads(self):
        sim = EdgeSimulator(n_nodes=5, seed=3)
        loads_before = [n.current_load for n in sim.nodes]
        sim.update_loads()
        loads_after = [n.current_load for n in sim.nodes]
        # At least some loads should change
        assert loads_before != loads_after


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
