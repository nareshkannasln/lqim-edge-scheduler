"""
tests/test_lqim.py
Unit tests for the LQIM algorithm components.
Run: python -m pytest tests/ -v
"""

import sys
import os
import math
import copy
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import unittest
from lqim import (
    QBit, EdgeNode, Task, ScheduleResult,
    LQIMScheduler, GAScheduler, PSOScheduler,
    compute_fitness, build_nodes, generate_tasks
)


class TestQBit(unittest.TestCase):
    """Tests for the quantum bit representation."""

    def test_initial_superposition(self):
        q = QBit()
        self.assertAlmostEqual(q.alpha ** 2 + q.beta ** 2, 1.0, places=6)
        self.assertAlmostEqual(q.prob_one, 0.5, places=6)

    def test_measure_returns_binary(self):
        q = QBit()
        for _ in range(50):
            self.assertIn(q.measure(), [0, 1])

    def test_rotation_preserves_normalisation(self):
        q = QBit()
        for angle in [0.01, 0.05, 0.1, 0.3, -0.1, -0.3]:
            q.rotate(angle * math.pi)
            self.assertAlmostEqual(q.alpha ** 2 + q.beta ** 2, 1.0, places=6)

    def test_positive_rotation_increases_beta(self):
        q = QBit()
        q.rotate(0.05 * math.pi)
        self.assertGreater(q.prob_one, 0.5)

    def test_negative_rotation_decreases_beta(self):
        q = QBit()
        q.rotate(-0.05 * math.pi)
        self.assertLess(q.prob_one, 0.5)

    def test_multiple_rotations_converge(self):
        q = QBit()
        for _ in range(5):
            q.rotate(0.05 * math.pi)
        self.assertGreater(q.prob_one, 0.7)

    def test_zero_rotation_no_change(self):
        q = QBit()
        old_alpha, old_beta = q.alpha, q.beta
        q.rotate(0.0)
        self.assertAlmostEqual(q.alpha, old_alpha, places=10)
        self.assertAlmostEqual(q.beta, old_beta, places=10)


class TestFitness(unittest.TestCase):
    """Tests for the multi-objective fitness function."""

    def test_fitness_returns_four_values(self):
        node = build_nodes()[0]
        task = Task(1, 0.8, 200, 60, 0.7)
        result = compute_fitness(node, task, node.current_load)
        self.assertEqual(len(result), 4)

    def test_fitness_all_positive(self):
        for node in build_nodes():
            task = Task(1, 0.8, 200, 60, 0.7)
            score, lat, eng, util = compute_fitness(node, task, node.current_load)
            self.assertGreater(score, 0)
            self.assertGreater(lat, 0)
            self.assertGreater(eng, 0)
            self.assertGreaterEqual(util, 0)
            self.assertLessEqual(util, 1.0)

    def test_heavier_load_worse_fitness(self):
        node = build_nodes()[4]  # 2.53 GHz
        task = Task(1, 0.8, 200, 60, 0.7)
        s_low, *_ = compute_fitness(node, task, 10.0)
        s_high, *_ = compute_fitness(node, task, 90.0)
        self.assertLess(s_low, s_high)

    def test_higher_cpu_node_lower_latency(self):
        slow = EdgeNode(0, 1.0, 4, 20, 100)
        fast = EdgeNode(1, 2.5, 4, 20, 100)
        task = Task(1, 0.8, 200, 60, 0.7)
        _, lat_slow, *_ = compute_fitness(slow, task, 20)
        _, lat_fast, *_ = compute_fitness(fast, task, 20)
        self.assertLess(lat_fast, lat_slow)

    def test_memory_pressure_penalty(self):
        node = EdgeNode(0, 2.0, 1, 10, 100)  # 1 GB = 1024 MB
        light = Task(1, 0.5, 100, 50, 0.5)   # 100/1024 = 10%
        heavy = Task(2, 0.5, 900, 50, 0.5)   # 900/1024 = 88% > 80%
        s_light, *_ = compute_fitness(node, light, 10)
        s_heavy, *_ = compute_fitness(node, heavy, 10)
        self.assertLess(s_light, s_heavy)

    def test_custom_weights(self):
        node = build_nodes()[0]
        task = Task(1, 0.8, 200, 60, 0.7)
        s1, *_ = compute_fitness(node, task, 20, w1=1.0, w2=0.0, w3=0.0)
        s2, *_ = compute_fitness(node, task, 20, w1=0.0, w2=1.0, w3=0.0)
        # Different weights should give different scores
        self.assertNotAlmostEqual(s1, s2, places=4)


class TestLQIM(unittest.TestCase):
    """Tests for the LQIM scheduler."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        self.scheduler = LQIMScheduler(build_nodes())

    def test_returns_valid_node(self):
        task = Task(1, 0.8, 200, 60, 0.7)
        result = self.scheduler.schedule(task)
        self.assertIn(result.assigned_node, range(10))

    def test_returns_schedule_result(self):
        task = Task(1, 0.8, 200, 60, 0.7)
        result = self.scheduler.schedule(task)
        self.assertIsInstance(result, ScheduleResult)
        self.assertEqual(result.method, "LQIM")

    def test_convergence_before_max(self):
        task = Task(1, 0.8, 200, 60, 0.7)
        result = self.scheduler.schedule(task)
        self.assertLessEqual(result.iterations, 80)

    def test_early_convergence_happens(self):
        """LQIM should converge before 80 iterations on simple tasks."""
        task = Task(1, 0.3, 100, 30, 0.5)
        result = self.scheduler.schedule(task)
        self.assertLess(result.iterations, 80)

    def test_result_fields_positive(self):
        task = Task(1, 1.0, 300, 80, 0.8)
        result = self.scheduler.schedule(task)
        self.assertGreater(result.latency_ms, 0)
        self.assertGreater(result.energy_j, 0)
        self.assertGreater(result.utilization, 0)
        self.assertGreater(result.solve_time_ms, 0)

    def test_load_updates_after_scheduling(self):
        scheduler = LQIMScheduler(build_nodes())
        initial_loads = scheduler.live_loads[:]
        task = Task(1, 0.8, 200, 60, 0.7)
        result = scheduler.schedule(task)
        # The assigned node's load should have increased
        self.assertGreater(
            scheduler.live_loads[result.assigned_node],
            initial_loads[result.assigned_node]
        )

    def test_multiple_tasks_increase_load(self):
        scheduler = LQIMScheduler(build_nodes())
        total_initial = sum(scheduler.live_loads)
        for i in range(10):
            scheduler.schedule(Task(i, 0.5, 150, 40, 0.6))
        total_after = sum(scheduler.live_loads)
        self.assertGreater(total_after, total_initial)


class TestGA(unittest.TestCase):
    """Tests for the GA baseline."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_uses_all_iterations(self):
        ga = GAScheduler(build_nodes())
        task = Task(1, 0.8, 200, 60, 0.7)
        res = ga.schedule(task)
        self.assertEqual(res.iterations, 80)

    def test_returns_valid_node(self):
        ga = GAScheduler(build_nodes())
        task = Task(1, 0.8, 200, 60, 0.7)
        res = ga.schedule(task)
        self.assertIn(res.assigned_node, range(10))
        self.assertEqual(res.method, "GA")

    def test_positive_results(self):
        ga = GAScheduler(build_nodes())
        task = Task(1, 0.8, 200, 60, 0.7)
        res = ga.schedule(task)
        self.assertGreater(res.latency_ms, 0)
        self.assertGreater(res.energy_j, 0)


class TestPSO(unittest.TestCase):
    """Tests for the PSO baseline."""

    def setUp(self):
        random.seed(42)
        np.random.seed(42)

    def test_uses_all_iterations(self):
        pso = PSOScheduler(build_nodes())
        task = Task(1, 0.8, 200, 60, 0.7)
        res = pso.schedule(task)
        self.assertEqual(res.iterations, 80)

    def test_returns_valid_node(self):
        pso = PSOScheduler(build_nodes())
        task = Task(1, 0.8, 200, 60, 0.7)
        res = pso.schedule(task)
        self.assertIn(res.assigned_node, range(10))
        self.assertEqual(res.method, "PSO")

    def test_positive_results(self):
        pso = PSOScheduler(build_nodes())
        task = Task(1, 0.8, 200, 60, 0.7)
        res = pso.schedule(task)
        self.assertGreater(res.latency_ms, 0)
        self.assertGreater(res.energy_j, 0)


class TestComparison(unittest.TestCase):
    """Tests comparing LQIM against baselines."""

    def test_lqim_fewer_iterations_than_ga(self):
        """LQIM should converge faster than GA's fixed 80."""
        random.seed(42)
        np.random.seed(42)
        lqim = LQIMScheduler(copy.deepcopy(build_nodes()))
        ga   = GAScheduler(copy.deepcopy(build_nodes()))

        tasks = generate_tasks(20)
        lqim_iters, ga_iters = 0, 0
        for t in tasks:
            rl = lqim.schedule(t); lqim_iters += rl.iterations
            rg = ga.schedule(t);   ga_iters += rg.iterations

        self.assertLess(lqim_iters, ga_iters)

    def test_lqim_comparable_quality(self):
        """LQIM latency should be within 15% of GA (comparable quality)."""
        random.seed(42)
        np.random.seed(42)
        lqim = LQIMScheduler(copy.deepcopy(build_nodes()))
        ga   = GAScheduler(copy.deepcopy(build_nodes()))

        tasks = generate_tasks(30)
        lqim_lat, ga_lat = 0, 0
        for t in tasks:
            rl = lqim.schedule(t); lqim_lat += rl.latency_ms
            rg = ga.schedule(t);   ga_lat += rg.latency_ms

        ratio = lqim_lat / ga_lat
        self.assertGreater(ratio, 0.85)
        self.assertLess(ratio, 1.15)


class TestGenerators(unittest.TestCase):
    """Tests for node and task generators."""

    def test_build_nodes_count(self):
        nodes = build_nodes()
        self.assertEqual(len(nodes), 10)

    def test_build_nodes_valid(self):
        for n in build_nodes():
            self.assertGreater(n.cpu_ghz, 0)
            self.assertGreater(n.ram_gb, 0)
            self.assertGreaterEqual(n.current_load, 0)
            self.assertLessEqual(n.current_load, 100)

    def test_generate_tasks_count(self):
        self.assertEqual(len(generate_tasks(10)), 10)
        self.assertEqual(len(generate_tasks(100)), 100)

    def test_generate_tasks_valid(self):
        random.seed(42)
        for t in generate_tasks(20):
            self.assertGreater(t.cpu_demand, 0)
            self.assertGreater(t.memory_mb, 0)
            self.assertGreater(t.exec_time_ms, 0)
            self.assertGreaterEqual(t.latency_sensitivity, 0)
            self.assertLessEqual(t.latency_sensitivity, 1.0)
            self.assertIn(t.task_type, ["IoT", "DB", "Camera", "Health", "Vehicle"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
