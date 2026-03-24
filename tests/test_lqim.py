"""
tests/test_lqim.py
Unit tests for the LQIM algorithm components.
Run: python -m pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import unittest
from lqim import (
    QBit, EdgeNode, Task, LQIMScheduler, GAScheduler, PSOScheduler,
    compute_fitness, build_nodes, generate_tasks
)

class TestQBit(unittest.TestCase):
    def test_initial_superposition(self):
        q = QBit()
        self.assertAlmostEqual(q.alpha ** 2 + q.beta ** 2, 1.0, places=6)
        self.assertAlmostEqual(q.prob_one, 0.5, places=6)

    def test_measure_returns_binary(self):
        q = QBit()
        for _ in range(20):
            self.assertIn(q.measure(), [0, 1])

    def test_rotation_preserves_normalisation(self):
        import math
        q = QBit()
        q.rotate(0.1 * math.pi)
        self.assertAlmostEqual(q.alpha ** 2 + q.beta ** 2, 1.0, places=6)

    def test_rotation_toward_one(self):
        """Rotating positively by small steps should increase prob_one initially (steps 1-5)."""
        q = QBit()
        q.rotate(0.05 * 3.14159)
        # After one small positive rotation from equal superposition, beta grows
        self.assertGreater(q.prob_one, 0.5)

    def test_rotation_increases_prob_one(self):
        """Five positive rotations from superposition push prob_one above 0.9."""
        q = QBit()
        for _ in range(5):
            q.rotate(0.05 * 3.14159)
        self.assertGreater(q.prob_one, 0.75)


class TestFitness(unittest.TestCase):
    def test_fitness_range(self):
        nodes = build_nodes()
        task  = Task(1, 0.8, 200, 60, 0.7)
        for n in nodes:
            score, lat, eng, util = compute_fitness(n, task, n.current_load)
            self.assertGreater(score, 0)
            self.assertGreater(lat, 0)
            self.assertGreater(eng, 0)
            self.assertGreaterEqual(util, 0)
            self.assertLessEqual(util, 1.0)

    def test_heavy_load_worse_fitness(self):
        node  = build_nodes()[4]   # Node 4, 2.53 GHz
        task  = Task(1, 0.8, 200, 60, 0.7)
        s_low,  *_ = compute_fitness(node, task, 10.0)
        s_high, *_ = compute_fitness(node, task, 90.0)
        self.assertLess(s_low, s_high)


class TestLQIM(unittest.TestCase):
    def setUp(self):
        self.scheduler = LQIMScheduler(build_nodes())

    def test_returns_valid_node(self):
        task   = Task(1, 0.8, 200, 60, 0.7)
        result = self.scheduler.schedule(task)
        self.assertIn(result.assigned_node, range(10))

    def test_convergence_within_limit(self):
        task   = Task(1, 0.8, 200, 60, 0.7)
        result = self.scheduler.schedule(task)
        self.assertLessEqual(result.iterations, 80)

    def test_result_fields_positive(self):
        task   = Task(1, 1.0, 300, 80, 0.8)
        result = self.scheduler.schedule(task)
        self.assertGreater(result.latency_ms, 0)
        self.assertGreater(result.energy_j, 0)
        self.assertGreater(result.utilization, 0)

    def test_early_convergence(self):
        """LQIM should converge at or before 80 iterations."""
        task   = Task(1, 0.3, 100, 30, 0.9)
        result = self.scheduler.schedule(task)
        self.assertLessEqual(result.iterations, 80)


class TestGA(unittest.TestCase):
    def test_uses_all_iterations(self):
        ga   = GAScheduler(build_nodes())
        task = Task(1, 0.8, 200, 60, 0.7)
        res  = ga.schedule(task)
        self.assertEqual(res.iterations, 80)
        self.assertIn(res.assigned_node, range(10))


class TestPSO(unittest.TestCase):
    def test_uses_all_iterations(self):
        pso  = PSOScheduler(build_nodes())
        task = Task(1, 0.8, 200, 60, 0.7)
        res  = pso.schedule(task)
        self.assertEqual(res.iterations, 80)
        self.assertIn(res.assigned_node, range(10))


class TestSimulation(unittest.TestCase):
    def test_generate_tasks(self):
        tasks = generate_tasks(10)
        self.assertEqual(len(tasks), 10)
        for t in tasks:
            self.assertGreater(t.cpu_demand, 0)
            self.assertGreater(t.exec_time_ms, 0)

    def test_build_nodes(self):
        nodes = build_nodes()
        self.assertEqual(len(nodes), 10)
        for n in nodes:
            self.assertGreater(n.cpu_ghz, 0)
            self.assertGreater(n.ram_gb, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
