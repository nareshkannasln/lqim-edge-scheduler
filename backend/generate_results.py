"""
Generate simulation results as JSON for the frontend demo.
Run: python backend/generate_results.py
Output: results/simulation_data.json
"""

import sys
import os
import json
import copy
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from lqim import (
    LQIMScheduler, GAScheduler, PSOScheduler,
    build_nodes, generate_tasks, ScheduleResult
)


def main():
    random.seed(42)
    np.random.seed(42)
    tasks = generate_tasks(50)

    lqim = LQIMScheduler(copy.deepcopy(build_nodes()))
    ga   = GAScheduler(copy.deepcopy(build_nodes()))
    pso  = PSOScheduler(copy.deepcopy(build_nodes()))

    records = []
    for task in tasks:
        rl = lqim.schedule(task)
        rg = ga.schedule(task)
        rp = pso.schedule(task)
        records.append({
            "task_id":   task.id,
            "task_type": task.task_type,
            "cpu":       task.cpu_demand,
            "mem":       task.memory_mb,
            "exec":      task.exec_time_ms,
            "sens":      task.latency_sensitivity,
            "lqim": {"node": rl.assigned_node, "lat": rl.latency_ms, "eng": rl.energy_j, "util": rl.utilization, "iters": rl.iterations, "solve": rl.solve_time_ms},
            "ga":   {"node": rg.assigned_node, "lat": rg.latency_ms, "eng": rg.energy_j, "util": rg.utilization, "iters": rg.iterations, "solve": rg.solve_time_ms},
            "pso":  {"node": rp.assigned_node, "lat": rp.latency_ms, "eng": rp.energy_j, "util": rp.utilization, "iters": rp.iterations, "solve": rp.solve_time_ms},
        })

    # Summary
    n = len(records)
    summary = {}
    for method in ["lqim", "ga", "pso"]:
        summary[method] = {
            "avg_latency":    round(sum(r[method]["lat"]  for r in records) / n, 2),
            "avg_energy":     round(sum(r[method]["eng"]  for r in records) / n, 4),
            "avg_utilization": round(sum(r[method]["util"] for r in records) / n, 1),
            "avg_iterations": round(sum(r[method]["iters"] for r in records) / n, 1),
            "avg_solve_time": round(sum(r[method]["solve"] for r in records) / n, 2),
        }

    output = {"summary": summary, "tasks": records}

    os.makedirs("results", exist_ok=True)
    with open("results/simulation_data.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated results/simulation_data.json ({n} tasks)")
    print(f"  LQIM: {summary['lqim']['avg_latency']}ms lat, {summary['lqim']['avg_iterations']} iters")
    print(f"  GA:   {summary['ga']['avg_latency']}ms lat, {summary['ga']['avg_iterations']} iters")
    print(f"  PSO:  {summary['pso']['avg_latency']}ms lat, {summary['pso']['avg_iterations']} iters")


if __name__ == "__main__":
    main()
