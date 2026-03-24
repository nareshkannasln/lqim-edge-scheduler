"""
generate_results.py
Runs the LQIM simulation and saves results as JSON.
Usage: python generate_results.py
Output: results/simulation_data.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lqim import run_simulation, build_nodes, generate_tasks

def main():
    os.makedirs("results", exist_ok=True)
    results = run_simulation(50)

    output = {}
    for method, res_list in results.items():
        output[method] = {
            "summary": {
                "avg_latency_ms"  : round(sum(r.latency_ms  for r in res_list) / len(res_list), 2),
                "avg_energy_j"    : round(sum(r.energy_j    for r in res_list) / len(res_list), 4),
                "avg_utilization" : round(sum(r.utilization for r in res_list) / len(res_list), 2),
                "avg_iterations"  : round(sum(r.iterations  for r in res_list) / len(res_list), 1),
                "avg_solve_ms"    : round(sum(r.solve_time_ms for r in res_list) / len(res_list), 2),
            },
            "tasks": [
                {
                    "task_id"      : r.task_id,
                    "assigned_node": r.assigned_node,
                    "latency_ms"   : r.latency_ms,
                    "energy_j"     : r.energy_j,
                    "utilization"  : r.utilization,
                    "fitness"      : r.fitness,
                    "iterations"   : r.iterations,
                    "solve_ms"     : r.solve_time_ms,
                }
                for r in res_list
            ]
        }

    out_path = os.path.join("results", "simulation_data.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
