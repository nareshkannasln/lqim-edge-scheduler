"""
LQIM Flask API Server
Connects the frontend UI to the Python LQIM/GA/PSO algorithms.

Run: python backend/app.py
Then open: http://localhost:5000

The frontend sends task parameters via POST /api/schedule
The backend runs LQIM, GA, PSO and returns JSON results.
"""

import sys
import os
import json
import copy
import random
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))
from lqim import (
    LQIMScheduler, GAScheduler, PSOScheduler,
    EdgeNode, Task, build_nodes, generate_tasks,
    compute_fitness
)

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# ── Persistent scheduler state (simulates a running edge cluster) ──
scheduler_state = {
    'lqim': None,
    'ga': None,
    'pso': None,
    'task_count': 0,
    'history': [],
}


def init_schedulers():
    """Reset all schedulers to fresh node state."""
    scheduler_state['lqim'] = LQIMScheduler(copy.deepcopy(build_nodes()))
    scheduler_state['ga']   = GAScheduler(copy.deepcopy(build_nodes()))
    scheduler_state['pso']  = PSOScheduler(copy.deepcopy(build_nodes()))
    scheduler_state['task_count'] = 0
    scheduler_state['history'] = []


init_schedulers()


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the frontend."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/schedule', methods=['POST'])
def schedule_task():
    """
    Schedule a single task through LQIM, GA, and PSO.

    POST body (JSON):
    {
        "cpu_demand": 0.8,
        "memory_mb": 200,
        "exec_time_ms": 60,
        "latency_sensitivity": 0.7,
        "task_type": "Camera"
    }

    Returns JSON with results from all 3 algorithms.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400

    scheduler_state['task_count'] += 1
    tid = scheduler_state['task_count']

    task = Task(
        id=tid,
        cpu_demand=float(data.get('cpu_demand', 0.8)),
        memory_mb=float(data.get('memory_mb', 200)),
        exec_time_ms=float(data.get('exec_time_ms', 60)),
        latency_sensitivity=float(data.get('latency_sensitivity', 0.7)),
        task_type=str(data.get('task_type', 'IoT')),
    )

    # Run all 3 schedulers
    r_lqim = scheduler_state['lqim'].schedule(task)
    r_ga   = scheduler_state['ga'].schedule(task)
    r_pso  = scheduler_state['pso'].schedule(task)

    result = {
        'task_id': tid,
        'task': {
            'cpu_demand': task.cpu_demand,
            'memory_mb': task.memory_mb,
            'exec_time_ms': task.exec_time_ms,
            'latency_sensitivity': task.latency_sensitivity,
            'task_type': task.task_type,
        },
        'lqim': {
            'assigned_node': r_lqim.assigned_node,
            'latency_ms': r_lqim.latency_ms,
            'energy_j': r_lqim.energy_j,
            'utilization': r_lqim.utilization,
            'fitness': r_lqim.fitness,
            'iterations': r_lqim.iterations,
            'solve_time_ms': r_lqim.solve_time_ms,
        },
        'ga': {
            'assigned_node': r_ga.assigned_node,
            'latency_ms': r_ga.latency_ms,
            'energy_j': r_ga.energy_j,
            'utilization': r_ga.utilization,
            'fitness': r_ga.fitness,
            'iterations': r_ga.iterations,
            'solve_time_ms': r_ga.solve_time_ms,
        },
        'pso': {
            'assigned_node': r_pso.assigned_node,
            'latency_ms': r_pso.latency_ms,
            'energy_j': r_pso.energy_j,
            'utilization': r_pso.utilization,
            'fitness': r_pso.fitness,
            'iterations': r_pso.iterations,
            'solve_time_ms': r_pso.solve_time_ms,
        },
        'node_loads': scheduler_state['lqim'].live_loads[:],
    }

    scheduler_state['history'].append(result)
    return jsonify(result)


@app.route('/api/schedule/batch', methods=['POST'])
def schedule_batch():
    """
    Schedule multiple tasks at once.

    POST body (JSON):
    {
        "tasks": [
            {"cpu_demand": 0.8, "memory_mb": 200, "exec_time_ms": 60, "latency_sensitivity": 0.7, "task_type": "Camera"},
            {"cpu_demand": 1.2, "memory_mb": 400, "exec_time_ms": 120, "latency_sensitivity": 0.5, "task_type": "DB"}
        ]
    }
    """
    data = request.get_json()
    if not data or 'tasks' not in data:
        return jsonify({'error': 'Provide {"tasks": [...]}'}), 400

    results = []
    for t in data['tasks']:
        scheduler_state['task_count'] += 1
        tid = scheduler_state['task_count']

        task = Task(
            id=tid,
            cpu_demand=float(t.get('cpu_demand', 0.5)),
            memory_mb=float(t.get('memory_mb', 200)),
            exec_time_ms=float(t.get('exec_time_ms', 50)),
            latency_sensitivity=float(t.get('latency_sensitivity', 0.7)),
            task_type=str(t.get('task_type', 'IoT')),
        )

        r_lqim = scheduler_state['lqim'].schedule(task)
        r_ga   = scheduler_state['ga'].schedule(task)
        r_pso  = scheduler_state['pso'].schedule(task)

        results.append({
            'task_id': tid,
            'task_type': task.task_type,
            'lqim': {'node': r_lqim.assigned_node, 'lat': r_lqim.latency_ms, 'eng': r_lqim.energy_j, 'iters': r_lqim.iterations},
            'ga':   {'node': r_ga.assigned_node, 'lat': r_ga.latency_ms, 'eng': r_ga.energy_j, 'iters': r_ga.iterations},
            'pso':  {'node': r_pso.assigned_node, 'lat': r_pso.latency_ms, 'eng': r_pso.energy_j, 'iters': r_pso.iterations},
        })

    # Summary
    n = len(results)
    summary = {
        'total_tasks': n,
        'lqim_avg_lat': round(sum(r['lqim']['lat'] for r in results) / n, 2),
        'ga_avg_lat':   round(sum(r['ga']['lat'] for r in results) / n, 2),
        'pso_avg_lat':  round(sum(r['pso']['lat'] for r in results) / n, 2),
        'lqim_avg_iters': round(sum(r['lqim']['iters'] for r in results) / n, 1),
        'node_loads': scheduler_state['lqim'].live_loads[:],
    }

    return jsonify({'summary': summary, 'results': results})


@app.route('/api/nodes', methods=['GET'])
def get_nodes():
    """Return current node states and load levels."""
    nodes = build_nodes()
    loads = scheduler_state['lqim'].live_loads[:]
    return jsonify({
        'nodes': [
            {
                'id': n.id,
                'cpu_ghz': n.cpu_ghz,
                'ram_gb': n.ram_gb,
                'initial_load': n.current_load,
                'current_load': round(loads[n.id], 1),
            }
            for n in nodes
        ]
    })


@app.route('/api/history', methods=['GET'])
def get_history():
    """Return all scheduling history for the report page."""
    return jsonify({
        'total': len(scheduler_state['history']),
        'results': scheduler_state['history'],
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset all schedulers and history."""
    init_schedulers()
    return jsonify({'status': 'reset', 'message': 'All schedulers reset to initial state'})


@app.route('/api/generate', methods=['POST'])
def generate_random():
    """
    Generate random tasks for testing.

    POST body: {"count": 10}
    """
    data = request.get_json() or {}
    count = int(data.get('count', 10))
    count = max(1, min(count, 200))

    random.seed()  # Use truly random for generated tasks
    tasks = generate_tasks(count)

    return jsonify({
        'tasks': [
            {
                'cpu_demand': t.cpu_demand,
                'memory_mb': t.memory_mb,
                'exec_time_ms': t.exec_time_ms,
                'latency_sensitivity': t.latency_sensitivity,
                'task_type': t.task_type,
            }
            for t in tasks
        ]
    })


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  LQIM Edge Scheduler — API Server")
    print("=" * 50)
    print(f"  Frontend:  http://localhost:5000")
    print(f"  API:       http://localhost:5000/api/schedule")
    print(f"  Nodes:     http://localhost:5000/api/nodes")
    print(f"  History:   http://localhost:5000/api/history")
    print("=" * 50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
