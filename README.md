# LQIM — Lightweight Quantum-Inspired Metaheuristic Edge Scheduler

> **Final Year Project** — B.Tech Artificial Intelligence & Data Science  
> Karpagam Institute of Technology · Anna University · 2025

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-32%20Passed-brightgreen)](#testing)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What is LQIM?

LQIM is a **real-time task scheduling algorithm** for edge computing environments. It decides which edge server should handle an incoming task (IoT sensor data, camera frame, health monitor reading, etc.) by optimizing across three objectives simultaneously: **low latency**, **low energy consumption**, and **high resource utilization**.

The key innovation is using **quantum-inspired computing principles** — Q-bit superposition and rotation gate mechanics — to explore the solution space more efficiently than classical algorithms. This enables **early convergence**: LQIM finds near-optimal solutions in ~40–55% fewer iterations than Genetic Algorithms (GA) or Particle Swarm Optimization (PSO).

### Why does this matter?

In real-time edge environments, **the time spent deciding where to schedule a task is itself a source of latency**. A scheduler that converges in 38 iterations instead of 80 makes decisions nearly 2× faster — critical when tasks arrive every few milliseconds from sensors, cameras, or medical devices.

---

## Product Feasibility

### Target Users

| User | Need | How LQIM Helps |
|------|------|----------------|
| **Edge Platform Operators** (AWS Wavelength, Azure Edge, Cloudflare Workers) | Schedule millions of tasks/day across heterogeneous edge nodes | Drop-in scheduling engine that auto-balances load, latency, and energy |
| **Smart Factory Managers** | Real-time sensor data processing with strict latency SLAs | Sub-50ms scheduling decisions for IoT streams |
| **Healthcare IoT Providers** | Continuous patient monitoring with guaranteed response times | Latency-sensitive scheduling with energy awareness |
| **Autonomous Vehicle Platforms** | Ultra-low-latency LIDAR/camera processing | Fastest convergence among compared algorithms |
| **5G MEC Operators** | Multi-access edge computing task offloading | Configurable weight parameters (latency vs energy vs utilization) |

### Deployment Architecture

```
┌────────────────────────────────────────────────────────┐
│                   IoT / Edge Devices                    │
│  Sensors · Cameras · Wearables · Vehicles · Drones     │
└───────────────────────┬────────────────────────────────┘
                        │ Tasks (JSON)
                        ▼
┌────────────────────────────────────────────────────────┐
│              LQIM Scheduling Engine                     │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐    │
│  │ Q-bit    │  │ Fitness  │  │ Adaptive Rotation │    │
│  │ Pop Init │→ │ Evaluate │→ │ + Convergence     │    │
│  └──────────┘  └──────────┘  └───────────────────┘    │
│                                                         │
│  Input:  task params (CPU, mem, time, sensitivity)      │
│  Output: optimal node assignment + metrics              │
│  Speed:  ~38 iterations (vs 80 for GA/PSO)              │
└───────────────────────┬────────────────────────────────┘
                        │ Assignment
                        ▼
┌────────────────────────────────────────────────────────┐
│              Heterogeneous Edge Nodes                   │
│  Node-0 (1.8GHz, 2GB)  ···  Node-9 (2.0GHz, 2GB)     │
│  Each with different CPU, RAM, load, energy budget      │
└────────────────────────────────────────────────────────┘
```

### Integration Options

LQIM can be deployed as:

1. **Python library** — `pip install` and call `LQIMScheduler.schedule(task)` directly
2. **REST API microservice** — wrap with Flask/FastAPI, deploy as a sidecar to edge orchestrators
3. **Embedded scheduler** — lightweight enough to run on edge gateways (Raspberry Pi, Jetson Nano)
4. **Kubernetes scheduler plugin** — extend K8s/K3s with LQIM-based pod placement

### Competitive Advantage

| Feature | LQIM | Round-Robin | GA | PSO |
|---------|------|-------------|----|----|
| Multi-objective optimization | ✓ | ✗ | ✓ | ✓ |
| Early convergence | ✓ (~38 iters) | N/A | ✗ (80 fixed) | ✗ (80 fixed) |
| Adaptive step size | ✓ | ✗ | ✗ | Partial |
| No quantum hardware needed | ✓ | ✓ | ✓ | ✓ |
| Configurable weights | ✓ | ✗ | ✓ | ✓ |
| Real-time capable (<50ms) | ✓ | ✓ | ✓ | ✓ |

---

## How the Algorithm Works

### Core Idea

Each task needs to be assigned to one of N edge nodes. Classical algorithms (GA, PSO) test one node assignment per individual per iteration. LQIM represents each assignment as a **Q-bit** — a probability distribution over nodes — allowing the entire population to simultaneously explore the full solution space.

### The 7-Phase Pipeline

```
Phase 1: Initialize Q-bit Population
         30 individuals × 10 Q-bits each
         All Q-bits start in equal superposition (α = β = 1/√2)

Phase 2: Measure → Candidate Assignments
         Collapse Q-bits probabilistically (P(node) = β²)
         Each individual selects a node based on Q-bit amplitudes

Phase 3: Compute Fitness
         F = 0.4 × Latency + 0.35 × Energy + 0.25 × (1 − Utilization)
         Lower F = better scheduling decision

Phase 4: Update Global Best
         Track the node assignment with lowest fitness across all iterations

Phase 5: Rotation Gate Update (Adaptive)
         θ = ROT × decay_factor (step shrinks as population converges)
         Rotate best-node Q-bits toward |1⟩, others toward |0⟩

Phase 6: Convergence Check
         If avg(β²) for best node > 0.92 and iteration > 12 → STOP
         This is LQIM's key advantage: early stopping saves iterations

Phase 7: Execute Schedule
         Dispatch task to the globally best node
```

### Fitness Function

The multi-objective fitness function evaluates each node assignment:

```
F = w₁ × Latency_norm + w₂ × Energy_norm + w₃ × (1 − Utilization)
```

Where:
- **Latency** = execution time scaled by node load, CPU availability, and memory pressure
- **Energy** = CPU demand × 2.5 × (latency / 1000) — proportional to computation time
- **Utilization** = current load + task CPU / node CPU — higher is better (more efficient)
- **Weights** (w₁=0.4, w₂=0.35, w₃=0.25) are configurable per deployment

---

## Results

### Simulation Setup

- **Nodes**: 10 heterogeneous edge nodes (1.2–2.5 GHz CPU, 1–8 GB RAM)
- **Tasks**: 50 tasks with Poisson-like arrivals across 5 types (IoT, DB, Camera, Health, Vehicle)
- **Seed**: Fixed at 42 for reproducibility
- **Each scheduler gets independent node state** (fair comparison)

### Summary

| Metric | LQIM | GA | PSO |
|--------|------|----|----|
| Avg Latency | ~48 ms | ~48 ms | ~48 ms |
| Avg Energy | ~0.13 J | ~0.13 J | ~0.13 J |
| Avg Iterations | **~38** | 80 (fixed) | 80 (fixed) |
| Convergence | **Early stop** | No | No |

### Key Finding

LQIM achieves **comparable scheduling quality** (within 1-2% of GA/PSO on latency and energy) while using **~50% fewer iterations**. In real-time edge environments where scheduling decisions must happen in milliseconds, this convergence speed advantage translates directly to faster task dispatch.

The iteration savings compound over time: for a system processing 10,000 tasks/hour, LQIM performs ~380,000 fewer fitness evaluations per hour than GA or PSO while maintaining equivalent solution quality.

---

## Project Structure

```
lqim-edge-scheduler/
├── backend/
│   ├── lqim.py              # Core: LQIM + GA + PSO algorithms
│   ├── app.py               # Flask API server (connects frontend ↔ backend)
│   └── generate_results.py  # Run simulation → JSON output
├── frontend/
│   └── index.html           # Interactive demo UI
├── tests/
│   └── test_lqim.py         # 32 unit tests
├── results/
│   └── simulation_data.json # Pre-generated results
├── requirements.txt
├── algorithm.md             # Algorithm documentation
└── README.md
```

### Architecture

```
Browser (index.html)  ──POST /api/schedule──►  Flask (app.py)  ──calls──►  lqim.py
                      ◄──── JSON results ────                              (Python LQIM/GA/PSO)
```

The frontend communicates with the Python backend through a REST API built with Flask. When a task is submitted in the UI, it is sent as a POST request to the Flask server, which runs the LQIM, GA, and PSO algorithms in Python and returns the results as JSON. The frontend then animates the 7-phase pipeline and displays the comparison.

If the Flask server is not running, the frontend automatically falls back to JavaScript implementations of the same algorithms, so the demo still works in standalone mode.

---

## Setup and Installation

### Prerequisites

```
Python 3.10+
numpy, flask, flask-cors, pytest
```

### Install

```bash
git clone https://github.com/nareshkannasln/lqim-edge-scheduler.git
cd lqim-edge-scheduler
pip install -r requirements.txt
```

### Run the Full Application (recommended)

```bash
python backend/app.py
```

Then open **http://localhost:5000** in your browser. This serves the frontend and connects it to the Python backend via API.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/schedule` | Schedule a single task (JSON body) |
| POST | `/api/schedule/batch` | Schedule multiple tasks at once |
| GET | `/api/nodes` | Get current node states and loads |
| GET | `/api/history` | Get all past scheduling results |
| POST | `/api/reset` | Reset schedulers to initial state |
| POST | `/api/generate` | Generate random tasks for testing |

### Example API Call

```bash
curl -X POST http://localhost:5000/api/schedule \
  -H "Content-Type: application/json" \
  -d '{"cpu_demand":0.8,"memory_mb":200,"exec_time_ms":60,"latency_sensitivity":0.7,"task_type":"Camera"}'
```

### Run Standalone Simulation (terminal only)

```bash
python backend/lqim.py
```

### Generate JSON Results

```bash
python backend/generate_results.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

Expected output: **32 passed**

### Run the Demo UI

```bash
# Option 1: Open directly
open frontend/index.html

# Option 2: Local server
cd frontend && python -m http.server 8000
# Visit http://localhost:8000
```

---

## Demo Walkthrough

1. **Start the server**: `python backend/app.py` → open **http://localhost:5000**
2. **Block 1 — Task Input** (4 modes):
   - *Custom Input*: Type exact values — CPU (GHz), Memory (MB), Execution Time (ms), Latency Sensitivity (0–1)
   - *Preset*: Click a real-world scenario (Factory IoT, CCTV, Patient Monitor, etc.)
   - *Batch / CSV*: Generate N random tasks or paste CSV data for bulk scheduling
   - *Live Simulator*: Start a real-time IoT data stream (generates tasks every 2 seconds)
3. **Click "Run LQIM Scheduler"**
4. **Block 2 — Processing**: Watch the 7-phase pipeline animate. The terminal log shows `[engine: python]` confirming the backend is connected.
5. **Block 3 — Results**: See the assigned node, latency/energy metrics, and side-by-side comparison bars (iterations + latency) for LQIM vs GA vs PSO.
6. **View Full Report**: Click to see a summary + task-by-task table of all scheduled tasks.

All three algorithms (LQIM, GA, PSO) run as **real independent implementations** in Python. The frontend calls the backend API — you can verify this in the browser's Network tab.

---

## Real-World Use Cases

| Domain | Task Type | Why LQIM Fits |
|--------|-----------|---------------|
| **Smart Factory** | Sensor anomaly detection | Fast convergence for continuous streams |
| **Healthcare** | ECG/SpO2 monitoring | Latency-sensitive weight configuration |
| **Autonomous Vehicles** | LIDAR processing | Fastest scheduling decisions |
| **5G MEC** | Task offloading | Multi-objective load balancing |
| **Smart Cities** | CCTV analytics | Energy-efficient scheduling |
| **Precision Agriculture** | Drone imaging | Battery-aware energy optimization |

---

## Technical Decisions

### Why quantum-inspired, not actual quantum?

Quantum computers are expensive, noisy, and require cryogenic cooling. LQIM simulates quantum mechanics principles (superposition, measurement, rotation gates) on classical hardware, making it deployable on any edge device with Python or JavaScript.

### Why not deep reinforcement learning?

DRL requires training data, GPU resources, and retraining when the environment changes. LQIM is model-free — it optimizes from scratch for each task in real time, adapting instantly to changing node loads without any training phase.

### Why early convergence matters

In a 10-node environment, the "optimal" node is often identifiable within 30-40 iterations. GA and PSO waste the remaining 40-50 iterations re-evaluating solutions that won't improve. LQIM's convergence detection avoids this waste, making it more efficient for real-time systems where every millisecond of scheduling delay adds to end-to-end latency.

---

## Team

| Name | Role |
|------|------|
| Dharanya A P | Algorithm design & testing |
| GandhimathiNathan T | Frontend development |
| Nareshkanna S | Backend & simulation |
| Sobika M | Documentation & analysis |
| **Mr. Vignesh M** | **Project Guide** |

**Department:** Artificial Intelligence & Data Science  
**College:** Karpagam Institute of Technology, Coimbatore — 641105

---

## License

MIT License — see [LICENSE](LICENSE) for details.
