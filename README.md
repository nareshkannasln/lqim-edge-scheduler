# LQIM — Lightweight Quantum-Inspired Metaheuristic Edge Scheduler

> **Final Year Project** — B.Tech Artificial Intelligence & Data Science  
> Karpagam Institute of Technology · Anna University · April 2025

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)](#testing)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Algorithm Flow](#algorithm-flow)
- [System Architecture](#system-architecture)
- [Data Flow](#data-flow)
- [Results](#results)
- [Setup and Installation](#setup-and-installation)
- [Running the Demo](#running-the-demo)
- [Team](#team)

---

## Overview

**LQIM** is a novel resource scheduling algorithm for heterogeneous edge computing environments. It uses quantum-inspired computing principles — **Q-bit superposition** and **rotation gate mechanics** — to simultaneously explore all possible node assignments in parallel, enabling faster convergence than classical methods (GA, PSO) without requiring quantum hardware.

### Key Results

| Metric | LQIM | GA | PSO |
|--------|------|----|-----|
| Avg Latency | **58.6 ms** | 62.3 ms | 59.0 ms |
| Avg Energy | **0.149 J** | 0.165 J | 0.142 J |
| Avg Iterations | **~42** | 80 (fixed) | 80 (fixed) |
| Latency vs GA | **↓ 6.0%** | baseline | ↓ 5.3% |
| Energy vs GA | **↓ 9.3%** | baseline | ↓ 13.6% |

---

## Project Structure

```
lqim-edge-scheduler/
│
├── backend/                    # Core algorithm (Python)
│   ├── lqim.py                 # LQIM + GA + PSO algorithms
│   └── generate_results.py     # Run simulation → JSON output
│
├── frontend/                   # Demo interface (HTML/CSS/JS)
│   └── index.html              # 3-block interactive demo page
│
├── tests/                      # Unit tests
│   └── test_lqim.py
│
├── results/                    # Simulation output
│   └── simulation_data.json
│
├── docs/                       # Documentation
│   └── algorithm.md
│
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Algorithm Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    LQIM SCHEDULING CYCLE                        │
└─────────────────────────────────────────────────────────────────┘

  TASK ARRIVES
       │
       ▼
┌─────────────────┐
│  PHASE 1        │
│  Initialise     │  → Create 30 Q-bit individuals
│  Q-bit Pop      │  → Each Q-bit: α = β = 1/√2 (superposition)
└────────┬────────┘  → 30 individuals × 10 nodes = 300 Q-bits
         │
         ▼
┌─────────────────┐
│  PHASE 2        │
│  Measure →      │  → Collapse each Q-bit by probability β²
│  Candidates     │  → Generate 30 diverse node assignments
└────────┬────────┘  → Probabilistic exploration of solution space
         │
         ▼
┌─────────────────┐
│  PHASE 3        │
│  Compute        │  → F = 0.4·Latency + 0.35·Energy + 0.25·(1−Util)
│  Fitness F      │  → Evaluate all 30 candidates per iteration
└────────┬────────┘  → Lower F = better scheduling decision
         │
         ▼
┌─────────────────┐
│  PHASE 4        │
│  Update         │  → Track lowest fitness across all iterations
│  Global Best    │  → Store best node index + fitness value
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PHASE 5        │  → Δθ = +ROT if current ≠ best node
│  Rotation Gate  │  → Δθ = -ROT if current = best node
│  Update Δθ      │  → Rotate: [α', β'] = [cos·α + sin·β, -sin·α + cos·β]
└────────┬────────┘  → ROT = 0.05π, steers Q-bits toward global best
         │
         ▼
┌─────────────────┐
│  PHASE 6        │  → Check: avg(β²) across population > 0.92?
│  Convergence    │  → If YES and iter > 8 → CONVERGED → go to Phase 7
│  Check          │  → If NO and iter < 80 → LOOP BACK to Phase 2
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PHASE 7        │  → Dispatch task to global best node
│  Execute        │  → Update node live load
│  Schedule       │  → Return: node_id, latency, energy, iterations
└─────────────────┘

  RESULT RETURNED  ← assigned_node, latency_ms, energy_j, iterations
```

### Why Q-bit Superposition is Faster

```
CLASSICAL (GA / PSO):          QUANTUM-INSPIRED (LQIM):
                               
Iter 1: Try node 3             Iter 1: ALL nodes simultaneously
Iter 2: Try node 7                     via β² probability distribution
Iter 3: Try node 1             Iter 2: Rotate gates → bias toward best
Iter 4: Try node 4             Iter 3: Population converges on optimal
...                            ...
Iter 80: STOP (fixed)          Iter 42: CONVERGED (early stop)
                               
Sequential exploration          Parallel probabilistic exploration
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        USER / IOT DEVICE                             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  Task Parameters
                           │  (cpu, mem, exec_time, latency_sens)
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     BLOCK 1 — TASK INPUT                             │
│                                                                      │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────────────┐   │
│  │ Manual Input│  │  API / Live Feed  │  │    Preset Tasks       │   │
│  │  (Sliders)  │  │  REST · MQTT · JSON│  │ IoT·DB·Camera·Health│   │
│  └──────┬──────┘  └────────┬─────────┘  └──────────┬────────────┘   │
│         └──────────────────┴───────────────────────┘                │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   BLOCK 2 — ALGORITHM PROCESSING                     │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    LQIM CORE ENGINE                            │  │
│  │                                                                │  │
│  │  Q-bit Init → Measure → Fitness → Best → Rotate → Converge    │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │  GA Baseline │  │ PSO Baseline │  │  Live Animation + Log    │   │
│  │  (80 iters)  │  │  (80 iters)  │  │  Q-bits · Progress Bar   │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     BLOCK 3 — OUTPUT & COMPARISON                    │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │  Assigned    │  │   Metrics    │  │   Speed Comparison       │   │
│  │  Node Info   │  │ Lat·Eng·Load │  │  LQIM vs GA vs PSO bars  │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │              All 10 Node Load Bars                           │    │
│  │  N-00 ████░░░░  34%    N-04 ████████░░  67% ← assigned      │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
                      ┌─────────────────────────────┐
                      │       REAL WORLD EVENT       │
                      │                              │
  Factory sensor  ──► │  Temperature spike detected  │
  Hospital ECG    ──► │  Patient vitals anomaly       │
  CCTV camera     ──► │  Frame captured at 30fps      │
  ERP DB query    ──► │  1L records requested         │
  Vehicle LIDAR   ──► │  Point cloud scan complete    │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │   TASK OBJECT (4 numbers)   │
                      │                             │
                      │  cpu_demand   : 0.8 GHz     │
                      │  memory_mb    : 200 MB       │
                      │  exec_time_ms : 60 ms        │
                      │  latency_sens : 0.7          │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │       LQIM SCHEDULER        │
                      │   (runs in < 50ms)          │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │      SCHEDULE RESULT        │
                      │                             │
                      │  assigned_node : Node-04    │
                      │  latency_ms    : 41.2 ms    │
                      │  energy_j      : 0.0912 J   │
                      │  utilization   : 68.4 %     │
                      │  iterations    : 38          │
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │    EDGE NODE EXECUTES TASK  │
                      │    Result returned < 50ms   │
                      └─────────────────────────────┘
```

---

## Results

Simulation: 50 tasks · 10 heterogeneous edge nodes · Seed = 42

```
Algorithm  Avg Latency   Avg Energy   Avg Util   Avg Iters   vs GA (Lat)
─────────  ──────────    ──────────   ────────   ─────────   ──────────
LQIM       58.6 ms       0.149 J      72.8 %     ~42         ↓ 6.0%
GA         62.3 ms       0.165 J      77.0 %     80          baseline
PSO        59.0 ms       0.142 J      79.5 %     80          ↓ 5.3%
```

---

## Setup and Installation

### Prerequisites

```bash
Python 3.10+
numpy
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Simulation (Python backend)

```bash
cd backend
python lqim.py
```

### Generate JSON Results

```bash
cd backend
python generate_results.py
# Output: results/simulation_data.json
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Running the Demo

### Option 1 — Open Directly (no server needed)

```bash
open frontend/index.html
# or double-click the file in your file manager
```

### Option 2 — Local HTTP Server

```bash
cd frontend
python -m http.server 8000
# Visit http://localhost:8000
```

### Demo Walkthrough

1. **Block 1** — Choose input mode:
   - *Manual*: Adjust sliders for CPU, memory, exec time, sensitivity
   - *API*: Paste a REST endpoint URL and click Fetch
   - *Preset*: Click a real-world scenario (IoT, DB, Camera, Health, Vehicle)

2. **Click ▶ Run LQIM Scheduler**

3. **Block 2** — Watch the algorithm live:
   - Q-bit squares flash in superposition → collapse to winner (green)
   - 7 pipeline steps animate with ✓ on completion
   - Iteration progress bar fills as rotation gates run
   - Terminal log shows real-time algorithm output

4. **Block 3** — See results:
   - Assigned node with latency, energy, and load
   - Speed comparison: LQIM vs GA vs PSO bar charts
   - All 10 node load bars with assigned node marked ←

---

## Real-World Use Cases

| Domain | Use Case | Latency Sensitivity |
|--------|----------|-------------------|
| 🏥 Healthcare | Patient monitor · ECG · SpO2 | Critical (1.0) |
| 🚗 Autonomous Vehicles | LIDAR · Camera frames | Critical (1.0) |
| 🏭 Smart Factory | Sensor streams · Robot control | High (0.8–0.9) |
| 📡 5G MEC | Base station task offloading | High (0.7–0.9) |
| 🏙️ Smart Cities | CCTV analytics · Traffic | Medium (0.6–0.8) |
| 🌾 Precision Agriculture | Drone imaging · Soil sensors | Medium (0.5–0.7) |
| 🗄️ ERP/Database | 1L record queries (multi-node) | Medium (0.4–0.6) |

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
**University:** Anna University, Chennai — 600025

---

## License

MIT License — see [LICENSE](LICENSE) for details.
