# LQIM Algorithm — Technical Documentation

## 1. Quantum Bit (Q-bit) Representation

A Q-bit encodes the probability of assigning a task to an edge node:

```
Qi = [αi]    |αi|² + |βi|² = 1
     [βi]

P(assigned)     = |βi|²
P(not assigned) = |αi|²
```

Initialisation: equal superposition → `α = β = 1/√2`, so P(assigned) = 0.5

## 2. Measurement (Probabilistic Collapse)

```python
def measure(q):
    if random() < q.beta²:
        return 1   # assign task to this node
    return 0       # do not assign
```

Repeated measurement across the population generates **diverse candidate solutions**, avoiding the premature convergence seen in deterministic algorithms.

## 3. Rotation Gate

The quantum rotation gate steers Q-bits toward the global best solution:

```
[α'] = [ cos θ   sin θ ] [α]
[β']   [-sin θ   cos θ ] [β]

θ_new = θ_old + Δθ
```

### Rotation Direction Table

| α state | β state | Current = Best? | Δθ |
|:---:|:---:|:---:|:---:|
| 0 | 1 | β < 0.5 | +Δθ (increase β) |
| 1 | 0 | β > 0.5 | −Δθ (increase α) |
| Same | Same | — | 0 |

## 4. Multi-Objective Fitness Function

```
F = w₁·L_norm + w₂·E_norm + w₃·(1 − U_norm)

where:
  L_norm = Latency / L_max         (normalised latency)
  E_norm = Energy / E_max          (normalised energy)
  U_norm = min(load + demand, 1.0) (node utilisation)

Default: w₁=0.4, w₂=0.35, w₃=0.25
```

Lower F = better solution.

## 5. Latency Model

```
Latency = exec_time × (1 + node_load) / speed_factor

speed_factor = max(available_cpu / task_cpu_demand, 0.1)
available_cpu = cpu_ghz × (1 − current_load)
```

## 6. Energy Model

```
Energy = cpu_demand × P_base × (latency / 1000)

P_base = 2.5 W per GHz (simplified edge power model)
```

## 7. Convergence Criterion

The algorithm converges when:
- Average β² of best node across population > 0.95 **and** iteration > 10, OR
- Iteration limit reached, OR
- Real-time limit (50 ms) exceeded

## 8. Complexity Analysis

| Component | Time Complexity |
|:--|:--|
| Q-bit initialisation | O(P × N) |
| Measurement + fitness | O(P × N) |
| Rotation update | O(P × N) |
| Full LQIM run | O(I × P × N) |

Where: P = population size, N = number of nodes, I = iterations

For default params (P=30, N=10, I=80): **24,000 operations** — easily real-time on edge hardware.

## 9. Comparison with Classical Methods

| Property | LQIM | GA | PSO |
|:--|:---:|:---:|:---:|
| Probabilistic search | ✅ | Partial | Partial |
| Local minima avoidance | ✅ | ❌ | ❌ |
| Superposition diversity | ✅ | ❌ | ❌ |
| Parameter sensitivity | Low | High | Medium |
| Real-time capable | ✅ | ✅ | ✅ |
| Hardware requirement | Classical | Classical | Classical |
