# LQIM Algorithm — Technical Documentation

## Mathematical Foundation

### Q-bit Representation

Each task-node assignment is encoded as a quantum bit:

```
|ψ⟩ = α|0⟩ + β|1⟩,  where α² + β² = 1
```

- α: amplitude of NOT assigning the task to this node
- β: amplitude of assigning the task to this node
- Initially: α = β = 1/√2 (equal superposition, 50% probability each)

### Measurement

Each Q-bit collapses based on β²:
- Random number r in [0, 1)
- If r < β² → collapse to |1⟩ (assign)
- If r ≥ β² → collapse to |0⟩ (don't assign)

### Rotation Gate

The rotation operator updates amplitudes to steer toward the best solution:

```
[α']   [cos(θ)  -sin(θ)] [α]
[β'] = [sin(θ)   cos(θ)] [β]
```

LQIM uses **adaptive rotation**: θ decays as iterations progress.

```
θ = ROT × max(0.4, 1 − iter / MAX_ITER)
```

- For the best node: rotate by +θ (increase β, increase probability)
- For other nodes: rotate by -0.4θ (decrease β, decrease probability)

### Convergence Detection

After a minimum number of iterations (12), check if the population has converged:

```
avg(β²[best_node]) across all individuals > 0.92 → CONVERGED → STOP
```

This is the key advantage: LQIM stops early when the population agrees on the optimal node.

## Fitness Function

```
F = 0.4 × (latency/300) + 0.35 × (energy/5) + 0.25 × (1 − utilization)
```

Where:
- **latency** = exec_time / cpu_ratio × (1 + load × 0.5) × (1 + sensitivity × 0.3)
- **energy** = cpu_demand × 2.5 × (latency / 1000)
- **utilization** = min(load + cpu_demand / node_cpu, 1.0)

Memory pressure adds a penalty when task memory > 80% of node RAM.

## Comparison with Baselines

| Property | LQIM | GA | PSO |
|----------|------|----|-----|
| Representation | Q-bit amplitudes | Integer (node index) | Continuous position |
| Update mechanism | Rotation gate | Crossover + mutation | Velocity + position |
| Exploration | Probabilistic (β²) | Random mutation | Particle swarm |
| Convergence | Adaptive (early stop) | Fixed iterations | Fixed iterations |
| Typical iterations | ~38 | 80 | 80 |
