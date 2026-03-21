### 1. Context Engineering File (`context_def.md`)

**Target Agent:** Lead Architect Agent

**Purpose:** Establishes the fundamental laws of the environment and the core paradigm shift of the project.

* **Project Name:** 3D-RNG (3D Recursive Neural Graph)
* **Core Paradigm:** We are abandoning sequential layered architectures and global backpropagation. The system is a layerless, isotropic 3D grid of nodes.
* **Execution Rule 1 (No Backprop):** Global gradients are strictly forbidden. The system must use local learning rules based on Traceback Reinforcement (digital pheromones).
* **Execution Rule 2 (No Dense Matrix Multiply for Routing):** Routing is treated as a graph traversal problem, not a matrix multiplication problem. Agents must implement a Guided Non-Backtracking Depth-First Search (DFS).
* **Execution Rule 3 (Shared State):** All nodes share the same core recursive weight matrix for state updates, but maintain individual biases and edge weights.
* **System Analogy:** The network behaves like an ant colony searching a 3D maze for food. Signals drop pheromones on successful paths and avoid paths that dry up.