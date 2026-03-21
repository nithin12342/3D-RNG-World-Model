### 2. Intention Engineering File (`intent_def.md`)

**Target Agent:** Product/Goal Alignment Agent

**Purpose:** Defines the success metrics and the "why" behind the architecture so agents can make autonomous optimization decisions.

* **Primary Objective:** Achieve functional task learning (e.g., XOR logic, MNIST classification) using a fraction of the active compute required by a standard Multi-Layer Perceptron (MLP).
* **Optimization Target 1 (Sparse Activation):** Maximize sparsity. A successful inference should only activate the nodes directly on the DFS path (target: <5% network activation per pass).
* **Optimization Target 2 (Emergent Specialization):** The graph must organically form specialized sub-sections based on input types without human-coded layers.
* **Optimization Target 3 (Resilience):** The system must gracefully degrade and self-heal. If 10% of nodes are randomly deleted post-training, the routing algorithm must find alternative paths to restore accuracy within minimal training cycles.