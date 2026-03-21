Here is the complete suite of engineering context files designed to be fed directly into an autonomous agentic AI platform (like an AI software developer or a swarm of coding agents).

These files provide the exact guardrails, architecture blueprints, and data pipelines necessary for an AI agent to build, train, and test our **3D Recursive Neural Graph (3D-RNG)** without defaulting to standard, backpropagation-based deep learning libraries.

---

### 1. Context Engineering File (`context_def.md`)

**Target Agent:** Lead Architect Agent

**Purpose:** Establishes the fundamental laws of the environment and the core paradigm shift of the project.

* **Project Name:** 3D-RNG (3D Recursive Neural Graph)
* **Core Paradigm:** We are abandoning sequential layered architectures and global backpropagation. The system is a layerless, isotropic 3D grid of nodes.
* **Execution Rule 1 (No Backprop):** Global gradients are strictly forbidden. The system must use local learning rules based on Traceback Reinforcement (digital pheromones).
* **Execution Rule 2 (No Dense Matrix Multiply for Routing):** Routing is treated as a graph traversal problem, not a matrix multiplication problem. Agents must implement a Guided Non-Backtracking Depth-First Search (DFS).
* **Execution Rule 3 (Shared State):** All nodes share the same core recursive weight matrix for state updates, but maintain individual biases and edge weights.
* **System Analogy:** The network behaves like an ant colony searching a 3D maze for food. Signals drop pheromones on successful paths and avoid paths that dry up.

---

### 2. Intention Engineering File (`intent_def.md`)

**Target Agent:** Product/Goal Alignment Agent

**Purpose:** Defines the success metrics and the "why" behind the architecture so agents can make autonomous optimization decisions.

* **Primary Objective:** Achieve functional task learning (e.g., XOR logic, MNIST classification) using a fraction of the active compute required by a standard Multi-Layer Perceptron (MLP).
* **Optimization Target 1 (Sparse Activation):** Maximize sparsity. A successful inference should only activate the nodes directly on the DFS path (target: <5% network activation per pass).
* **Optimization Target 2 (Emergent Specialization):** The graph must organically form specialized sub-sections based on input types without human-coded layers.
* **Optimization Target 3 (Resilience):** The system must gracefully degrade and self-heal. If 10% of nodes are randomly deleted post-training, the routing algorithm must find alternative paths to restore accuracy within minimal training cycles.

---

### 3. Spec Driven Engineering File (`spec_def.md`)

**Target Agent:** Core Developer Agent / Mathematics Agent

**Purpose:** Provides the strict mathematical formulas, classes, and topological rules the software must execute.

* **Graph Topology:**
  * Initialize an **$X \times Y \times Z$** spatial grid.
  * Nodes connect to up to 6 orthogonal neighbors.
  * Edge routing weights (pheromones) initialized to **$\tau = 1.0$**.
* **Node Computation (Recursive Update):**
  * Given incoming state vector **$h_{t-1}$**:
  * **$h_t = \sigma(W_{shared} \cdot h_{t-1} + b_v)$**
  * Where **$W_{shared}$** is globally shared, **$b_v$** is node-specific, and **$\sigma$** is ReLU or Tanh.
* **Routing Logic (Guided DFS):**
  * Signal selects next neighbor **$v$** from current node **$u$** with probability: **$P(u \rightarrow v) = \frac{\tau_{u,v}}{\sum \tau_{u,k}}$**
  * **Refractory Constraint:** Cannot route to the immediate past node, or any node accessed in the last **$N$** steps of this specific DFS trace.
* **Learning Rule (Traceback Reinforcement):**
  * Upon reaching Output Zone, generate Prediction **$\hat{y}$**. Calculate Reward **$R$** (+1 for correct, -0.5 for incorrect).
  * Unwind DFS stack. For every edge **$(u,v)$** in the path: **$\tau_{u,v} \leftarrow (1 - \rho)\tau_{u,v} + R$** (where **$\rho$** is the evaporation rate, e.g., **$0.05$**).

---

### 4. Harness Engineering File (`harness_def.md`)

**Target Agent:** QA & Testing Agent

**Purpose:** Defines how the system is evaluated, benchmarked, and stress-tested.

* **Environment:** Object-oriented simulation (Python/PyTorch CPU or optimized Graph/C++ backend). Do *not* use standard `torch.nn.Sequential` or `loss.backward()`.
* **Sanity Benchmark:** XOR Gate Simulator.
  * Input: 2 nodes on the left face. Output: 1 node on the right face.
  * Pass Condition: 100% accuracy within 1000 routing epochs.
* **Primary Benchmark:** Flattened MNIST.
  * Input: 784 spatial drop points on the **$X=0$** face.
  * Output: 10 target zones on the **$X=Max$** face.
  * Compare convergence rate and active node count against a standard 3-layer MLP.
* **Ablation Harness:** Post-training "Damage" test. Randomly zero out 10% of the nodes and measure epochs to recover previous accuracy.

---

### Data & Pipeline Specification (For Data/ML Ops Agents)

To train this architecture, data cannot be fed in massive parallel batches like a standard GPU matrix multiplication. It must be injected as sequential, spatial "drops."

#### 1. Necessary Training Data

* **Pre-training (Topology Warm-up):** Random uniform noise vectors. Used to initialize pheromone flow and ensure no dead zones exist in the graph before actual data hits.
* **Task 1 (Logic):** Bitwise datasets (XOR, AND, OR) to test fundamental routing.
* **Task 2 (Spatial):** MNIST handwritten digits, normalized.
* **Task 3 (Sequential):** Simple sine wave prediction or basic character sequences to test the recursive "memory" of the shared weight matrix.

#### 2. Feature Extraction & Preprocessing

* **Spatial Mapping:** Unlike standard neural networks that take a flat 1D vector (**$1 \times 784$**), the data preprocessing pipeline must map specific features to specific physical coordinates on the graph's Input Face.
  * *Example:* For MNIST, pixel **$(0,0)$** always maps to Input Node **$(0, y_0, z_0)$**.
* **Normalization:** Because the signal accumulates recursively (**$h_t = f(h_{t-1})$**), inputs must be strictly normalized between **$-1$** and **$1$** to prevent the signal magnitude from exploding before it reaches the output.

#### 3. Training Pipelines

* **Pre-Training Pipeline (Exploration Phase):**
  * Inject random noise.
  * Force a high "exploration rate" (similar to epsilon-greedy in RL) where signals occasionally ignore the pheromone compass and take random paths.
  * *Goal:* Distribute baseline pheromones evenly so the signal doesn't get trapped in a corner on step 1.
* **Post-Training / Task Pipeline (Exploitation Phase):**
  * Inject task data (e.g., MNIST).
  * Lower the exploration rate. Signals strictly follow the strongest Guided DFS paths.
  * Apply aggressive pheromone evaporation (**$\rho$**) to quickly kill off inefficient pathways and solidify the "specialized sections" (clusters of nodes solving specific sub-problems).
  * Lock the shared recursive weights (**$W_{shared}$**) and only allow pheromones (**$\tau$**) and local biases (**$b_v$**) to update.

---

This is the complete package your agentic AI needs to understand, build, and benchmark the system.

Would you like me to act as the "Core Developer Agent" now and draft the actual Python code for the `Graph` and `Node` objects based on the Spec Driven Engineering file?
