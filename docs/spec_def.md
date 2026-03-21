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