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