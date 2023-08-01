# annotated-sfm
Structural Functional Models (SFM) for defining causality.

_Causality is just functions: "x causes y" roughly means y=f(x, ...)._

Paper: Reducing Causality to Functions with Structural Models (https://arxiv.org/abs/2307.07524)

This is a simple toy algorithm that illustrates the philosophical concept
of "defining causality as functions."
It includes algorithms that I didn't fully specify in the paper.

For practical causal inference in machine learning,
please refer to specialized libraries like
[DoWhy](https://github.com/py-why/dowhy)
and [CausalML](https://github.com/uber/causalml).
I personally recommend
[_Elements of Causal Inference: Foundations and Learning Algorithms_](https://mitpress.mit.edu/9780262037310/elements-of-causal-inference/)
for a good introduction book.

This repository contains:
- `sfm/model.py`
  - **SFM**: In a directed acyclic graph, the value of a node is determined
by the values of its parents through a function.
- `sfm/inference.py`
  - **Vanilla forward inference (VFI)**: `w = VFI(M, w_exo)` Given the SFM and the values of all
exogenous nodes (root nodes), compute the values of all nodes.
  - **Contrastive forward inference (CFI)**: `w1 = VFI(M, w0, w1_exo)` Given the SFM,
a reference `{node: value}` assignment over all nodes,
and the new values of exo-nodes, compute the values of all nodes.
This could reduce the number of function evaluations,
since we know from the graph structure that some nodes' values
will remain the same.
- `sfm/partial.py`
  - **Partial VFI**: `w_targets = partial_VFI(M, w_exo, targets)` Same input as VFI,
except we're only interested in the values of some target nodes.
Using graph information, we further reduce the number of function evaluations.
  - **Partial CFI**: `w1_targets = partial_CFI(M, w0, w1_exo, targets)` Same input as CFI,
except we're only interested in the values of some target nodes.
This is the setting with the fewest function evaluations.
- `generate.py`
  - Generate random directed acyclic graphs
by assigning random topological orders over undirected Erdos-Renyi graphs.
  - Generate random SFMs with random linear functions/congruences:
  - `f(x1, x2, ...) = a1 * x1 + a2 * x2 + ...`
  - `f(x1, x2, ...) = a1 * x1 + a2 * x2 + ... (mod m)`
When all variables/coefficients have integer values,
this helps create non-injective functions for testing CFI.
