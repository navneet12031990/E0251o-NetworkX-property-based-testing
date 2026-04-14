## Property-Based Testing for NetworkX — Algebraic Connectivity and Fiedler Vector via the Graph Laplacian

**Course:** E0 251o (2026)  
**Author:** Navneet Chandan  
**Algorithm:** Algebraic Connectivity (λ₂) and Fiedler Vector via the Graph Laplacian

## Environment

| Package    | Version  |
|------------|----------|
| Python     | 3.10.13  |
| NetworkX   | 3.4.2    |
| Hypothesis | 6.151.12 |
| pytest     | 9.0.3    |
| NumPy      | 1.26.4   |

## Files

- `test_algebraic_connectivity.py` — main property-based test suite (25 tests)
- `test_bug_probes.py` — bug probe tests investigating API inconsistencies (8 tests)

## Install dependencies

```bash
pip install hypothesis networkx numpy scipy pytest
```

## Run main suite

```bash
pytest test_algebraic_connectivity.py -v
```

## Run bug probes

```bash
pytest test_bug.py -v -s -W always
```
The `-s -W always` flags are required to see the warning box documenting the API inconsistency finding.

## Key findings


### API semantic inconsistency (bug probe)
NetworkX silently replaces edge weights `w` with `|w|` inside `algebraic_connectivity` via `_preprocess_graph`, but `laplacian_matrix` uses raw weights. On the same graph with negative weights:

```python
nx.algebraic_connectivity(G)                    # uses |w| → λ₂ > 0
np.linalg.eigvalsh(nx.laplacian_matrix(G))[1]   # uses raw w → λ₂ < 0
```
Two functions in the same `nx.linalg` namespace, on the same graph object, return results with **opposite signs** from different mathematical objects, with no warning emitted. The only documentation is a single line in the `Notes` section of `algebraic_connectivity`. See `test_bug_probes.py::TestAbsoluteWeightInconsistency::test_inconsistency_is_not_warned` for the full mathematical explanation.

## Test results

```
test_algebraic_connectivity.py: 25 passed
test_bug_probes.py:              8 passed with API inconsistency Warning
```
