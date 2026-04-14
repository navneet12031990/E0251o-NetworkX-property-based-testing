"""
================================================================================
E0 251o (2026) — Bug Probe Tests for NetworkX Algebraic Connectivity
Team Members : [NAVNEET CHANDAN]
SR NUMBER    : 13-19-02-19-52-25-1-26109 
Purpose      : Investigate potential bugs / API inconsistencies in
               nx.algebraic_connectivity and nx.fiedler_vector
================================================================================

BACKGROUND — WHY THESE TESTS EXIST
─────────────────────────────────────────────────────────────────────────────
The main test suite (test_algebraic_connectivity.py) validates known
mathematical properties using positive-weight graphs.  This companion
script probes two specific behaviours that are underspecified in the
NetworkX documentation and may constitute genuine bugs or misleading
API design:

  BUG PROBE 1 — Absolute weight inconsistency
  ─────────────────────────────────────────────
  nx.algebraic_connectivity internally calls _preprocess_graph which
  replaces every edge weight w with |w|.  However nx.laplacian_matrix
  uses raw weights without taking absolute values.  This means:

      nx.algebraic_connectivity(G)   uses |w|  → always PSD
      nx.laplacian_matrix(G)         uses w    → may NOT be PSD

  A user who manually computes L = nx.laplacian_matrix(G) and then
  extracts λ₂ via numpy will get a DIFFERENT answer from
  nx.algebraic_connectivity(G) on the same graph with negative weights.
  This silent inconsistency is undocumented and counterintuitive.

  BUG PROBE 2 — Normalised Laplacian scale invariance
  ─────────────────────────────────────────────────────
  The normalised Laplacian is L_norm = D^{-1/2} L D^{-1/2}.
  Its eigenvalues lie in [0, 2] regardless of edge weights — scaling
  all weights by α leaves L_norm unchanged because the degree matrix
  D scales by α in both the numerator (L) and denominator (D^{-1/2}),
  cancelling out.  Therefore algebraic_connectivity(G, normalized=True)
  should be INVARIANT to uniform weight scaling.

  By contrast, algebraic_connectivity(G, normalized=False) should scale
  linearly with α.  This script verifies both behaviours are correctly
  implemented and that NetworkX's documentation accurately reflects the
  distinction.

REFERENCE
  NX source: networkx/linalg/algebraicconnectivity.py
  Documented note: "Edge weights are interpreted by their absolute values."
================================================================================
"""

import math
import numpy as np
import networkx as nx
import pytest

from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

# ── Tolerances ────────────────────────────────────────────────────────────────
ATOL = 1e-6
RTOL = 1e-5


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

def connected_graph(n_min: int = 3, n_max: int = 8):
    """
    Connected graph strategy using Bernoulli edge inclusion.
    n_max capped at 8 — large enough to exercise all spectral properties,
    small enough that st.permutations and the eigensolver never hang.
    """
    @st.composite
    def _strategy(draw):
        n = draw(st.integers(min_value=n_min, max_value=n_max))
        G = nx.Graph()
        G.add_nodes_from(range(n))
        node_order = draw(st.permutations(range(n)))
        for i in range(n - 1):
            G.add_edge(node_order[i], node_order[i + 1])
        for u in range(n):
            for v in range(u + 1, n):
                if not G.has_edge(u, v):
                    if draw(st.booleans()):
                        G.add_edge(u, v)
        return G
    return _strategy()


# ══════════════════════════════════════════════════════════════════════════════
# BUG PROBE 1 — Absolute value of weights: inconsistency between
#               algebraic_connectivity and laplacian_matrix
# ══════════════════════════════════════════════════════════════════════════════

class TestAbsoluteWeightInconsistency:
    """
    NetworkX takes |w| in algebraic_connectivity but uses raw w in
    laplacian_matrix.  These tests expose and document that inconsistency.
    """

    def test_negative_weights_give_same_connectivity_as_positive(self):
        """
        Bug probe: algebraic_connectivity(G with w=-k) ==
                   algebraic_connectivity(G with w=+k)

        because _preprocess_graph replaces every weight with |w|.

        This is documented behaviour ("Edge weights are interpreted by
        their absolute values") but is counterintuitive — a graph with
        negative weights (which mathematically has a non-PSD Laplacian)
        returns the same λ₂ as its positive-weight counterpart.

        Verification: if this test PASSES, the |w| preprocessing is
        confirmed.  If it FAILS, NetworkX changed its weight handling.
        """
        G_pos = nx.Graph()
        G_pos.add_edge(0, 1, weight=3.0)
        G_pos.add_edge(1, 2, weight=2.0)
        G_pos.add_edge(0, 2, weight=1.0)

        G_neg = nx.Graph()
        G_neg.add_edge(0, 1, weight=-3.0)
        G_neg.add_edge(1, 2, weight=-2.0)
        G_neg.add_edge(0, 2, weight=-1.0)

        lam2_pos = nx.algebraic_connectivity(G_pos)
        lam2_neg = nx.algebraic_connectivity(G_neg)

        assert abs(lam2_pos - lam2_neg) < ATOL * 100, (
            f"Expected |w| symmetry: lam2(+w)={lam2_pos:.6f} "
            f"!= lam2(-w)={lam2_neg:.6f}"
        )

    def test_laplacian_matrix_does_not_take_absolute_value(self):
        """
        Bug probe: nx.laplacian_matrix uses RAW weights, not |w|.

        This directly contradicts the behaviour of algebraic_connectivity
        and creates a silent API inconsistency.

        With all-negative weights:
          - algebraic_connectivity → uses |w| → matrix IS PSD → λ₂ > 0
          - laplacian_matrix       → uses w   → matrix NOT PSD → λ₂ < 0

        A user who computes L = nx.laplacian_matrix(G) and then extracts
        λ₂ via numpy will get a NEGATIVE value for a "connected" graph,
        completely contradicting what algebraic_connectivity returns.

        This is the core inconsistency: two NetworkX functions on the
        same graph object return results based on different interpretations
        of the same edge weight attribute, with no warning to the user.

        Evidence of inconsistency:
            lam2_from_nx_function  > 0   (uses |w|)
            lam2_from_nx_matrix    < 0   (uses raw w)
            difference             >> numerical tolerance
        """
        G = nx.Graph()
        G.add_edge(0, 1, weight=-3.0)
        G.add_edge(1, 2, weight=-2.0)
        G.add_edge(0, 2, weight=-1.0)

        # algebraic_connectivity silently takes |w|
        lam2_api = nx.algebraic_connectivity(G)

        # laplacian_matrix uses raw negative weights
        L_raw = nx.laplacian_matrix(G).toarray().astype(float)
        eigenvalues = np.linalg.eigvalsh(L_raw)
        lam2_matrix = eigenvalues[1]   # second smallest

        # The inconsistency: same graph, same function namespace, opposite signs
        assert lam2_api > 0, (
            f"algebraic_connectivity should be positive (uses |w|): {lam2_api}"
        )
        assert lam2_matrix < 0, (
            f"laplacian_matrix with negative weights gives non-PSD matrix, "
            f"so λ₂ should be negative: {lam2_matrix:.6f}"
        )

        # The gap between the two is not numerical noise — it is the
        # entire difference between |w| and w interpretations
        gap = abs(lam2_api - lam2_matrix)
        assert gap > 0.1, (
            f"Gap between |w| and raw-w interpretations should be large, "
            f"got {gap:.6f}"
        )

    @given(connected_graph(n_min=3, n_max=8),
           st.floats(min_value=0.5, max_value=5.0,
                     allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_sign_flip_of_all_weights_preserves_algebraic_connectivity(
            self, G, base_weight):
        """
        Property (Metamorphic / Bug probe): Flipping the sign of every
        edge weight leaves algebraic_connectivity unchanged because
        _preprocess_graph takes |w| before computing anything.

        This is a property-based version of the inconsistency test:
        for any connected graph and any positive weight assignment,
        negating all weights must produce the exact same λ₂.

        Mathematical note: this property does NOT hold for the raw
        mathematical Laplacian — it only holds because of NetworkX's
        specific implementation choice to take absolute values.

        If this test fails, NetworkX changed its weight preprocessing
        and the documented behaviour "Edge weights are interpreted by
        their absolute values" is no longer accurate.
        """
        # Assign uniform positive weights
        for u, v in G.edges():
            G[u][v]['weight'] = base_weight

        lam2_positive = nx.algebraic_connectivity(G)

        # Negate all weights
        G_neg = G.copy()
        for u, v in G_neg.edges():
            G_neg[u][v]['weight'] = -base_weight

        lam2_negative = nx.algebraic_connectivity(G_neg)

        assert abs(lam2_positive - lam2_negative) < ATOL * 100, (
            f"Sign flip changed λ₂: {lam2_positive:.6f} → {lam2_negative:.6f}\n"
            f"This means NetworkX stopped taking |w| in preprocessing."
        )

    def test_inconsistency_is_not_warned(self):
        """
        Bug probe: NetworkX raises NO warning when a user passes negative
        edge weights to algebraic_connectivity, even though the result
        silently diverges from the mathematically expected behaviour of
        the standard combinatorial Laplacian.

        MATHEMATICAL EXPLANATION OF THE INCONSISTENCY
        ───────────────────────────────────────────────
        The combinatorial Laplacian of a weighted graph G = (V, E, w) is:
            L = D − A
        where A[u,v] = w_{uv} and D = diag(Σ_j w_{ij}).

        PSD guarantee:  L is positive semi-definite iff all weights w_{uv} ≥ 0.
        Proof via quadratic form:
            x^T L x = Σ_{(u,v)∈E} w_{uv} (x_u − x_v)²
        If any w_{uv} < 0, the sum can be negative → L is NOT PSD → λ₂ < 0.

        NetworkX's algebraic_connectivity internally calls _preprocess_graph
        which silently replaces every weight w with |w| before constructing L.
        This means:

            nx.algebraic_connectivity(G)   ← uses |w|, L is always PSD, λ₂ > 0
            nx.laplacian_matrix(G)         ← uses raw w, L may NOT be PSD

        On the SAME graph with negative weights, these two functions return
        contradictory spectral information from the same graph object,
        with no warning emitted by NetworkX to alert the user.

        WHAT A CORRECT API WOULD DO
        ────────────────────────────
        Option A — raise ValueError:
            if any(w < 0 for _,_,w in G.edges.data('weight', default=1)):
                raise ValueError("Negative edge weights are not supported.")

        Option B — emit UserWarning (least disruptive fix):
            warnings.warn(
                "Negative edge weights detected. Weights will be replaced "
                "by their absolute values before computing the Laplacian. "
                "This means the result may differ from np.linalg.eigvalsh("
                "nx.laplacian_matrix(G)) on the same graph.",
                UserWarning
            )

        Option C — document prominently in the function signature (not just
        buried in the Notes section as "Edge weights are interpreted by their
        absolute values.").

        This test PASSES when NetworkX emits NO warning — which is the current
        (broken) behaviour.  We issue our OWN warning below to document the
        inconsistency for anyone reading the test output, since NetworkX itself
        stays silent.

        The test will FAIL if NetworkX ever adds a warning — which would
        actually be the correct fix.  A failure here means improvement.
        """
        import warnings

        G = nx.Graph()
        G.add_edge(0, 1, weight=-5.0)
        G.add_edge(1, 2, weight=-3.0)
        G.add_edge(0, 2, weight=-1.0)

        # ── Step 1: capture whatever NetworkX does ────────────────────────
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lam2_api = nx.algebraic_connectivity(G)

        # ── Step 2: compute λ₂ the way a mathematician would ──────────────
        L_raw = nx.laplacian_matrix(G).toarray().astype(float)
        lam2_matrix = np.linalg.eigvalsh(L_raw)[1]

        # ── Step 3: emit OUR warning documenting the inconsistency ─────────
        warnings.warn(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║         NETWORKX API INCONSISTENCY — BUG PROBE FINDING       ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║                                                              ║\n"
            "║  Graph G has ALL NEGATIVE edge weights.                      ║\n"
            "║                                                              ║\n"
            "║  MATHEMATICAL EXPECTATION                                    ║\n"
            "║  The combinatorial Laplacian L = D − A with negative w       ║\n"
            "║  is NOT positive semi-definite.  Its quadratic form:         ║\n"
            "║      x^T L x = Σ w_{uv}(x_u − x_v)²                          ║\n"
            "║  can be negative when w_{uv} < 0.  Hence λ₂ < 0 is           ║\n"
            "║  mathematically valid and expected for such a matrix.        ║\n"
            "║                                                              ║\n"
            "║  WHAT NETWORKX ACTUALLY DOES                                 ║\n"
            "║  nx.algebraic_connectivity calls _preprocess_graph which     ║\n"
            "║  silently replaces every weight w with |w| before building L.║\n"
            "║  nx.laplacian_matrix does NOT do this — it uses raw weights. ║\n"
            "║                                                              ║\n"
            "║  OBSERVED DIVERGENCE ON THIS GRAPH                           ║\n"
           f"║    nx.algebraic_connectivity(G)            λ₂ = {lam2_api:+.6f}    ║\n"
           f"║    eigvalsh(nx.laplacian_matrix(G))           λ₂ = {lam2_matrix:+.6f} ║\n"
           f"║    gap = {abs(lam2_api - lam2_matrix):.6f}  "
            "(not numerical noise — entirely due to |w|)                    ║\n"
            "║                                                              ║\n"
            "║  ROOT CAUSE                                                  ║\n"
            "║  Two functions in the same nx.linalg namespace silently use  ║\n"
            "║  DIFFERENT interpretations of the same 'weight' edge attr.   ║\n"
            "║  No UserWarning, ValueError, or prominent documentation      ║\n"
            "║  alerts the user to this divergence.                         ║\n"
            "║                                                              ║\n"
            "║  SUGGESTED FIX                                               ║\n"
            "║  NetworkX should emit a UserWarning when negative weights    ║\n"
            "║  are passed to algebraic_connectivity / fiedler_vector,      ║\n"
            "║  stating that |w| will be used and results will differ from  ║\n"
            "║  nx.laplacian_matrix on the same graph.                      ║\n"
            "╚══════════════════════════════════════════════════════════════╝",
            UserWarning,
            stacklevel=2,
        )

        # ── Step 4: assert NetworkX itself stayed silent ───────────────────
        nx_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(nx_warnings) == 0, (
            "NetworkX now warns about negative weights — this is the correct fix!\n"
            "Update this test to assert the warning IS present and well-worded.\n"
            f"Warning emitted: {[str(w.message) for w in nx_warnings]}"
        )

        # ── Step 5: assert the inconsistency values ────────────────────────
        assert lam2_api > 0, (
            f"algebraic_connectivity should return λ₂ > 0 (uses |w|), "
            f"got {lam2_api:.6f}"
        )
        assert lam2_matrix < 0, (
            f"eigvalsh(laplacian_matrix) should return λ₂ < 0 for negative-weight "
            f"graph (uses raw w), got {lam2_matrix:.6f}"
        )
        assert abs(lam2_api - lam2_matrix) > 0.1, (
            f"Gap between API and matrix λ₂ should be large (not numerical noise), "
            f"got gap={abs(lam2_api - lam2_matrix):.6f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# BUG PROBE 2 — Normalised vs unnormalised Laplacian: scale invariance
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalisedLaplacianScaleInvariance:
    """
    The normalised Laplacian L_norm = D^{-1/2} L D^{-1/2} is scale-invariant
    under uniform weight scaling.  The unnormalised Laplacian is not.
    These tests verify both properties are correctly implemented in NetworkX.
    """

    @given(connected_graph(n_min=3, n_max=5),
           st.floats(min_value=0.1, max_value=10.0,
                     allow_nan=False, allow_infinity=False))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_normalised_connectivity_is_scale_invariant(self, G, alpha):
        """
        Property (Invariant): algebraic_connectivity(αG, normalized=True)
                            == algebraic_connectivity(G,  normalized=True)

        Mathematical basis:
            The normalised Laplacian is defined as:
                L_norm = D^{-1/2} (D - A) D^{-1/2}
            Scaling all weights by α gives:
                D → αD,   A → αA,   D - A → α(D - A)
                L_norm(αG) = (αD)^{-1/2} · α(D-A) · (αD)^{-1/2}
                           = α^{-1/2} D^{-1/2} · α(D-A) · α^{-1/2} D^{-1/2}
                           = D^{-1/2} (D-A) D^{-1/2}
                           = L_norm(G)
            The α cancels exactly — normalised λ₂ is scale-invariant.

        Significance:
            This means that for normalised spectral clustering, the
            absolute scale of edge weights is irrelevant — only the
            *ratios* between weights matter.  This is a key property
            that makes normalised spectral methods robust to weight
            magnitude differences across datasets.

        If this test FAILS it would indicate a bug in how NetworkX
        applies the D^{-1/2} normalisation after weight preprocessing.
        """
        lam2_orig = nx.algebraic_connectivity(G, normalized=True)

        G_scaled = G.copy()
        for u, v in G_scaled.edges():
            G_scaled[u][v]['weight'] = (
                G_scaled[u][v].get('weight', 1.0) * alpha
            )

        lam2_scaled = nx.algebraic_connectivity(G_scaled, normalized=True)

        assert abs(lam2_orig - lam2_scaled) < ATOL * 100, (
            f"Normalised λ₂ changed under weight scaling by α={alpha:.3f}: "
            f"{lam2_orig:.6f} → {lam2_scaled:.6f}\n"
            f"Normalised Laplacian should be scale-invariant."
        )

    @given(connected_graph(n_min=3, n_max=5),
           st.floats(min_value=0.5, max_value=5.0,
                     allow_nan=False, allow_infinity=False))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_unnormalised_connectivity_scales_linearly(self, G, alpha):
        """
        Property (Metamorphic): algebraic_connectivity(αG, normalized=False)
                              == α · algebraic_connectivity(G, normalized=False)

        Mathematical basis:
            The unnormalised Laplacian L = D - A scales linearly:
                L(αG) = αD - αA = α(D - A) = αL(G)
            So eigenvalues scale by α:  λ₂(αG) = α · λ₂(G).

        This contrasts with the normalised case (scale-invariant) and
        confirms the two modes behave as mathematically expected.

        Cross-check: combining this test with the normalised test above
        proves that NetworkX correctly implements both Laplacian variants.
        A failure here would mean the unnormalised eigensolver has a
        weight handling bug, while a failure in the normalised test
        would mean the D^{-1/2} normalisation is incorrectly applied.
        """
        # Add explicit unit weights so scaling is unambiguous
        for u, v in G.edges():
            G[u][v]['weight'] = G[u][v].get('weight', 1.0)

        lam2_orig = nx.algebraic_connectivity(G, normalized=False)

        G_scaled = G.copy()
        for u, v in G_scaled.edges():
            G_scaled[u][v]['weight'] = G[u][v]['weight'] * alpha

        lam2_scaled = nx.algebraic_connectivity(G_scaled, normalized=False)
        expected = alpha * lam2_orig

        assert abs(lam2_scaled - expected) < ATOL * 100 + RTOL * abs(expected), (
            f"Unnormalised λ₂ does not scale linearly with α={alpha:.3f}: "
            f"got {lam2_scaled:.6f}, expected α·λ₂={expected:.6f}"
        )

    @given(connected_graph(n_min=3, n_max=5))
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_normalised_eigenvalue_bounded_in_0_2(self, G):
        """
        Property (Invariant): 0 < λ₂(G, normalized=True) ≤ 2
        for any connected graph.

        Mathematical basis:
            All eigenvalues of the normalised Laplacian lie in [0, 2]
            (Chung 1997, Spectral Graph Theory, Lemma 1.7).
            The lower bound 0 corresponds to the trivial eigenvector,
            and 2 is achieved only by bipartite graphs.
            For connected graphs λ₂ > 0 strictly.

        This is an additional invariant specific to the NORMALISED case
        that has no analogue for the unnormalised Laplacian (where λ₂
        can be arbitrarily large, e.g. λ₂(Kₙ) = n).

        Failure diagnosis:
            λ₂ > 2 would indicate the D^{-1/2} normalisation was not
            applied correctly — the raw Laplacian was returned instead.
            λ₂ ≤ 0 would mean a connected graph was treated as disconnected.
        """
        lam2 = nx.algebraic_connectivity(G, normalized=True)

        assert lam2 > -ATOL, (
            f"Normalised λ₂ should be > 0 for connected graph, got {lam2:.6f}"
        )
        assert lam2 <= 2.0 + ATOL, (
            f"Normalised λ₂ should be ≤ 2 (Chung 1997), got {lam2:.6f}\n"
            f"This would indicate the normalisation was not applied."
        )

    def test_normalised_vs_unnormalised_differ_on_irregular_graph(self):
        """
        Sanity check: normalised and unnormalised λ₂ give DIFFERENT values
        on an irregular graph (one where not all degrees are equal).

        This confirms the two modes are genuinely computing different things
        and not accidentally returning the same result.

        On a regular graph (all degrees equal d), the normalised Laplacian
        equals (1/d)·L, so normalised λ₂ = λ₂/d.  But on irregular graphs
        the two diverge in a non-trivial way.

        We use a path graph P₄ which is irregular (endpoints have degree 1,
        internal nodes have degree 2).
        """
        G = nx.path_graph(6)   # irregular: degrees are 1,2,2,2,2,1

        lam2_unnorm = nx.algebraic_connectivity(G, normalized=False)
        lam2_norm   = nx.algebraic_connectivity(G, normalized=True)

        # They should differ on an irregular graph
        assert abs(lam2_unnorm - lam2_norm) > 1e-3, (
            f"Normalised and unnormalised λ₂ are suspiciously equal on an "
            f"irregular graph: unnorm={lam2_unnorm:.6f}, norm={lam2_norm:.6f}"
        )

        # And normalised must be in [0,2]
        assert 0 < lam2_norm <= 2.0 + ATOL

        # Unnormalised can be > 2 (not bounded by 2)
        # For P₆: λ₂ = 2(1 - cos(π/6)) ≈ 0.268 — still < 2 here,
        # but for complete graphs it equals n which can be >> 2
        expected_unnorm = 2.0 * (1.0 - math.cos(math.pi / 6))
        assert abs(lam2_unnorm - expected_unnorm) < ATOL * 100, (
            f"P₆ unnormalised λ₂ should be {expected_unnorm:.6f}, "
            f"got {lam2_unnorm:.6f}"
        )
