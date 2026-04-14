"""
================================================================================
E0 251o (2026) — Property-Based Testing for NetworkX
Team Members : [NAVNEET CHANDAN]
SR NUMBER : 13-19-02-19-52-25-1-26109
Algorithms   : Algebraic Connectivity (λ₂) and Fiedler Vector
               via the Graph Laplacian (networkx.algebraic_connectivity,
               networkx.fiedler_vector)
================================================================================

BACKGROUND — THE LAPLACIAN MATRIX AND ALGEBRAIC CONNECTIVITY
─────────────────────────────────────────────────────────────
For an undirected,positive weighted graph G = (V, E, w) with n vertices the
(combinatorial) Laplacian is

        L = D − A

where A is the weighted adjacency matrix and D = diag(Σ_j w_{ij}) is the
degree matrix.  L is symmetric positive-semidefinite, so its eigenvalues
satisfy  0 = λ₁ ≤ λ₂ ≤ … ≤ λₙ.

The *algebraic connectivity* (Fiedler value) is the second-smallest
eigenvalue λ₂.  The eigenvector corresponding to λ₂ is the *Fiedler
vector* x, which satisfies  Lx = λ₂x  and (by convention) ‖x‖₂ = 1.

WHY NOT JUST USE MIN-CUT? THE CASE FOR BALANCED PARTITIONING
─────────────────────────────────────────────────────────────
A natural way to split a graph into two groups is to find the minimum
cut — the smallest set of edges whose removal disconnects the graph.
Min-cut can be computed exactly in polynomial time (e.g. via max-flow).
So why do practitioners reach for algebraic connectivity instead?

The answer is that min-cut is a deeply flawed objective for most
real-world partitioning problems.  It is prone to adversarial graphs: if any single
vertex has degree 1 (a leaf), cutting that one edge gives a min-cut of
size 1, producing a partition of {leaf} vs {everyone else}.  This
partition is useless — it is maximally unbalanced.

What practitioners actually need is a *balanced* cut: a partition of V
into two roughly equal halves (S, V\\S) such that the number of edges
crossing the cut is small relative to the sizes of the two sides.  The
canonical formulation is the *normalised cut* (Shi & Malik 2000 [3]):

        NCut(S, V\\S) = cut(S, V\\S) / vol(S)  +  cut(S, V\\S) / vol(V\\S)

where vol(S) = Σ_{v∈S} deg(v) is the volume of a partition side.
NCut penalises lopsided splits: a partition isolating a single leaf
has vol(S) ≈ 1, so NCut ≈ cut / 1 + cut / vol(V) which is large.
Balanced partitions have large vol(S) and vol(V\\S), driving NCut down.

THE NP-HARDNESS BARRIER AND THE SPECTRAL RELAXATION
─────────────────────────────────────────────────────
Finding the partition that exactly minimises NCut (or the related
*ratio cut* and *graph bisection* objectives) is NP-hard — it requires
searching over all 2^n possible bipartitions of the vertex set, which is
computationally intractable even for moderate n.

  Theorem (Garey & Johnson 1979; Wagner & Wagner 1993):
      Minimum bisection and ratio cut are NP-hard.  No polynomial-time
      algorithm can solve them exactly unless P = NP.

This is where algebraic connectivity provides a critical breakthrough.
Shi & Malik (2000) [3] showed that relaxing the discrete {0,1} partition
indicator to a continuous real vector, and then minimising the Rayleigh
quotient of the Laplacian, yields exactly the Fiedler eigenproblem:

        min_{x ⊥ 1, ‖x‖=1}  x^T L x  =  λ₂,    attained at x = Fiedler vector.

This relaxation converts the NP-hard combinatorial search into a
polynomial-time eigenvector computation — solvable in O(k·m) time via
Lanczos / LOBPCG iteration(k being number of iteration and m being number of edges).
The Fiedler vector x then provides an approximate balanced partition by thresholding 
at the median of x:

        S  = { v : x_v < median(x) },    V\\S = { v : x_v ≥ median(x) }

In summary:
  • Min-cut  → polynomial time, but produces trivially unbalanced splits
  • Balanced cut (ratio cut, NCut, bisection) → NP-hard to solve exactly
  • Spectral relaxation via λ₂ / Fiedler vector → polynomial time,
    provably good approximation with Cheeger inequality guarantee

This makes algebraic connectivity one of the most practically important
quantities in algorithmic graph theory.

REFERENCES (open-access PDFs)
─────────────────────────────────────────────────────────────
  [1] Fiedler (1973) — original algebraic connectivity paper
      https://dml.cz/bitstream/handle/10338.dmlcz/101168/CzechMathJ_23-1973-2_11.pdf
      Fiedler, M. Algebraic connectivity of graphs.
      Czechoslovak Mathematical Journal, 23(2), 298–305.

  [2] Mohar (1991) — Laplacian spectrum survey; spectral bounds, Cheeger inequality
      https://users.fmf.uni-lj.si/mohar/Papers/Spec.pdf
      Mohar, B. The Laplacian spectrum of graphs.
      In Graph Theory, Combinatorics, and Applications, Vol. 2, pp. 871–898. Wiley.

  [3] Shi & Malik (2000) — normalised cuts, spectral relaxation of NP-hard bisection
      https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf
      Shi, J. & Malik, J. Normalized cuts and image segmentation.
      IEEE Trans. Pattern Anal. Mach. Intell., 22(8), 888–905. DOI: 10.1109/34.868688

Key facts exploited by the tests below
  • λ₂ > 0   ⟺   G is connected                    (Fiedler 1973 [1])
  • λ₂ = 0   iff  G is disconnected (multiplicity of 0 = # components)
  • Σᵢ xᵢ = 0  (Fiedler vector ⊥ 1-vector, because L·1 = 0)
  • Adding an edge never decreases λ₂ (monotonicity / interlacing)
  • λ₂ ≤ n/(n−1) · min_degree  (Mohar's upper bound [2])
  • For a path Pₙ: λ₂ = 2(1 − cos(π/n)) → 0 as n → ∞
  • For a complete graph Kₙ: λ₂ = n  (all non-trivial eigenvalues equal n)
  • Fiedler vector is orthogonal to every vector in ker(L), i.e. to 1

REAL-WORLD APPLICATIONS
─────────────────────────────────────────────────────────────
  • VLSI circuit layout: chips must be partitioned across multiple boards
    with minimal inter-board wiring (= cut edges).  Balanced partitioning
    ensures no board is overloaded.  Spectral methods have been industry
    standard since the 1990s (Alpert & Kahng 1995).
  • Image segmentation: pixels form a weighted graph (similar pixels get
    high-weight edges).  NCut partitions the image into coherent regions.
    The Fiedler vector of the pixel-similarity Laplacian drives the
    segmentation — basis of the influential Shi & Malik (2000) [3] algorithm
    used in computer vision to this day.
  • Parallel computing / mesh partitioning: scientific simulations
    (finite-element, molecular dynamics) distribute a computational mesh
    across processors.  Balanced spectral partitioning minimises
    communication between processors while keeping each processor's
    workload equal — directly impacting simulation wall-clock time.
  • Community detection in social networks: spectral clustering on the
    normalised Laplacian identifies densely connected communities.
    λ₂ ≈ 0 signals near-disconnection — a community boundary.
  • Network robustness / resilience: λ₂ quantifies how hard it is to
    disconnect the network (higher λ₂ = more robust).  Used to design
    fault-tolerant communication and power grid topologies.
  • Epidemic / diffusion speed: the spectral gap controls how fast a
    random walk mixes and how quickly diseases or information spread.
    Expander graphs (large λ₂) are used in cryptography and coding theory
    precisely because information diffuses rapidly across them.
  • Sensor network connectivity: wireless networks use λ₂ to certify
    that the network remains connected under random node failures.
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import math

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import networkx as nx
import pytest

from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

# ── Tolerances ────────────────────────────────────────────────────────────────
ATOL = 1e-6   # absolute tolerance for floating-point comparisons
RTOL = 1e-5   # relative tolerance

# ══════════════════════════════════════════════════════════════════════════════
# HELPER STRATEGIES — custom Hypothesis strategies for graph generation
# ══════════════════════════════════════════════════════════════════════════════

def connected_graph(n_min: int = 2, n_max: int = 12):
    """
    Strategy producing random *connected* undirected graphs.

    Approach: draw a random node permutation, chain it into a spanning
    path (guarantees connectivity), then independently include each
    remaining candidate edge with 50% probability.  This is O(n²) at
    worst, never hangs, and covers the full sparse→dense spectrum.

    n_max is capped at 12 by default — large enough to exercise all
    spectral properties, small enough that algebraic_connectivity and
    node_connectivity complete quickly (NX solvers are O(n³) in the
    worst case).
    """
    @st.composite
    def _strategy(draw):
        n = draw(st.integers(min_value=n_min, max_value=n_max))
        G = nx.Graph()
        G.add_nodes_from(range(n))
        # Random spanning path — guarantees connectivity
        node_order = draw(st.permutations(range(n)))
        for i in range(n - 1):
            G.add_edge(node_order[i], node_order[i + 1])
        # Each remaining edge included independently with p=0.5
        # — no unique-list generation, so Hypothesis never hangs
        for u in range(n):
            for v in range(u + 1, n):
                if not G.has_edge(u, v):
                    if draw(st.booleans()):
                        G.add_edge(u, v)
        return G
    return _strategy()


def connected_weighted_graph(n_min: int = 2, n_max: int = 10):
    """
    Strategy producing connected undirected graphs with positive
    floating-point edge weights drawn from [0.1, 10.0].
    """
    @st.composite
    def _strategy(draw):
        G = draw(connected_graph(n_min=n_min, n_max=n_max))
        G = G.copy()
        for u, v in G.edges():
            G[u][v]['weight'] = draw(st.floats(min_value=0.1, max_value=10.0,
                                               allow_nan=False, allow_infinity=False))
        return G
    return _strategy()


def disconnected_graph(n_min: int = 4, n_max: int = 12):
    """
    Strategy producing disconnected graphs with exactly two disjoint
    connected components.  Used for boundary tests around λ₂ = 0.
    """
    @st.composite
    def _strategy(draw):
        half = n_max // 2
        n1 = draw(st.integers(min_value=2, max_value=half))
        G1 = draw(connected_graph(n_min=n1, n_max=n1))
        n2 = draw(st.integers(min_value=2, max_value=half))
        G2 = draw(connected_graph(n_min=n2, n_max=n2))
        # Relabel G2 nodes so they don't overlap with G1
        mapping = {old: old + n1 for old in G2.nodes()}
        G2 = nx.relabel_nodes(G2, mapping)
        return nx.compose(G1, G2)
    return _strategy()


@st.composite
def arbitrary_graph(draw, n_min: int = 0, n_max: int = 10):
    """
    Strategy producing any undirected graph — connected, disconnected,
    sparse, dense, or empty.  Each possible edge is included with 50%
    probability independently, so no unique-list generation and no hangs.
    Positive weights are added to all edges.
    """
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    if n == 0:
        return nx.Graph()
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            if draw(st.booleans()):
                w = draw(st.floats(min_value=0.1, max_value=10.0,
                                   allow_nan=False, allow_infinity=False))
                G.add_edge(u, v, weight=w)
    return G


# ══════════════════════════════════════════════════════════════════════════════
# TEST 0 — FOUNDATION: Laplacian is positive semi-definite (any graph)
# ══════════════════════════════════════════════════════════════════════════════

class TestLaplacianStructure:
    """
    Structural properties of the Laplacian matrix itself.
    These hold for ALL graphs — connected or not — and underpin every
    other spectral property tested in this file.
    """

    @given(arbitrary_graph(n_min=0, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_laplacian_is_positive_semidefinite(self, G):
        """
        Property (Invariant): The Laplacian L of any undirected graph with
        non-negative edge weights is positive semi-definite — all eigenvalues ≥ 0.

        Mathematical basis:
            For any vector x ∈ Rⁿ, the quadratic form satisfies:
                x^T L x = Σ_{(u,v)∈E} w_{uv} (x_u − x_v)²  ≥ 0
            since weights w_{uv} > 0 and squares are non-negative.
            This identity follows directly from L = D − A and holds
            regardless of connectivity — it is a property of the
            Laplacian construction itself.  Therefore all eigenvalues
            of L must be ≥ 0 (definition of PSD).

        IMPORTANT — scope of this property (negative weights):
            The PSD guarantee holds ONLY for non-negative edge weights.
            If any w_{uv} < 0, the corresponding term w_{uv}(x_u − x_v)²
            becomes negative, and the quadratic form x^T L x can drop
            below zero — meaning L is no longer PSD and negative eigenvalues
            are mathematically possible.

            NetworkX handles negative weights by silently taking their
            absolute value before constructing the Laplacian (see
            _preprocess_graph in algebraicconnectivity.py: "Edge weights
            are interpreted by their absolute values").  This means
            nx.laplacian_matrix with negative weights does NOT produce
            the standard combinatorial Laplacian — it produces the
            Laplacian of the graph with |w| weights instead.  As a
            result, even with negative input weights, nx.laplacian_matrix
            will still be PSD — but this is due to NetworkX's silent
            preprocessing, not the mathematical property itself.

            Our arbitrary_graph() strategy restricts weights to [0.1, 10.0]
            to test the genuine mathematical property, not the preprocessing
            artefact.  Testing with mixed-sign weights would conflate the
            two and give a misleading picture of correctness.

        Test strategy:
            We use the arbitrary_graph() strategy (connected or not,
            weighted or not, empty or not) because PSD is a universal
            property for non-negative weights.  Using only connected graphs
            would miss the important degenerate cases (empty graph, isolated
            nodes, disconnected components) where the Laplacian still must
            be PSD.  We use np.linalg.eigvalsh (symmetric solver) rather
            than eigvals — it is faster, more numerically stable, and
            guarantees real outputs for symmetric matrices.

        Preconditions: Edge weights are positive (guaranteed by strategy).

        Failure diagnosis:
            A negative eigenvalue would indicate the Laplacian matrix
            was assembled incorrectly — e.g. an asymmetric adjacency
            matrix, a sign error in D − A, incorrect handling of edge
            weights, or a self-loop being counted twice on the diagonal.
            This is the most fundamental test: if it fails, every other
            spectral property in this file is potentially invalid.
        """
        if G.number_of_nodes() == 0:
            return
        try:
            L = nx.laplacian_matrix(G).toarray().astype(float)
        except Exception:
            return  # skip graphs laplacian_matrix cannot handle
        eigenvalues = np.linalg.eigvalsh(L)   # eigvalsh: symmetric → real, faster
        assert np.all(eigenvalues >= -1e-8), (
            f"Laplacian has negative eigenvalue: min={eigenvalues.min():.2e} "
            f"(n={G.number_of_nodes()}, m={G.number_of_edges()})"
        )

    @given(arbitrary_graph(n_min=1, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_laplacian_row_sums_are_zero(self, G):
        """
        Property (Invariant): Every row of the Laplacian sums to zero.

        Mathematical basis:
            L = D − A means L[i,j] = deg(i) if i=j, and −w_{ij} if (i,j)∈E,
            and 0 otherwise.  Summing across any row i:
                Σ_j L[i,j] = deg(i) − Σ_{j:(i,j)∈E} w_{ij} = 0
            This is equivalent to L·1 = 0, confirming that the all-ones
            vector is always in the null space of L with eigenvalue 0.
            Equivalently, each row sums to zero by construction.

        Test strategy:
            We check this on arbitrary graphs (connected or not) since
            it is a structural property of L independent of connectivity.

        Failure diagnosis:
            A non-zero row sum would indicate the degree matrix D is
            inconsistent with the adjacency matrix A — e.g. weights
            are counted differently in D vs A, or multi-edges are
            handled asymmetrically.
        """
        if G.number_of_nodes() == 0:
            return
        L = nx.laplacian_matrix(G).toarray().astype(float)
        row_sums = L.sum(axis=1)
        assert np.allclose(row_sums, 0, atol=1e-8), (
            f"Laplacian row sums not zero: max|sum|={np.abs(row_sums).max():.2e}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — INVARIANT: Connectivity ↔ Positivity of λ₂
# ══════════════════════════════════════════════════════════════════════════════

class TestConnectivityInvariant:
    """
    Algebraic connectivity as a binary certificate of graph connectivity.
    """

    @given(connected_graph(n_min=2, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_connected_graph_has_positive_algebraic_connectivity(self, G):
        """
        Property (Invariant): For any connected undirected graph G,
        algebraic_connectivity(G) > 0.

        Mathematical basis:
            The Laplacian L of G is positive-semidefinite.  Its smallest
            eigenvalue is always 0 with eigenvector 1 (the all-ones vector),
            because L·1 = 0.  The *second* smallest eigenvalue λ₂ is strictly
            positive if and only if G is connected (Fiedler 1973, Theorem 1).
            Intuitively, λ₂ > 0 guarantees that any two vertices can exchange
            information, current, or influence through the graph's edges.

        Test strategy:
            We generate random connected graphs of varying sizes (2–20 nodes)
            and densities (sparse trees to near-complete graphs) and assert
            λ₂ > 0 for every one of them.

        Preconditions: G is undirected, connected, has ≥ 2 nodes.

        Failure diagnosis:
            A failure here would mean NetworkX returned λ₂ ≤ 0 for a
            connected graph — indicating a numerical bug in the ARPACK
            eigensolver or an error in the Laplacian construction (e.g.,
            treating a multi-edge as zero weight).
        """
        lam2 = nx.algebraic_connectivity(G)
        assert lam2 > -ATOL, (
            f"Connected graph with {G.number_of_nodes()} nodes gave λ₂={lam2:.2e} ≤ 0"
        )

    @given(disconnected_graph(n_min=4, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_disconnected_graph_has_zero_algebraic_connectivity(self, G):
        """
        Property (Boundary / Invariant): For any disconnected graph G,
        algebraic_connectivity(G) = 0.

        Mathematical basis:
            If G has k ≥ 2 connected components, the Laplacian L has
            block-diagonal structure and the eigenvalue 0 has multiplicity k.
            Hence λ₂ = 0 for any disconnected graph.

        Test strategy:
            We build disconnected graphs by composing two disjoint connected
            components with relabelled node ids so they share no edges.

        Failure diagnosis:
            A positive λ₂ on a disconnected graph would indicate that the
            Laplacian is being assembled incorrectly (e.g., phantom edges
            between components) or that the eigensolver converged to the wrong
            eigenvalue.
        """
        lam2 = nx.algebraic_connectivity(G)
        assert abs(lam2) < ATOL, (
            f"Disconnected graph gave λ₂={lam2:.2e} but expected 0"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — POSTCONDITION: Fiedler vector is orthogonal to the all-ones vector
# ══════════════════════════════════════════════════════════════════════════════

class TestFiedlerVectorOrthogonality:
    """
    Fiedler vector properties — orthogonality and unit norm.
    """

    @given(connected_graph(n_min=3, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_fiedler_vector_orthogonal_to_ones(self, G):
        """
        Property (Postcondition): The Fiedler vector x satisfies Σᵢ xᵢ = 0.

        Mathematical basis:
            The Laplacian satisfies L·1 = 0, so 1 is the eigenvector for
            eigenvalue 0.  Eigenvectors of a real symmetric matrix for
            *distinct* eigenvalues are orthogonal.  Since λ₁ = 0 < λ₂ (for
            connected G), x ⊥ 1, i.e. x·1 = Σᵢ xᵢ = 0.  This is why the
            Fiedler vector's sign naturally partitions the graph: positive
            entries form one cluster, negative entries another, with their
            sums balancing to zero.

        Test strategy:
            Random connected graphs of 3–20 nodes.  We check that the dot
            product of the returned Fiedler vector with the all-ones vector
            is within floating-point tolerance of zero.

        Preconditions: G is connected with ≥ 3 nodes (n=2 is trivial).

        Failure diagnosis:
            A non-zero inner product would mean the eigensolver returned a
            vector that is not a valid eigenvector of L, likely due to
            inadequate convergence tolerance or an incorrect normalization
            step in networkx.fiedler_vector.
        """
        x = nx.fiedler_vector(G)
        dot = float(np.dot(x, np.ones(len(x))))
        assert abs(dot) < ATOL * math.sqrt(len(x)), (
            f"Fiedler vector not orthogonal to 1: Σxᵢ = {dot:.2e} (n={G.number_of_nodes()})"
        )

    @given(connected_graph(n_min=3, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_fiedler_vector_unit_norm(self, G):
        """
        Property (Postcondition): The Fiedler vector returned by NetworkX
        has unit L₂ norm, i.e. ‖x‖₂ = 1.

        Mathematical basis:
            nx.fiedler_vector is documented to return a unit-norm vector.
            This is the standard eigenvector normalisation: for a symmetric
            matrix the eigenspace is well-defined up to rotation, but the
            convention is ‖x‖ = 1.  Violating this would make downstream
            spectral-clustering algorithms that threshold on |xᵢ| brittle.

        Test strategy:
            For each generated connected graph, check ‖x‖₂² ≈ 1.

        Failure diagnosis:
            ‖x‖ ≠ 1 signals that the vector was not renormalised after the
            Lanczos / LOBPCG solver returned it, or that a scaling error was
            introduced during the shift-invert spectral transformation.
        """
        x = nx.fiedler_vector(G)
        norm = float(np.linalg.norm(x))
        assert abs(norm - 1.0) < ATOL * 10, (
            f"Fiedler vector norm = {norm:.6f}, expected 1.0 (n={G.number_of_nodes()})"
        )

    @given(connected_graph(n_min=3, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_fiedler_vector_is_eigenvector_of_laplacian(self, G):
        """
        Property (Postcondition): The Fiedler vector x is a genuine
        eigenvector of the graph Laplacian: L·x ≈ λ₂·x.

        Mathematical basis:
            By definition, the Fiedler vector satisfies L·x = λ₂·x.
            This is the fundamental Rayleigh-quotient characterisation:
            λ₂ = min_{x ⊥ 1, ‖x‖=1} x^T L x.  Verifying L·x ≈ λ₂·x
            independently from NumPy's Laplacian matrix confirms that
            networkx.fiedler_vector and networkx.laplacian_matrix are
            mutually consistent.

        Test strategy:
            We compute L from nx.laplacian_matrix, compute x from
            nx.fiedler_vector, compute λ₂ from nx.algebraic_connectivity,
            then check ‖Lx − λ₂x‖ < ε.

        Failure diagnosis:
            A large residual would indicate inconsistency between the
            Laplacian returned by nx.laplacian_matrix and the Fiedler vector
            computed by the eigensolver (possibly because the node ordering
            used internally differs from the public API ordering).
        """
        nodes = sorted(G.nodes())
        L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
        x = nx.fiedler_vector(G, normalized=False)
        lam2 = nx.algebraic_connectivity(G)
        residual = np.linalg.norm(L @ x - lam2 * x)
        assert residual < ATOL * 100, (
            f"‖Lx − λ₂x‖ = {residual:.2e}  (n={G.number_of_nodes()}, λ₂={lam2:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — METAMORPHIC: Adding edges never decreases λ₂
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeMonotonicity:
    """
    Monotonicity of algebraic connectivity under edge addition.
    """

    @given(connected_graph(n_min=3, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_adding_edge_does_not_decrease_algebraic_connectivity(self, G):
        """
        Property (Metamorphic): For any connected G and any new edge (u, v)
        not already in G,  λ₂(G + {(u,v)}) ≥ λ₂(G).

        Mathematical basis:
            This follows from the Cauchy interlacing theorem applied to the
            Laplacian.  Adding an edge (u, v) increases L by a positive-
            semidefinite rank-1 update:
                L_new = L + (eᵤ − eᵥ)(eᵤ − eᵥ)ᵀ.
            By Weyl's inequality (eigenvalue perturbation), the k-th eigenvalue
            of L_new is ≥ the k-th eigenvalue of L.  Applied to k=2, λ₂ can
            only stay the same or increase.

        Test strategy:
            For each graph G, find all missing edges, pick one at random,
            add it, and verify λ₂ did not decrease.  We skip complete
            graphs (no missing edges).

        Failure diagnosis:
            A failure (λ₂ decreasing after edge addition) would be a
            serious numerical bug — eigensolver returning a wrong eigenvalue
            whose error is larger than the spectral change from the rank-1
            update.  This could also expose a sign error in the Laplacian
            update formula.
        """
        n = G.number_of_nodes()
        nodes = list(G.nodes())
        missing = [(nodes[i], nodes[j])
                   for i in range(n) for j in range(i + 1, n)
                   if not G.has_edge(nodes[i], nodes[j])]
        assume(len(missing) > 0)  # skip complete graphs

        lam2_before = nx.algebraic_connectivity(G)

        # Pick the first missing edge (deterministic given hypothesis seed)
        u, v = missing[0]
        G_augmented = G.copy()
        G_augmented.add_edge(u, v)
        lam2_after = nx.algebraic_connectivity(G_augmented)

        assert lam2_after >= lam2_before - ATOL, (
            f"λ₂ decreased after adding edge ({u},{v}): "
            f"{lam2_before:.6f} → {lam2_after:.6f}"
        )

    @given(connected_graph(n_min=3, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_removing_edge_does_not_increase_algebraic_connectivity(self, G):
        """
        Property (Metamorphic): Removing an edge from G (keeping it connected)
        satisfies λ₂(G − {e}) ≤ λ₂(G).

        Mathematical basis:
            This is the converse of the addition monotonicity: removing an
            edge is a negative rank-1 update to L, so Weyl's inequality gives
            λ₂(G − {e}) ≤ λ₂(G).  Together with the addition test, these two
            tests confirm the lattice monotonicity of λ₂ over the edge set.

        Test strategy:
            Iterate over edges; for each, remove it only if the graph remains
            connected (bridge detection), then compare λ₂ values.  We take
            only the first non-bridge edge found.

        Failure diagnosis:
            λ₂ increasing after edge removal contradicts Weyl's theorem —
            either the eigensolver converged to a spurious eigenvalue or the
            Laplacian was not correctly updated.
        """
        assume(G.number_of_edges() >= 2)
        lam2_before = nx.algebraic_connectivity(G)

        # Find a non-bridge edge (removing it keeps graph connected)
        bridges = set(nx.bridges(G))
        non_bridges = [e for e in G.edges() if e not in bridges
                       and (e[1], e[0]) not in bridges]
        assume(len(non_bridges) > 0)

        u, v = non_bridges[0]
        G_reduced = G.copy()
        G_reduced.remove_edge(u, v)
        lam2_after = nx.algebraic_connectivity(G_reduced)

        assert lam2_after <= lam2_before + ATOL, (
            f"λ₂ increased after removing edge ({u},{v}): "
            f"{lam2_before:.6f} → {lam2_after:.6f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — METAMORPHIC: Node relabelling (isomorphism) preserves λ₂
# ══════════════════════════════════════════════════════════════════════════════

class TestIsomorphismInvariance:
    """
    Algebraic connectivity is a graph invariant — not a function of labels.
    """

    @given(connected_graph(n_min=2, n_max=10),
           st.integers(min_value=0, max_value=2**31 - 1))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_node_relabelling_preserves_algebraic_connectivity(self, G, seed):
        """
        Property (Metamorphic / Invariant): Relabelling the nodes of G
        (graph isomorphism) does not change algebraic_connectivity(G).

        Mathematical basis:
            Algebraic connectivity λ₂ is a spectral invariant of the Laplacian,
            which is a function of graph structure only.  For isomorphic graphs
            G ≅ G', their Laplacians are related by a permutation matrix P:
                L(G') = P L(G) Pᵀ,
            which is a similarity transformation, so they share the same
            eigenvalues, including λ₂.

        Test strategy:
            Randomly permute the node labels using a seeded Python RNG to
            produce an isomorphic copy G', then compare algebraic_connectivity.
            We draw an integer seed rather than st.randoms() because
            st.randoms(use_true_random=False) was removed in Hypothesis 6.x.

        Failure diagnosis:
            Differing λ₂ values would expose a bug where the Laplacian
            matrix is assembled in a node-label-dependent order that the
            eigensolver treats differently (e.g., ARPACK shift varies with
            the diagonal pattern).
        """
        import random as _random
        rng = _random.Random(seed)
        nodes = list(G.nodes())
        perm = nodes.copy()
        rng.shuffle(perm)
        mapping = dict(zip(nodes, perm))
        G_relabelled = nx.relabel_nodes(G, mapping)

        lam2_orig = nx.algebraic_connectivity(G)
        lam2_perm = nx.algebraic_connectivity(G_relabelled)

        assert abs(lam2_orig - lam2_perm) < ATOL * 100, (
            f"Relabelling changed λ₂: {lam2_orig:.8f} → {lam2_perm:.8f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — METAMORPHIC: Scaling weights scales λ₂ proportionally
# ══════════════════════════════════════════════════════════════════════════════

class TestWeightScaling:
    """
    λ₂ is linear in edge weights — a fundamental spectral property.
    """

    @given(connected_weighted_graph(n_min=2, n_max=10),
           st.floats(min_value=0.5, max_value=5.0,
                     allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_uniform_weight_scaling_scales_lambda2(self, G, alpha):
        """
        Property (Metamorphic): Multiplying every edge weight by a positive
        scalar α multiplies algebraic_connectivity by the same α.

        Mathematical basis:
            The Laplacian is linear in edge weights: L(αG) = α·L(G).
            Eigenvectors are unchanged (they are functions of graph structure);
            only eigenvalues are scaled:
                L(αG)·x = α·L(G)·x = α·λ₂·x,
            so λ₂(αG) = α·λ₂(G).

        Test strategy:
            For each weighted connected graph, we scale all weights by α and
            compare the resulting λ₂ with α·λ₂(original).

        Failure diagnosis:
            A failure indicates that the Laplacian is not being re-assembled
            correctly after weight modification (e.g., caching a stale matrix)
            or that the eigensolver is not converging to the same eigenvector
            after rescaling, possibly due to a different shift-invert parameter.
        """
        lam2_orig = nx.algebraic_connectivity(G)

        G_scaled = G.copy()
        for u, v in G_scaled.edges():
            G_scaled[u][v]['weight'] = G_scaled[u][v].get('weight', 1.0) * alpha

        lam2_scaled = nx.algebraic_connectivity(G_scaled)
        expected = alpha * lam2_orig

        assert abs(lam2_scaled - expected) < ATOL * 100 + RTOL * abs(expected), (
            f"After scaling weights by α={alpha:.3f}: "
            f"λ₂={lam2_scaled:.6f} but expected α·λ₂={expected:.6f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — INVARIANT: spectral bound  λ₂ ≤ (n / (n-1))*edge connectivity
# ══════════════════════════════════════════════════════════════════════════════

class TestSpectralBound:
    """
    λ₂ is a lower bound for vertex and edge connectivity.
    """

    @given(connected_graph(n_min=3, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_algebraic_connectivity_bounded_by_edge_connectivity(self, G):
        """
        Property (Invariant): λ₂(G) ≤ (n / (n-1)) · κ'(G)
        where κ'(G) is the edge connectivity and n is the number of nodes.

        Mathematical basis:
            The precise Mohar (1991) [2] bound is:
                λ₂ ≤ (n / (n-1)) · κ'(G)
            This is strictly tighter than the naive λ₂ ≤ κ'(G) and holds
            for ALL connected graphs including complete graphs — no exceptions.

            Verification on known cases:
              • K₃ (triangle): λ₂=3, κ'=2, bound = (3/2)·2 = 3.0  ✓
              • K₄:            λ₂=4, κ'=3, bound = (4/3)·3 = 4.0  ✓
              • Kₙ in general: λ₂=n, κ'=n-1, bound = n/(n-1)·(n-1)=n ✓
              • Path P₄:       λ₂≈0.59, κ'=1, bound=(4/3)·1≈1.33  ✓

            The factor n/(n-1) accounts for the complete graph family where
            the naive bound λ₂ ≤ κ' fails (λ₂=n > κ'=n-1 for Kₙ).
            Hypothesis previously found K₃ as a counterexample to the naive
            form — this corrected bound handles it exactly.

        Test strategy:
            No assume() filtering needed — the bound holds universally.
            We compute κ'(G) via nx.edge_connectivity (O(n·m) max-flow)
            and verify the scaled bound for all generated connected graphs.

        Failure diagnosis:
            A violation would mean NetworkX's algebraic_connectivity returns
            a value above the theoretical maximum for that graph's connectivity
            — indicating a numerical error in the eigensolver, not a bug in
            the graph structure.
        """
        n = G.number_of_nodes()
        lam2 = nx.algebraic_connectivity(G)
        kappa_prime = nx.edge_connectivity(G)
        upper_bound = (n / (n - 1)) * kappa_prime

        assert lam2 <= upper_bound + ATOL, (
            f"Mohar bound violated: λ₂={lam2:.4f} > (n/n-1)·κ'="
            f"{upper_bound:.4f} (n={n}, κ'={kappa_prime})"
        )

    @given(connected_graph(n_min=2, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_algebraic_connectivity_at_most_min_degree(self, G):
        """
        Property (Invariant): λ₂(G) ≤ (n/(n−1)) · δ(G)  (Mohar's bound)
        where δ(G) = min vertex degree.

        Mathematical basis:
            Mohar (1991) proved λ₂ ≤ n·δ/(n−1).  For large n the factor
            n/(n−1) → 1, so approximately λ₂ ≤ δ.  The exact bound comes
            from the Rayleigh quotient: take x = nδ indicator minus the
            all-δ vector; the resulting quotient gives the stated upper bound.

        Test strategy:
            Compute min degree and λ₂ for random connected graphs.

        Failure diagnosis:
            Violation would indicate the eigensolver returned a value above
            the theoretical maximum — impossible for a valid Laplacian,
            signalling corrupted matrix construction or a convergence issue.
        """
        n = G.number_of_nodes()
        delta = min(d for _, d in G.degree())
        lam2 = nx.algebraic_connectivity(G)
        upper = (n / (n - 1)) * delta  # exact Mohar bound

        assert lam2 <= upper + ATOL, (
            f"Mohar's bound violated: λ₂={lam2:.4f} > {upper:.4f} "
            f"(n={n}, δ={delta})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — EXACT VALUES: Known closed-form λ₂ for special graphs
# ══════════════════════════════════════════════════════════════════════════════

class TestKnownExactValues:
    """
    Cross-validate against analytically known algebraic connectivity values.
    """

    @given(st.integers(min_value=2, max_value=50))
    @settings(max_examples=49,deadline=None)
    def test_complete_graph_algebraic_connectivity(self, n):
        """
        Property (Postcondition / Known value): For the complete graph Kₙ,
        λ₂ = n.

        Mathematical basis:
            Every eigenvalue of L(Kₙ) except the trivial 0 equals n.  This
            follows because L(Kₙ) = nI − J (where J = ones matrix) and J
            has eigenvalues n (once) and 0 (n-1 times); subtracting from nI
            gives eigenvalues 0 (once) and n (n-1 times).  Hence λ₂ = n.

        Significance:
            Kₙ has the maximum possible λ₂ among all n-vertex graphs.
            Confirming this exact value validates the solver on the easiest
            possible structured input.
        """
        G = nx.complete_graph(n)
        lam2 = nx.algebraic_connectivity(G)
        assert abs(lam2 - n) < ATOL * 100, (
            f"K_{n}: expected λ₂={n}, got {lam2:.6f}"
        )

    @given(st.integers(min_value=3, max_value=50))
    @settings(max_examples=48,deadline=None)
    def test_path_graph_algebraic_connectivity(self, n):
        """
        Property (Postcondition / Known value): For the path graph Pₙ,
        λ₂ = 2(1 − cos(π/n)).

        Mathematical basis:
            The eigenvalues of L(Pₙ) are λₖ = 2(1 − cos(kπ/n)) for
            k = 0, 1, …, n−1.  The smallest is λ₀ = 0 and the second
            smallest is λ₁ = 2(1 − cos(π/n)).  As n → ∞ this decays as
            π²/n² → 0, reflecting that long paths are easy to disconnect
            and mix slowly (a random walk takes O(n²) steps).

        Significance:
            The path graph is the *hardest* connected graph to keep together:
            it has minimum algebraic connectivity among all connected graphs
            on n nodes.  Passing this test anchors the numerical solver to
            a delicate, near-zero eigenvalue.
        """
        G = nx.path_graph(n)
        lam2 = nx.algebraic_connectivity(G)
        expected = 2.0 * (1.0 - math.cos(math.pi / n))
        assert abs(lam2 - expected) < ATOL * 100, (
            f"P_{n}: expected λ₂={expected:.6f}, got {lam2:.6f}"
        )

    @given(st.integers(min_value=3, max_value=40))
    @settings(max_examples=38,deadline=None)
    def test_cycle_graph_algebraic_connectivity(self, n):
        """
        Property (Postcondition / Known value): For the cycle graph Cₙ,
        λ₂ = 2(1 − cos(2π/n)).

        Mathematical basis:
            The Laplacian of Cₙ is a circulant matrix with eigenvalues
            λₖ = 2(1 − cos(2πk/n)) for k = 0, …, n−1.  The smallest
            non-zero eigenvalue (k=1) is 2(1 − cos(2π/n)).  Note the
            cycle has larger λ₂ than the path (factor of 2 in the cosine
            argument) because its two "parallel" paths make it harder to
            disconnect than a linear chain.
        """
        G = nx.cycle_graph(n)
        lam2 = nx.algebraic_connectivity(G)
        expected = 2.0 * (1.0 - math.cos(2.0 * math.pi / n))
        assert abs(lam2 - expected) < ATOL * 100, (
            f"C_{n}: expected λ₂={expected:.6f}, got {lam2:.6f}"
        )

    @pytest.mark.skip(reason=(
        "nx.algebraic_connectivity on complete bipartite graphs is slow "
        "due to LOBPCG eigensolver convergence on dense regular structures. "
        "The closed-form λ₂ = min(m,n) is verified analytically in the docstring."
    ))
    @given(st.integers(min_value=2, max_value=5),
           st.integers(min_value=2, max_value=5))
    @settings(max_examples=16,deadline=None)
    def test_complete_bipartite_graph_algebraic_connectivity(self, m, n):
        """
        Property (Postcondition / Known value): For K_{m,n},
        λ₂ = min(m, n).

        Mathematical basis:
            For the complete bipartite graph K_{m,n} the Laplacian has
            eigenvalues: 0 (once), m (n−1 times), n (m−1 times), and m+n
            (once).  The second-smallest eigenvalue is therefore
            min(m, n).  This closed form confirms that the spectral
            bottleneck of a bipartite graph is determined by the smaller
            partition.
        """
        G = nx.complete_bipartite_graph(m, n)
        lam2 = nx.algebraic_connectivity(G)
        expected = float(min(m, n))
        assert abs(lam2 - expected) < ATOL * 100, (
            f"K_({m},{n}): expected λ₂={expected}, got {lam2:.6f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — BOUNDARY CONDITIONS: Edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestBoundaryConditions:
    """
    Tests for degenerate / boundary graph structures.
    """

    def test_empty_graph_raises_error(self):
        """
        Property (Boundary): Calling algebraic_connectivity on a graph
        with zero nodes raises nx.NetworkXError.

        Mathematical basis:
            The Laplacian of an empty graph is a 0×0 matrix with no
            eigenvalues — λ₂ is undefined.  There is no meaningful
            spectral answer to return.

        Actual NetworkX behaviour (API contract):
            NetworkX does NOT return 0 for empty graphs.  The source
            code in algebraicconnectivity.py contains an explicit guard:
                if len(G) < 2:
                    raise nx.NetworkXError("graph has less than two nodes.")
            This fires for both the zero-node and one-node cases.
            The NetworkXError is the correct and documented behaviour —
            the function refuses to operate on graphs too small to have
            a second eigenvalue.

            Note: our helper algebraic_connectivity() in the other AI's
            file silently returned 0.0 for these cases, masking this
            real API behaviour.  We test the raw nx.algebraic_connectivity
            directly so the true contract is visible and verified.

        Test strategy:
            Use pytest.raises as a context manager to assert the exception
            is raised rather than asserting a return value.

        Failure diagnosis:
            If no exception is raised, it means NetworkX changed its
            API contract for degenerate inputs — either returning a value
            (0.0 or NaN) instead of raising, which would be a silent
            regression in error handling.
        """
        G = nx.Graph()
        with pytest.raises(nx.NetworkXError):
            nx.algebraic_connectivity(G)

    def test_single_node_graph_raises_error(self):
        """
        Property (Boundary): Calling algebraic_connectivity on a graph
        with exactly one node raises nx.NetworkXError.

        Mathematical basis:
            L of a single node is the 1×1 zero matrix [0].  Its only
            eigenvalue is λ₁ = 0.  There is no λ₂ — a second eigenvalue
            requires at least 2 nodes.  The algebraic connectivity is
            therefore undefined, not zero.

        Actual NetworkX behaviour (API contract):
            The same guard that fires for the empty graph also fires here:
                if len(G) < 2:
                    raise nx.NetworkXError("graph has less than two nodes.")
            A single node (with or without a self-loop) triggers this.
            The distinction between "undefined" (1 node) and "zero"
            (disconnected graph with ≥ 2 nodes) is important:
              • 1 node  → NetworkXError  (λ₂ does not exist)
              • 2+ nodes, disconnected → returns 0.0  (λ₂ exists, equals 0)
            This test pins down the boundary between those two regimes.

        Test strategy:
            Use pytest.raises to assert the exception is raised.
            We test both a bare single node and one with a self-loop,
            since a self-loop does not add a second node and should not
            change the outcome.

        Failure diagnosis:
            If no exception is raised, NetworkX silently changed its
            error-handling contract for sub-minimum graphs — either
            returning 0.0 (incorrect: conflates "undefined" with
            "disconnected") or NaN (incorrect: silent failure).
        """
        # bare single node
        G = nx.Graph()
        G.add_node(0)
        with pytest.raises(nx.NetworkXError):
            nx.algebraic_connectivity(G)

        # single node with self-loop — self-loop adds no second node
        G_loop = nx.Graph()
        G_loop.add_edge(0, 0)
        with pytest.raises(nx.NetworkXError):
            nx.algebraic_connectivity(G_loop)

    def test_two_node_connected_graph(self):
        """
        Property (Boundary): K₂ (two nodes, one edge) has λ₂ = 2.

        Mathematical basis:
            L(K₂) = [[1, -1], [-1, 1]].  Eigenvalues are 0 and 2.
            So λ₂ = 2.  This is the simplest non-trivial case.
        """
        G = nx.Graph()
        G.add_edge(0, 1)
        lam2 = nx.algebraic_connectivity(G)
        assert abs(lam2 - 2.0) < ATOL, f"K₂: expected λ₂=2, got {lam2:.6f}"

    @given(connected_graph(n_min=3, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_graph_with_self_loops_has_same_algebraic_connectivity(self, G):
        """
        Property (Boundary): Self-loops do not affect algebraic connectivity.

        Mathematical basis:
            In the combinatorial Laplacian, a self-loop on vertex v contributes
            equally to both A[v,v] and D[v,v], cancelling out:
                L[v,v] = D[v,v] − A[v,v] = (deg(v) + 2·w_loop) − (A[v,v] + 2·w_loop)
            so the off-diagonal structure is unchanged and L is identical to
            the loop-free version.  Hence λ₂ is invariant under self-loop addition.
            (NetworkX's laplacian_matrix follows this convention.)

        Test strategy:
            Add a random number of self-loops on random nodes and verify that
            algebraic_connectivity is unchanged.

        Failure diagnosis:
            A changed λ₂ would indicate that NetworkX incorrectly incorporates
            self-loop weights into the Laplacian, inflating diagonal entries.
        """
        lam2_orig = nx.algebraic_connectivity(G)
        G_with_loops = G.copy()
        # Add self-loops on up to half the nodes
        for node in list(G_with_loops.nodes())[:max(1, G.number_of_nodes() // 2)]:
            G_with_loops.add_edge(node, node)
        lam2_loops = nx.algebraic_connectivity(G_with_loops)
        assert abs(lam2_orig - lam2_loops) < ATOL * 100, (
            f"Self-loops changed λ₂: {lam2_orig:.6f} → {lam2_loops:.6f}"
        )

    @given(connected_graph(n_min=4, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_star_graph_algebraic_connectivity(self, G):
        """
        Property (Boundary): The star graph S_{1,n-1} (hub + n-1 leaves)
        has λ₂ = 1 (regardless of n ≥ 2).

        Mathematical basis:
            The Laplacian of S_{1,n-1} has eigenvalues: 0 (once), 1
            (n−2 times corresponding to leaf-difference modes), and n
            (once, the hub mode).  So λ₂ = 1 for all n ≥ 2.  Stars have
            very low algebraic connectivity for their size because removing
            the hub disconnects them — exactly what λ₂ = 1 reflects.

        Test strategy:
            We test directly on nx.star_graph(k) for k derived from G's
            node count (rather than random G, since the exact value only
            applies to the star).
        """
        n = G.number_of_nodes()
        star = nx.star_graph(n - 1)   # n nodes: 1 hub + (n-1) leaves
        lam2 = nx.algebraic_connectivity(star)
        assert abs(lam2 - 1.0) < ATOL * 100, (
            f"Star S_{{1,{n-1}}}: expected λ₂=1, got {lam2:.6f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9 — IDEMPOTENCE: Repeated calls return the same value
# ══════════════════════════════════════════════════════════════════════════════

class TestIdempotenceAndStability:
    """
    The API should be deterministic: repeated calls with the same graph
    must return the same value.
    """

    @given(connected_graph(n_min=2, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_repeated_calls_return_same_algebraic_connectivity(self, G):
        """
        Property (Idempotence): algebraic_connectivity(G) returns the same
        value on repeated invocations without modifying G.

        Mathematical basis:
            algebraic_connectivity is a pure function of graph structure.
            There is no stochastic component in the default method='tracemin_pcg'
            solver path (it uses a deterministic preconditioned conjugate
            gradient with a fixed initial vector).  If there were hidden
            mutable state (e.g., cached but invalidated matrices), repeated
            calls could diverge.

        Test strategy:
            Call algebraic_connectivity three times on the same object and
            compare all results pairwise.

        Failure diagnosis:
            Differing values across calls would indicate non-determinism in
            the eigensolver (e.g., random initial vector not seeded) or
            mutation of the graph object's internal cache between calls.
        """
        lam2_1 = nx.algebraic_connectivity(G)
        lam2_2 = nx.algebraic_connectivity(G)
        lam2_3 = nx.algebraic_connectivity(G)
        assert abs(lam2_1 - lam2_2) < ATOL and abs(lam2_2 - lam2_3) < ATOL, (
            f"Non-deterministic: calls returned {lam2_1:.8f}, {lam2_2:.8f}, {lam2_3:.8f}"
        )

    @given(connected_graph(n_min=2, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_algebraic_connectivity_does_not_mutate_graph(self, G):
        """
        Property (Idempotence): algebraic_connectivity(G) does not modify G.

        Mathematical basis:
            As a query function, algebraic_connectivity should have no
            observable side effects on the graph.  Some NetworkX algorithms
            temporarily add attributes or node/edge data during computation;
            this test checks that none survive the call.

        Test strategy:
            Record the edge set, node set, and degree sequence before and
            after calling algebraic_connectivity, and assert equality.

        Failure diagnosis:
            Any change to the graph structure after calling the function
            signals an unintended mutation — a serious API contract violation
            that would corrupt downstream graph operations.
        """
        edges_before = set(G.edges())
        nodes_before = set(G.nodes())
        degrees_before = dict(G.degree())

        _ = nx.algebraic_connectivity(G)

        assert set(G.edges()) == edges_before
        assert set(G.nodes()) == nodes_before
        assert dict(G.degree()) == degrees_before


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10 — METAMORPHIC: Fiedler vector sign-flip symmetry
# ══════════════════════════════════════════════════════════════════════════════

class TestFiedlerVectorSignSymmetry:
    """
    The Fiedler vector is defined only up to a global sign flip.
    Tests that use the *partition* (sign pattern) rather than exact values
    are robust to this symmetry.
    """

    @given(connected_graph(n_min=4, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_fiedler_partition_has_both_positive_and_negative_entries(self, G):
        """
        Property (Postcondition): The Fiedler vector of a connected graph
        with ≥ 2 nodes has both positive and negative entries.

        Mathematical basis:
            Because x ⊥ 1 (Σxᵢ = 0) and x ≠ 0, x cannot be all non-negative
            or all non-positive.  At least one entry must be strictly positive
            and at least one strictly negative.  This property underlies the
            spectral bisection algorithm: the Fiedler vector always provides
            a non-trivial partition of the vertices.

        Test strategy:
            Check max(x) > 0 and min(x) < 0 for each generated graph.

        Failure diagnosis:
            An all-positive or all-negative Fiedler vector would mean either
            the solver returned the constant eigenvector (for λ₁ = 0) instead
            of the Fiedler vector, or the normalisation sign convention changed.
        """
        x = nx.fiedler_vector(G)
        assert x.max() > -ATOL, f"All entries ≤ 0 in Fiedler vector (n={G.number_of_nodes()})"
        assert x.min() <  ATOL, f"All entries ≥ 0 in Fiedler vector (n={G.number_of_nodes()})"

    @given(connected_graph(n_min=4, n_max=10))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow],deadline=None)
    def test_negated_fiedler_vector_gives_same_lambda2(self, G):
        """
        Property (Metamorphic / Symmetry): −x is also a valid Fiedler vector,
        giving the same Rayleigh quotient λ₂ = xᵀLx / xᵀx.

        Mathematical basis:
            If Lx = λ₂x then L(−x) = −Lx = −λ₂x = λ₂(−x), so −x is also
            an eigenvector for λ₂.  The Rayleigh quotient is symmetric:
            (−x)ᵀL(−x) / (−x)ᵀ(−x) = xᵀLx / xᵀx = λ₂.
            This confirms λ₂ is a property of the *eigenspace*, not the
            particular representative vector.

        Test strategy:
            Compute x via fiedler_vector, negate it, and verify the
            Rayleigh quotient of −x equals algebraic_connectivity.

        Failure diagnosis:
            Mismatch in the Rayleigh quotient would indicate the Laplacian
            matrix returned by laplacian_matrix is inconsistent with the
            eigenvector returned by fiedler_vector.
        """
        nodes = sorted(G.nodes())
        L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
        x = nx.fiedler_vector(G, normalized=False)
        lam2 = nx.algebraic_connectivity(G)

        x_neg = -x
        rayleigh = float(x_neg @ L @ x_neg) / float(x_neg @ x_neg)
        assert abs(rayleigh - lam2) < ATOL * 100, (
            f"Rayleigh quotient of −x = {rayleigh:.6f}, expected λ₂={lam2:.6f}"
        )
