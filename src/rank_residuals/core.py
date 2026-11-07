"""Float local push, rational residual verification, explicit dangling semantics."""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
import math
import re

MAX_NODES = 200
MAX_EDGES = 10_000
MAX_WEIGHT = 10**9
MAX_PUSHES = 200_000


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _alpha(value):
    if isinstance(value, str):
        if len(value) > 64 or not re.fullmatch(r"[+-]?[0-9]+(?:/[0-9]+|\.[0-9]+)?", value):
            raise ValueError("alpha must be an integer, fraction, or decimal string without an exponent")
        try:
            value = Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("invalid alpha") from exc
    if type(value) not in (int, Fraction):
        raise ValueError("alpha must be exact: use a string or Fraction, not a float")
    value = Fraction(value)
    if not 0 <= value < 1 or value.denominator > 1_000_000:
        raise ValueError("alpha must lie in [0,1) with denominator at most 1000000")
    return value


def _tolerance(value):
    if type(value) not in (int, float) or not 1e-12 <= value <= 1:
        raise ValueError("tolerance must be a finite number in [1e-12,1]")
    return float(value)


def _distribution(weights, nodes, name):
    if weights is None:
        weights = [1] * nodes
    if not isinstance(weights, (list, tuple)) or len(weights) != nodes:
        raise ValueError(f"{name} must contain one integer weight per node")
    for weight in weights:
        _integer(weight, name + " weight", 0, MAX_WEIGHT)
    total = sum(weights)
    if total == 0:
        raise ValueError(f"{name} must have positive total weight")
    return tuple(Fraction(weight, total) for weight in weights)


@dataclass(frozen=True, init=False)
class Graph:
    """Directed graph on 0..nodes-1; parallel edge weights add; self-loops allowed."""

    nodes: int
    rows: tuple
    input_edges: int

    def __init__(self, nodes, edges):
        nodes = _integer(nodes, "nodes", 1, MAX_NODES)
        if not isinstance(edges, (list, tuple)) or len(edges) > MAX_EDGES:
            raise ValueError("edges must be a list of at most 10000 triples")
        rows = [{} for _ in range(nodes)]
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 3:
                raise ValueError("each edge must be [source, target, positive integer weight]")
            source = _integer(edge[0], "source", 0, nodes - 1)
            target = _integer(edge[1], "target", 0, nodes - 1)
            weight = _integer(edge[2], "edge weight", 1, MAX_WEIGHT)
            rows[source][target] = rows[source].get(target, 0) + weight
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "rows", tuple(tuple(sorted(row.items())) for row in rows))
        object.__setattr__(self, "input_edges", len(edges))


def _model(graph, personalization, dangling, alpha):
    if not isinstance(graph, Graph):
        raise ValueError("graph must be a Graph")
    v = _distribution(personalization, graph.nodes, "personalization")
    d = v if dangling is None else _distribution(dangling, graph.nodes, "dangling")
    return v, d, _alpha(alpha)


def _scores(values, nodes):
    if not isinstance(values, (list, tuple)) or len(values) != nodes:
        raise ValueError("scores must contain one finite float or integer per node")
    result = []
    for value in values:
        if type(value) not in (int, float):
            raise ValueError("scores must contain finite floats or integers")
        try:
            x = float(value)
        except OverflowError as exc:
            raise ValueError("score too large") from exc
        if not math.isfinite(x) or abs(x) > 1e6:
            raise ValueError("score magnitude must be at most 1000000")
        result.append(Fraction.from_float(x))
    return tuple(result)


def _upward_float(value):
    rounded = float(value)
    return math.nextafter(rounded, math.inf) if Fraction.from_float(rounded) < value else rounded


def verify(graph, scores, *, personalization=None, dangling=None, alpha="17/20", tolerance=1e-8, top_k=None):
    """Recompute residual from original integer/rational model and exact binary floats.

    No floating transition probabilities, push residuals, or convergence flags
    are used. The L1 bound is mathematically valid for this fully specified model.
    """
    v, d, a = _model(graph, personalization, dangling, alpha)
    tolerance = _tolerance(tolerance)
    top_k = min(3, graph.nodes) if top_k is None else _integer(top_k, "top_k", 1, graph.nodes)
    x = _scores(scores, graph.nodes)
    residual = [(1-a)*v[i] - x[i] for i in range(graph.nodes)]
    dangling_mass = Fraction(0)
    for source, row in enumerate(graph.rows):
        if not row:
            dangling_mass += x[source]
        else:
            total = sum(weight for _, weight in row)
            for target, weight in row:
                residual[target] += a * x[source] * Fraction(weight, total)
    for target in range(graph.nodes):
        residual[target] += a * dangling_mass * d[target]
    residual_norm = sum(abs(value) for value in residual)
    bound = residual_norm / (1-a)
    order = sorted(range(graph.nodes), key=lambda i: (-x[i], i))
    gap = x[order[top_k-1]] - x[order[top_k]] if top_k < graph.nodes else None
    # Pairwise score error |e_i-e_j| <= ||e||_1, so gap > B suffices (not 2B).
    separated = gap is None or gap > bound
    return {
        "exact_residual_l1": str(residual_norm), "exact_error_bound_l1": str(bound),
        "error_bound_l1_upper_float": _upward_float(bound),
        "tolerance_met": bound <= Fraction.from_float(tolerance),
        "score_sum": float(sum(x)), "exact_score_sum": str(sum(x)),
        "top_nodes": order[:top_k], "top_k_membership_proved": separated,
        "exact_boundary_gap": str(gap) if gap is not None else None,
        "alpha": str(a),
        "dangling_policy": "personalization" if dangling is None else "explicit fixed distribution",
        "arithmetic": "exact fractions from integer graph weights and exact binary-float candidate scores",
    }


def rank(graph, *, personalization=None, dangling=None, alpha="17/20", tolerance=1e-8, max_pushes=100_000, top_k=None):
    """FIFO forward push, followed by an independent rational residual check.

    Returns an unnormalized reserve approximation; remaining mass is not spread
    arbitrarily. A stopped or exhausted solve still has a valid residual bound.
    """
    v, d, exact_a = _model(graph, personalization, dangling, alpha)
    tolerance = _tolerance(tolerance)
    max_pushes = _integer(max_pushes, "max_pushes", 0, MAX_PUSHES)
    top_k = min(3, graph.nodes) if top_k is None else _integer(top_k, "top_k", 1, graph.nodes)
    a = float(exact_a)
    rows = []
    for row in graph.rows:
        total = sum(weight for _, weight in row)
        rows.append(tuple((j, weight / total) for j, weight in row))
    dangling_row = tuple((i, float(weight)) for i, weight in enumerate(d) if weight)
    reserve = [0.0] * graph.nodes
    residual = [float(weight) for weight in v]
    queue = deque(i for i, value in enumerate(residual) if value)
    queued = [bool(value) for value in residual]
    mass = math.fsum(residual)
    pushes = edge_updates = dangling_updates = 0
    # Aim below the user tolerance, leaving room for arithmetic drift.
    while queue and pushes < max_pushes:
        if mass <= tolerance / 2:
            mass = math.fsum(residual)
            if mass <= tolerance / 2:
                break
        source = queue.popleft()
        queued[source] = False
        amount = residual[source]
        residual[source] = 0.0
        reserve[source] += (1-a) * amount
        targets = rows[source] if rows[source] else dangling_row
        for target, probability in targets:
            increment = a * amount * probability
            if increment:
                residual[target] += increment
                if not queued[target]:
                    queue.append(target)
                    queued[target] = True
        if rows[source]:
            edge_updates += len(targets)
        else:
            dangling_updates += len(targets)
        pushes += 1
        mass -= (1-a) * amount
        if pushes % 256 == 0:
            mass = math.fsum(residual)
    certificate = verify(graph, reserve, personalization=personalization, dangling=dangling,
                         alpha=exact_a, tolerance=tolerance, top_k=top_k)
    if certificate["tolerance_met"]:
        status = "converged"
    elif pushes == max_pushes:
        status = "budget_exhausted"
    else:
        status = "verification_limit"
    return {
        "status": status, "scores": reserve, "push_residual": residual,
        "remaining_push_mass": math.fsum(residual), "pushes": pushes,
        "edge_updates": edge_updates, "dangling_updates": dangling_updates,
        "tolerance": tolerance, "certificate": certificate,
    }
