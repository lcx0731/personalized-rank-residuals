from fractions import Fraction
import math
import random

import pytest

from rank_residuals import Graph, rank, verify
from rank_residuals.reference import solve_exact


@pytest.mark.parametrize("seed", range(12))
def test_push_bound_contains_exact_error_for_random_small_graphs(seed):
    rng = random.Random(seed)
    n = 6
    # Last node is deliberately dangling; self-loops and repeated edges allowed.
    edges = [(rng.randrange(n-1), rng.randrange(n), rng.randrange(1, 8)) for _ in range(15)]
    v = [rng.randrange(5) for _ in range(n)]
    v[0] += 1
    d = None if seed % 2 else [1, 0, 2, 0, 3, 0]
    alpha = [Fraction(1, 2), Fraction(17, 20), Fraction(19, 20)][seed % 3]
    graph = Graph(n, edges)
    _, exact = solve_exact(n, edges, v, d, alpha)
    result = rank(graph, personalization=v, dangling=d, alpha=alpha, tolerance=1e-9)
    assert result["status"] == "converged"
    candidate = list(map(Fraction.from_float, result["scores"]))
    actual_error = sum(abs(x-p) for x, p in zip(candidate, exact))
    bound = Fraction(result["certificate"]["exact_error_bound_l1"])
    assert actual_error <= bound <= Fraction.from_float(1e-9)
    assert Fraction.from_float(result["certificate"]["error_bound_l1_upper_float"]) >= bound
    assert abs(math.fsum(result["scores"]) + result["remaining_push_mass"] - 1) < 5e-14
    if result["certificate"]["top_k_membership_proved"]:
        selected = result["certificate"]["top_nodes"]
        assert min(exact[i] for i in selected) > max(exact[i] for i in range(n) if i not in selected)


@pytest.mark.parametrize("budget", [0, 1, 2, 7, 30])
def test_budgeted_output_still_has_a_correct_independent_bound(budget):
    edges = [(0, 1, 1), (1, 2, 1)]
    graph = Graph(3, edges)
    result = rank(graph, personalization=[1, 0, 0], alpha="1/2", max_pushes=budget, tolerance=1e-12)
    exact = [Fraction(4, 7), Fraction(2, 7), Fraction(1, 7)]
    error = sum(abs(Fraction.from_float(x)-p) for x, p in zip(result["scores"], exact))
    assert error <= Fraction(result["certificate"]["exact_error_bound_l1"])
    assert result["status"] == "budget_exhausted"
    assert verify(graph, result["scores"], personalization=[1, 0, 0], alpha="1/2", tolerance=1e-12) == result["certificate"]


def test_hand_computable_cycle_with_dangling_restart():
    result = rank(Graph(3, [(0, 1, 1), (1, 2, 1)]), personalization=[1, 0, 0], alpha="1/2", tolerance=1e-12)
    assert result["scores"] == pytest.approx([4/7, 2/7, 1/7], abs=1e-12)
    assert result["dangling_updates"] > 0


def test_alpha_zero_and_single_node():
    result = rank(Graph(3, []), alpha=0, personalization=[1, 2, 1], tolerance=1e-12)
    assert result["scores"] == [0.25, 0.5, 0.25]
    assert result["certificate"]["exact_error_bound_l1"] == "0"
    assert result["remaining_push_mass"] == 0
    singleton = rank(Graph(1, []))
    assert singleton["certificate"]["top_k_membership_proved"]
    assert singleton["scores"][0] == pytest.approx(1, abs=1e-8)


def test_self_loop_and_repeated_edges_conserve_and_aggregate():
    graph = Graph(2, [(0, 0, 2), (0, 1, 1), (0, 1, 3), (1, 0, 1)])
    combined = Graph(2, [(0, 0, 2), (0, 1, 4), (1, 0, 1)])
    assert graph.rows == combined.rows
    assert rank(graph)["scores"] == rank(combined)["scores"]
    _, exact = solve_exact(2, [(0, 0, 2), (0, 1, 4), (1, 0, 1)])
    assert rank(graph)["scores"] == pytest.approx(list(map(float, exact)), abs=1e-8)


def test_personalization_mixture_requires_fixed_dangling_distribution():
    edges = [(0, 1, 1)]
    _, p0 = solve_exact(2, edges, [1, 0])
    _, p1 = solve_exact(2, edges, [0, 1])
    _, mixed = solve_exact(2, edges, [1, 1])
    average = [(a+b)/2 for a, b in zip(p0, p1)]
    assert sum(abs(a-b) for a, b in zip(average, mixed)) == Fraction(340, 2109)
    _, fixed0 = solve_exact(2, edges, [1, 0], [1, 1])
    _, fixed1 = solve_exact(2, edges, [0, 1], [1, 1])
    assert tuple((a+b)/2 for a, b in zip(fixed0, fixed1)) == mixed


def test_disconnected_unreachable_component_gets_no_reserve():
    edges = [(0, 1, 1), (1, 0, 1), (2, 3, 1), (3, 2, 1)]
    result = rank(Graph(5, edges), personalization=[1, 0, 0, 0, 0])
    assert result["scores"][2:] == [0, 0, 0]
    assert result["push_residual"][2:] == [0, 0, 0]


def test_positive_rescaling_of_integer_weights_changes_no_model():
    edges = [(0, 1, 2), (0, 2, 3), (1, 0, 1)]
    original = rank(Graph(3, edges), personalization=[1, 2, 3], dangling=[3, 1, 2])
    scaled = rank(Graph(3, [(s,t,7*w) for s,t,w in edges]), personalization=[5, 10, 15], dangling=[21, 7, 14])
    assert original == scaled


def test_permuted_nodes_preserve_solution_within_verified_bounds():
    edges = [(0, 1, 2), (0, 2, 3), (1, 3, 1), (2, 0, 4)]
    permutation = [2, 0, 3, 1]
    v = [1, 0, 2, 0]
    changed_v = [0] * 4
    for i, weight in enumerate(v):
        changed_v[permutation[i]] = weight
    x = rank(Graph(4, edges), personalization=v, tolerance=1e-10)
    y = rank(Graph(4, [(permutation[s], permutation[t], w) for s,t,w in edges]), personalization=changed_v, tolerance=1e-10)
    distance = sum(abs(x["scores"][i] - y["scores"][permutation[i]]) for i in range(4))
    assert distance <= x["certificate"]["error_bound_l1_upper_float"] + y["certificate"]["error_bound_l1_upper_float"]


def test_top_set_gap_rule_handles_ties_and_nonconverged_candidates():
    graph = Graph(3, [])
    exact = verify(graph, [0.5, 0.25, 0.25], alpha=0, personalization=[2, 1, 1], top_k=1)
    assert exact["exact_error_bound_l1"] == "0"
    assert exact["top_k_membership_proved"]
    tie = verify(graph, [0.5, 0.25, 0.25], alpha=0, personalization=[2, 1, 1], top_k=2)
    assert not tie["top_k_membership_proved"]
    wrong = verify(graph, [0, 0, 1], alpha=0, personalization=[2, 1, 1], top_k=1)
    assert not wrong["tolerance_met"]
    assert not wrong["top_k_membership_proved"]


def test_exact_verifier_catches_floating_subtraction_that_would_hide_error():
    # Candidate 1/3 rounded to float has a nonzero exact residual, even at alpha=0.
    result = verify(Graph(3, []), [1/3]*3, alpha=0)
    assert Fraction(result["exact_error_bound_l1"]) == Fraction(1, 2**54)
    assert result["exact_score_sum"] != "1"


@pytest.mark.parametrize("nodes,edges", [
    (0, []), (201, []), (True, []), (2, {}), (2, [[0, 2, 1]]),
    (2, [[0, 1, 0]]), (2, [[0, 1, -1]]), (2, [[0, 1, 1.0]]),
    (2, [[0, 1, True]]), (2, [[0, 1]]), (2, [[0, 1, 1]]*10001),
])
def test_invalid_graph(nodes, edges):
    with pytest.raises(ValueError):
        Graph(nodes, edges)


@pytest.mark.parametrize("alpha", [1, -1, 0.85, True, "1e-3", "0/0", "1/1000001", "NaN", "1" * 65])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        rank(Graph(3, []), alpha=alpha)


@pytest.mark.parametrize("weights", [[0, 0, 0], [1, 2], [1.0, 0, 1], [True, 0, 1], [-1, 2, 1]])
def test_invalid_distributions(weights):
    for name in ["personalization", "dangling"]:
        with pytest.raises(ValueError):
            rank(Graph(3, []), **{name: weights})


def test_candidate_and_resource_validation():
    graph = Graph(3, [])
    for scores in [[0, 1], [True, 0, 1], [math.inf, 0, 1], [math.nan, 0, 1], [10**400, 0, 0]]:
        with pytest.raises(ValueError):
            verify(graph, scores)
    for value in [math.nan, math.inf, 0, 1e-15, True, 10**400]:
        with pytest.raises(ValueError):
            rank(graph, tolerance=value)
    for value in [-1, 200001, True, 1.2]:
        with pytest.raises(ValueError):
            rank(graph, max_pushes=value)
    with pytest.raises(ValueError):
        rank(graph, top_k=4)
