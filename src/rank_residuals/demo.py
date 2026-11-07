"""Deterministic weighted graphs, model contrasts, and reproducible error/work data."""

from fractions import Fraction
import argparse
import json
from pathlib import Path

import numpy as np

from .core import Graph, rank, verify
from .reference import solve_exact


def dense_reference(nodes, edges, v, alpha):
    """Independent numerical solve from raw edges, used only for the larger demo."""
    adjacency = np.zeros((nodes, nodes))
    for i, j, weight in edges:
        adjacency[i, j] += weight
    personalization = np.asarray(v, dtype=float) / sum(v)
    transition = np.zeros_like(adjacency)
    for i in range(nodes):
        transition[i] = adjacency[i] / adjacency[i].sum() if adjacency[i].sum() else personalization
    return np.linalg.solve(np.eye(nodes) - float(alpha)*transition.T, (1-float(alpha))*personalization)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    n = 120
    edges = []
    for block in range(3):
        start = 40 * block
        for i in range(39):
            edges += [(start+i, start+(i+1) % 40, 3), (start+i, start+(i+7) % 40, 1)]
    edges += [(15, 40, 1)]
    v = [1] + [0] * (n-1)
    graph = Graph(n, edges)
    exact_a = Fraction(17, 20)
    numerical_reference = dense_reference(n, edges, v, exact_a)
    reference_check = verify(graph, numerical_reference.tolist(), personalization=v, alpha=exact_a, tolerance=1e-12)
    assert reference_check["tolerance_met"]
    accuracy = []
    final = None
    for tolerance in [1e-2, 1e-4, 1e-6, 1e-8]:
        result = rank(graph, personalization=v, tolerance=tolerance, top_k=5)
        assert result["status"] == "converged"
        accuracy.append({"tolerance": tolerance, "pushes": result["pushes"],
            "edge_updates": result["edge_updates"], "dangling_updates": result["dangling_updates"],
            "l1_distance_to_numerical_reference": float(np.sum(np.abs(np.array(result["scores"])-numerical_reference))),
            "verified_error_bound": result["certificate"]["error_bound_l1_upper_float"],
            "top_five_membership_proved": result["certificate"]["top_k_membership_proved"]})
        final = result

    policies = {}
    for name, d in [("Restart at seed", None), ("Uniform restart", [1, 1, 1]), ("Stay at sink", [0, 0, 1])]:
        _, p = solve_exact(3, [(0, 1, 1), (1, 2, 1)], [1, 0, 0], d)
        policies[name] = [str(x) for x in p]

    _, default0 = solve_exact(2, [(0, 1, 1)], [1, 0])
    _, default1 = solve_exact(2, [(0, 1, 1)], [0, 1])
    _, default_mix = solve_exact(2, [(0, 1, 1)], [1, 1])
    default_average = tuple((a+b)/2 for a,b in zip(default0, default1))
    _, fixed0 = solve_exact(2, [(0, 1, 1)], [1, 0], [1, 1])
    _, fixed1 = solve_exact(2, [(0, 1, 1)], [0, 1], [1, 1])
    fixed_average = tuple((a+b)/2 for a,b in zip(fixed0, fixed1))
    assert fixed_average == default_mix
    mixture_error = sum(abs(a-b) for a,b in zip(default_average, default_mix))

    summary = {
        "data": "deterministic synthetic directed integer-weight graphs; no external or personal data",
        "large_graph": {"nodes": n, "edges": [list(edge) for edge in edges], "personalization": v,
            "alpha": str(exact_a), "dangling": "personalization", "accuracy_work": accuracy,
            "final_scores": final["scores"], "nonzero_score_nodes": sum(x > 0 for x in final["scores"]),
            "top_five": final["certificate"]["top_nodes"], "dense_reference_certificate": reference_check},
        "dangling_policies": {"graph": "0 -> 1 -> 2; node 2 has no outgoing edges", "personalization": [1, 0, 0],
            "alpha": "17/20", "exact_scores": policies},
        "mixture_counterexample": {"nodes": 2, "edges": [[0, 1, 1]], "alpha": "17/20",
            "personalizations": [[1, 0], [0, 1]], "mixture": [1, 1],
            "average_with_personalized_dangling": list(map(str, default_average)),
            "direct_mixture_with_personalized_dangling": list(map(str, default_mix)),
            "exact_l1_difference": str(mixture_error),
            "average_with_fixed_uniform_dangling": list(map(str, fixed_average)),
            "fixed_dangling_identity": True},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    example = {"nodes": 4, "edges": [[0, 1, 3], [0, 2, 1], [1, 2, 2], [2, 0, 1]],
               "personalization": [1, 0, 0, 0], "alpha": "17/20", "tolerance": 1e-8, "top_k": 2}
    example_result = rank(Graph(example["nodes"], example["edges"]),
                          **{k:v for k,v in example.items() if k not in {"nodes", "edges"}})
    assert example_result["status"] == "converged"
    verification = {**example, "scores": example_result["scores"]}
    for name, payload in [("example.json", example), ("example-result.json", example_result), ("verification.json", verification)]:
        (args.output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8), constrained_layout=True)
    ax = axes[0, 0]
    work = [r["edge_updates"] + r["dangling_updates"] for r in accuracy]
    ax.loglog(work, [r["verified_error_bound"] for r in accuracy], color="#17786b", marker="o", label="Exact residual upper bound")
    ax.loglog(work, [r["l1_distance_to_numerical_reference"] for r in accuracy], color="#b34c35", ls="none", marker="x", label="Distance to dense numerical solve")
    ax.set(xlabel="Neighbor-update attempts (push phase only)", ylabel="L1 score error", title="Accuracy is verified after the local solve")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for offset, (name, scores) in zip([-0.25, 0, 0.25], policies.items()):
        ax.bar(np.arange(3) + offset, [float(Fraction(s)) for s in scores], width=0.23, label=name)
    ax.set(xticks=range(3), xticklabels=["Node 0 (seed)", "Node 1", "Node 2 (sink)"],
           ylabel="Exact PageRank score", title="Dangling policy changes the target ranking")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    labels = ["Average\ndefault d=v", "Direct mixture\ndefault d=v", "Average\nfixed d", "Direct mixture\nfixed d"]
    values = [float(default_average[0]), float(default_mix[0]), float(fixed_average[0]), float(default_mix[0])]
    bars = ax.bar(labels, values, color=["#b34c35", "#758da6", "#17786b", "#758da6"], width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.008, f"{value:.4f}", ha="center", fontsize=9)
    ax.set(ylabel="Node 0 score", ylim=(0, 0.44), title="Changing dangling policy can break linear mixing")
    ax.text(0.02, 0.95, f"Default-policy L1 difference = {float(mixture_error):.6f}", transform=ax.transAxes, va="top", fontsize=9)

    ax = axes[1, 1]
    ax.plot(range(n), final["scores"], color="#17786b")
    ax.axvspan(80, 119, color="0.85", alpha=0.6, label="Unreachable component")
    ax.axvline(39.5, ls=":", color="0.5")
    ax.set(xlabel="Synthetic node index", ylabel="Reserve score", title="Personalization can leave components unreachable")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Personalized PageRank: residuals, dangling nodes, and model semantics", fontsize=14)
    fig.savefig(args.output / "rank-residuals.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"accuracy_work": accuracy, "mixture_l1_difference": float(mixture_error),
                      "reachable_scored_nodes": summary["large_graph"]["nonzero_score_nodes"],
                      "top_five": summary["large_graph"]["top_five"]}, indent=2))


if __name__ == "__main__":
    main()
