"""Rank or independently verify a candidate against a strict JSON graph model."""

import argparse
import json
from pathlib import Path
import sys

from .core import Graph, rank, verify


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["rank", "verify"])
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.input.open("rb") as handle:
            raw = handle.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("input exceeds 1 MiB")
        request = json.loads(raw, object_pairs_hook=_pairs)
        allowed = {"nodes", "edges", "personalization", "dangling", "alpha", "tolerance", "top_k"}
        allowed.add("max_pushes" if args.operation == "rank" else "scores")
        required = {"nodes", "edges"} | ({"scores"} if args.operation == "verify" else set())
        if not isinstance(request, dict) or set(request) - allowed or required - set(request):
            raise ValueError("missing or unknown request fields")
        graph = Graph(request["nodes"], request["edges"])
        kwargs = {key: request[key] for key in request if key not in {"nodes", "edges", "scores"}}
        kwargs.setdefault("top_k", min(3, graph.nodes))
        if args.operation == "rank":
            result = rank(graph, **kwargs)
            success = result["certificate"]["tolerance_met"]
        else:
            result = verify(graph, request["scores"], **kwargs)
            success = result["tolerance_met"]
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if success else 3
    except (OSError, ValueError, TypeError, ArithmeticError) as exc:
        print(f"rank-residuals: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
