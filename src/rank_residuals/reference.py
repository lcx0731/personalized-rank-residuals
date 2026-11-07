"""Independent small exact reference: dense rational Gaussian elimination."""

from fractions import Fraction


def solve_exact(nodes, edges, personalization=None, dangling=None, alpha=Fraction(17, 20)):
    """Solve <=12 nodes from raw records, without using Graph or push/verify code.

    Intended for already validated small inputs in tests and deterministic demos.
    Returns the dense stochastic transition and the rational stationary vector.
    """
    if not 1 <= nodes <= 12:
        raise ValueError("exact reference supports 1 to 12 nodes")
    a = Fraction(alpha)
    if not 0 <= a < 1:
        raise ValueError("reference alpha must be in [0,1)")
    raw_v = [1] * nodes if personalization is None else personalization
    raw_d = raw_v if dangling is None else dangling
    v = [Fraction(x, sum(raw_v)) for x in raw_v]
    d = [Fraction(x, sum(raw_d)) for x in raw_d]
    adjacency = [[0] * nodes for _ in range(nodes)]
    for source, target, weight in edges:
        adjacency[source][target] += weight
    transition = []
    for row in adjacency:
        total = sum(row)
        transition.append([Fraction(x, total) for x in row] if total else list(d))
    system = [[Fraction(i == j) - a * transition[j][i] for j in range(nodes)]
              + [(1-a)*v[i]] for i in range(nodes)]
    for column in range(nodes):
        pivot = next(i for i in range(column, nodes) if system[i][column])
        system[column], system[pivot] = system[pivot], system[column]
        divisor = system[column][column]
        system[column] = [x / divisor for x in system[column]]
        for i in range(nodes):
            if i != column:
                multiple = system[i][column]
                system[i] = [x - multiple*y for x, y in zip(system[i], system[column])]
    return transition, tuple(row[-1] for row in system)
