"""
Generate random SFMs using randomly initialized functions
and random graphs.
"""
import numpy as np
import networkx as nx


class RandomQuadratic:
    def __init__(self, nodes):
        """
        A quadratic function that takes in a dictionary of node-float mappings,
        and returns a float.
        With vector input `x = [x1, x2, ..., xn]`,
        the formula is of the form:
        `f(x) = x^T A x + b x + c`
        The weights of the quadratic function are randomly initialized
        from a standard normal distribution (mean=0, standard deviation=1)

        This is useful for generating random SFMs with
        random graph connections and random functions,
        so the correctness of inference algorithms can be easily tested.

        Parameters
        ----------
        nodes: list
            List of parent nodes
        """
        self.nodes = tuple(nodes)
        n = len(self.nodes)
        self.A = np.random.randn(n, n)  # weights for second-order terms
        self.b = np.random.randn(n)     # weights for first-order terms
        self.c = np.random.randn()      # constant bias

    def __call__(self, w: dict) -> float:
        """
        Compute the quadratic function based on parent valuation.

        Parameters
        ----------
        w: dict
            The parent valuation

        Returns
        -------
        float
            The computed result as a floating point number.
        """
        # convert parent valuation into a vector
        x = np.array([w[node] for node in self.nodes])
        return x.T @ self.A @ x + self.b @ x + self.c


class RandomCongruence:
    def __init__(self, nodes, m):
        """
        A linear congruence function that takes in a dictionary of node-integer mappings,
        and returns an integer.
        The formula is of the form:
        `f(x1, x2, ..., xn) = a1 x1 + a2 x2 + ... + an xn + c (mod m)`

        The weights are randomly initialized as integers from 1 to m-1 inclusive.

        Parameters
        ----------
        nodes: list
            List of parent nodes

        m: int
            The modulo/divisor parameter m in "output mod m".
        """
        self.nodes = tuple(nodes)
        self.m = int(m)
        n = len(self.nodes)
        # weights are also random integers from 0 to m-1 inclusive
        self.a = tuple(int(a) for a in np.random.randint(1, m, size=n))
        self.c = int(np.random.randint(1, m))

    def __call__(self, w: dict) -> int:
        x = [w[node] for node in self.nodes]
        s = sum(a*value for a, value in zip(self.a, x)) + self.c
        return s % self.m


