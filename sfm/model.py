import networkx as nx
from functools import cached_property, cache


class SFM:
    def __init__(self, graph: nx.DiGraph, domains: dict, functions: dict):
        """
        Structural functional model for causal inference.

        It is our assumption that the model itself
        (the graph structure, the domain set, and the function set)
        remains constant/immutable in the inference process.

        For composing different SFMs or learning an SFM from data,
        we always construct a new SFM.

        Parameters
        ----------
        graph
        domains
        functions
        """
        self.graph = graph
        self.domains = domains
        self.functions = functions

    @cached_property
    def exo_nodes(self):
        """
        Get a tuple of all exogenous nodes
        """
        return frozenset(node for node, in_degree in self.graph.in_degree() if in_degree == 0)

    @cached_property
    def endo_nodes(self):
        """
        Get a tuple of all endogenous nodes
        """
        return frozenset(node for node, in_degree in self.graph.in_degree() if in_degree > 0)

    @cached_property
    def topological_order(self):
        """
        Get the topological order of nodes.

        For any directed edge (u, v), node u must come before node v.

        This property is cached to save computations.
        """
        return tuple(nx.topological_sort(self.graph))

    @cached_property
    def is_directed_acyclic_graph(self):
        return nx.is_directed_acyclic_graph(self.graph)

    @cache
    def parents(self, node):
        return frozenset(self.graph.predecessors(node))

    def satisfied_by(self, w_total: dict) -> bool:
        """
        Check whether a total valuation will satisfy this SFM.

        A total valuation assigns a value to each node in the graph.

        Parameters
        ----------
        w_total: dict
            The total valuation to be checked.

        Returns
        -------
        bool
            True if w_total satisfies the SFM, False if it doesn't.

        """
        # check whether the nodes in w_total are also in the graph
        assert all(node in self.graph for node in w_total), "Some nodes in w_total isn't in the graph"

        # check whether all values are in the domain
        # omitted, this might not be necessary for python programs

        for node, value in w_total.items():
            if self.graph.in_degree(node) > 0:  # the node has at least 1 parent node
                # valuation over the parents of the node
                w_parent = {p: w_total[p] for p in self.graph.predecessors(node)}
                if value != self.functions[node](w_parent):
                    # this structural function is not satisfied by the valuation
                    return False
        return True

    def all_violations(self, w_total):
        """
        Get all nodes whose valuation violates

        This provides more information than `SFM.satisfied_by`,
        making it useful for debugging.

        Parameters
        ----------
        w_total: dict
            The total valuation to evaluate

        Returns
        -------
        A list of (node, expected, actual) tuples,
        where the expected value is computed based on parent valuation
        and structural function.

        """
        violations = []
        for node, actual in w_total.items():
            if self.graph.in_degree(node) > 0:  # the node has at least 1 parent node
                # valuation over the parents of the node
                w_parent = {p: w_total[p] for p in self.graph.predecessors(node)}
                expected = self.functions[node](w_parent)
                if expected != actual:
                    # this structural function is not satisfied by the valuation
                    violations.append((node, expected, actual))
        return violations


def main():
    return


if __name__ == '__main__':
    main()
