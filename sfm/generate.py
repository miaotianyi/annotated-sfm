"""
Generate random SFMs using randomly initialized functions
and random graphs.
"""
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from functools import partial


from sfm.model import SFM


class RandomLinear:
    """
    Generate a random linear function
    with weights and bias initialized from unit normal N(0, 1).
    The input is a {node: value} dictionary.
    """
    def __init__(self, nodes):
        self.nodes = tuple(nodes)
        n = len(self.nodes)
        self.a = np.random.randn(n)  # weights for first-order terms
        self.b = np.random.randn()  # constant bias

    def __call__(self, w: dict) -> float:
        # convert parent assignment into a vector
        x = np.array([w[node] for node in self.nodes])
        return self.a @ x + self.b


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

        Warning: the quadratic nature of this function could result in
        divergence (infinity/nan) when such functions are repeatedly applied.
        Use linear models instead.

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
        Compute the quadratic function based on parent assignment.

        Parameters
        ----------
        w: dict
            The parent assignment

        Returns
        -------
        float
            The computed result as a floating point number.
        """
        # convert parent assignment into a vector
        x = np.array([w[node] for node in self.nodes])
        return x.T @ self.A @ x + self.b @ x + self.c


class RandomCongruence:
    def __init__(self, nodes, m):
        """
        A linear congruence function that takes in a {node: int value} dictionary
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


def undirected_to_dag(G: nx.Graph, order=None):
    """
    Convert an undirected graph into a directed acyclic graph.
    An order can be imposed on G's nodes so that the DAG is consistent with it.
    Otherwise, the order will be a random permutation of nodes.
    """
    G2 = nx.DiGraph()
    if order is None:
        order = np.random.permutation(G.nodes)
    order_dict = {x: i for i, x in enumerate(order)}
    for u, v in G.edges:
        if order_dict[u] < order_dict[v]:
            G2.add_edge(u, v)
        else:
            G2.add_edge(v, u)
    return G2


def random_dag(n, p):
    """
    Generate a random directed acyclic graph (DAG) with the following steps:
    1. Generate an Erdos-Renyi random graph with (n, p), which is undirected.
        Here `n` is the number of nodes and `p` is the probability that an edge
        exists in a random pair of nodes.
        If `p > ln(n)/n`, this graph is almost surely connected, see Erdos&Renyi paper.
    2. Generate an ordering of nodes using a random permutation.
    3. Assign directions to edges, such that only earlier nodes (in the ordering)
        can point to later nodes.
        This ensures the resulting directed graph is acyclic.

    Parameters
    ----------
    n: int
        Number of nodes

    p: float
        Probability of an edge existing between any 2 nodes.
        Between 0 and 1.

    Returns
    -------
    nx.DiGraph
        The generated DAG.
    """
    return undirected_to_dag(nx.fast_gnp_random_graph(n, p))


class RandomSFM(SFM):
    def __init__(self, n, p, function_class=RandomLinear):
        graph = random_dag(n, p)
        super().__init__(graph=graph, domains={}, functions={})
        for node in self.endo_nodes:
            # get the parents of node
            parents = list(self.graph.predecessors(node))
            # randomly initialize structural function associated with node
            # initialization needs to know node's parent nodes
            self.functions[node] = function_class(parents)


def plot_dag(graph: nx.DiGraph):
    # we need to set a new "subset" attribute for each node
    # a shallow copy so the subset attribute isn't added in the original graph
    graph = graph.copy()

    # set group based on each topological generation
    # (no edges within each group)
    subset_dict = {}
    for group_idx, node_list in enumerate(nx.topological_generations(graph)):
        for u in node_list:
            subset_dict[u] = group_idx
    nx.set_node_attributes(graph, subset_dict, name="subset")
    # use multipartite layout for each group of nodes
    # so all edges point from left to right
    pos = nx.multipartite_layout(graph)
    # randomly perturb y positions so it's harder to have overlapping edges
    for node in pos:
        pos[node][1] += np.random.randn() * 0.05

    # use tableau palette to color the edges
    # so it's visually easier to track edge connections
    palette = tuple(mcolors.TABLEAU_COLORS.keys())
    edge_color = [palette[i % len(palette)] for i in range(len(graph.edges))]

    # set the curvature (rad) for each edge
    # So edges in the upper half are convex (n-shaped)
    # and edges in the lower half are concave (u-shaped)
    # The average y coordinate from multipartite_layout is 0
    # so an edge is in the upper half if its average y > 0; vice versa
    rad_dict = {}
    for u, v in graph.edges:
        mean_y = (pos[u][1] + pos[v][1]) / 2
        rad_dict[(u, v)] = -0.2 if mean_y > 0 else 0.2
    nx.set_edge_attributes(graph, rad_dict, "rad")

    node_size = 1500    # define node size

    nx.draw_networkx_nodes(graph, pos, node_color="lightblue", node_size=node_size)
    nx.draw_networkx_labels(graph, pos)

    for i, edge in enumerate(graph.edges(data=True)):
        nx.draw_networkx_edges(graph, pos, edgelist=[(edge[0], edge[1])],
                               connectionstyle=f'arc3, rad = {edge[2]["rad"]}',
                               node_size=node_size,
                               edge_color=palette[i % len(palette)])

    plt.axis("off")

    # nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=1500,
    #         connectionstyle="arc3, rad=0.2", edge_color=edge_color)
    plt.show()


def main():
    # create and visualize a random DAG
    g1 = random_dag(10, 0.5)
    print(g1)
    print(tuple(nx.topological_sort(g1)))
    print(list(nx.topological_generations(g1)))
    plot_dag(g1)

    # create and visualize a random SFM
    sfm1 = RandomSFM(10, 0.5, partial(RandomCongruence, m=5))
    plot_dag(sfm1.graph)

    # create and visualize a random SFM
    sfm2 = RandomSFM(10, 0.5, RandomQuadratic)
    plot_dag(sfm2.graph)


if __name__ == '__main__':
    main()
