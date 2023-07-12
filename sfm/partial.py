"""
Partial inference (vanilla and contrastive)

Useful when the graph is large but the number of affected nodes is small.

Suppose the graph is very big,
so topological sort isn't worth it.
"""
from sfm.model import SFM


def partial_vfi(sfm: SFM, w_exo: dict, target_nodes: set):
    """
    Partial vanilla forward inference

    Parameters
    ----------
    sfm
    w_exo

    target_nodes : set
        A set of nodes whose values we want to infer

    Returns
    -------

    """
    target_nodes = set(target_nodes)
    assert sfm.is_directed_acyclic_graph, \
        "Forward inference is only allowed in directed acyclic graphs"
    assert frozenset(w_exo.keys()) == sfm.exo_nodes,\
        "w_exo must contain all exogenous nodes for vanilla forward inference"
    assert not target_nodes.difference(sfm.graph.nodes),\
        "all target nodes should be in the SFM"

    w = w_exo.copy()
    # perform iterative DFS because DFS reaches root/exo nodes faster
    stack = list(target_nodes)

    COUNT = 0

    while stack:
        node = stack.pop()  # pop a node from the stack

        if node not in w:
            parent_nodes = sfm.parents(node)  # get parent nodes
            if parent_nodes:    # is not root node
                unknown_parents = [p for p in parent_nodes if p not in w]
                if unknown_parents:  # there are unknown parents
                    stack.append(node)  # push node back onto the stack
                    stack.extend(unknown_parents)   # push unknown parents to the stack
                else:
                    COUNT += 1
                    # all parent values are computed
                    w[node] = sfm.functions[node](w)

    print(f"partial vfi evaluations: {COUNT}/{len(sfm.endo_nodes)}")
    return {u: w[u] for u in target_nodes}

