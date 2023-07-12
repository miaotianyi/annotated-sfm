import numpy as np
import networkx as nx

from sfm.model import SFM


def delta_encode(w1: dict, w0: dict):
    """
    Return the node-value pairs in w1 that differ from w0.

    Parameters
    ----------
    w1: dict

    w0: dict

    Returns
    -------

    """
    w1_change = {}
    for node, new_value in w1.items():
        if node not in w0 or new_value != w0[node]:
            w1_change[node] = new_value
    return w1_change


def delta_decode(w1_change: dict, w0: dict):
    """

    Parameters
    ----------
    w1_change: dict

    w0: dict

    Returns
    -------

    """
    w1 = w0.copy()
    for node in w1_change:
        w1[node] = w1_change[node]
    return w1


def vfi(sfm: SFM, w_exo):
    """
    Vanilla forward inference

    Parameters
    ----------
    sfm : SFM
        The structural functional model

    w_exo : dict
        The assignment over all exo-nodes.

    Returns
    -------
    w : dict
        The induced complete assignment
    """
    assert sfm.is_directed_acyclic_graph, \
        "Forward inference is only allowed in directed acyclic graphs"
    assert frozenset(w_exo.keys()) == sfm.exo_nodes,\
        "w_exo must contain all exogenous nodes for non-contrastive forward inference"
    w = w_exo.copy()    # shallow copy to initialize output assignment w
    for node in sfm.topological_order:
        if node not in w:
            # this method comes from the definition of forward inference
            # w_parent = {parent: w[parent] for parent in sfm.parents(node)}
            # w[node] = sfm.functions[node](w_parent)
            # this second method is slightly faster because it doesn't involve
            # creating a parent assignment dictionary every time.
            # assuming the structural function doesn't check the length of w dict.
            w[node] = sfm.functions[node](w)
    return w


def cfi(sfm: SFM, w1_change_exo: dict, w0: dict):
    """
    Contrastive forward inference.

    Parameters
    ----------
    sfm
    w1_change_exo
    w0

    Returns
    -------

    """
    assert sfm.is_directed_acyclic_graph, \
        "Forward inference is only allowed in directed acyclic graphs"
    assert all(u in sfm.exo_nodes for u in w1_change_exo), \
        "All nodes in w_exo must be exogenous in the SFM"
    assert all(u in w0 for u in sfm.graph.nodes), \
        "w_ref must contain assignment over all nodes"
    changed = {}  # node -> whether the value changes
    for u in sfm.graph.nodes:
        if u in w1_change_exo and w1_change_exo[u] != w0[u]:
            changed[u] = True
        else:
            changed[u] = False
    w1 = {}

    COUNT = 0   # debug: count the number of function evaluations

    for u in sfm.topological_order:
        recompute = changed[u]
        # recompute node u if u or any of its parents have changed
        for parent in sfm.parents(u):
            recompute = recompute or changed[parent]
        if recompute:
            if sfm.is_exo_node(u):
                w1[u] = w1_change_exo[u]
            else:   # u is endogenous
                f = sfm.functions[u]
                # w1_parent = {parent: w1[parent] for parent in sfm.parents(u)}
                # new_val = f(w1_parent)
                new_val = f(w1)
                COUNT += 1
                if new_val != w0[u]:
                    w1[u] = new_val
                    changed[u] = True
                else:
                    w1[u] = w0[u]
        else:   # don't recompute
            w1[u] = w0[u]
    print(f"CFI evaluations: {COUNT}/{len(sfm.endo_nodes)}")
    return w1


def main():
    a, b, c = np.random.rand(3, 10)
    d1 = {x: y for x, y in zip(a, b)}
    d2 = {x: y for x, y in zip(d1, c)}
    print(delta_decode(delta_encode(d1, w0=d2), w0=d2) == d1)


if __name__ == '__main__':
    main()
