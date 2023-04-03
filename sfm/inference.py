import numpy as np
import networkx as nx

from sfm.model import SFM


def contrast_encode(w: dict, w_ref: dict):
    """

    Parameters
    ----------
    w: dict

    w_ref: dict

    Returns
    -------

    """
    w_partial = {}
    for node, new_value in w.items():
        if node not in w_ref:
            raise ValueError(f"Node {node} doesn't exist in reference valuation")
        if new_value != w_ref[node]:
            w_partial[node] = new_value
    return w_partial


def contrast_decode(w: dict, w_ref: dict):
    """

    Parameters
    ----------
    w: dict

    w_ref: dict

    Returns
    -------

    """
    w_total = {}
    for node in w_ref:
        if node in w:
            w_total[node] = w[node]
        else:
            w_total[node] = w_ref[node]
    return w_total


def vanilla_forward_infer(sfm: SFM, w_exo):
    assert sfm.is_directed_acyclic_graph, \
        "Forward inference is only allowed in directed acyclic graphs"
    assert frozenset(w_exo.keys()) == sfm.exo_nodes,\
        "w_exo must contain exactly all exogenous nodes for total forward inference"
    w = w_exo.copy()    # shallow copy to initialize output valuation w
    for node in sfm.topological_order:
        if node not in w:
            # this method comes from the definition of forward inference
            w_parent = {parent: w[parent] for parent in sfm.graph.predecessors(node)}
            w[node] = sfm.functions[node](w_parent)
            # this second method is slightly faster because it doesn't involve
            # creating a parent valuation dictionary every time.
            # assuming the structural function doesn't check the length of w dict.
            # w[node] = sfm.functions[node](w)
    return w    # induced total valuation


def contrastive_forward_infer(sfm: SFM, w_exo: dict, w_ref: dict):
    assert sfm.is_directed_acyclic_graph, \
        "Forward inference is only allowed in directed acyclic graphs"
    assert all(u in sfm.exo_nodes for u in w_exo), \
        "All nodes in w_exo must be exogenous in the SFM"
    assert all(u in w_ref for u in sfm.graph.nodes), \
        "w_ref must contain valuation over all nodes"
    changed = {}  # node -> whether the value changes
    for u in sfm.graph.nodes:
        if u in w_exo and w_exo[u] != w_ref[u]:
            changed[u] = True
        else:
            changed[u] = False
    w = {}

    COUNT = 0   # debug: count the number of function evaluations

    for u in sfm.topological_order:
        recompute = changed[u]
        # recompute node u if u or any of its parents have changed
        for parent in sfm.graph.predecessors(u):
            recompute = recompute or changed[parent]
        if recompute:
            if u in sfm.exo_nodes:
                w[u] = w_exo[u]
            else:   # u is endogenous
                f = sfm.functions[u]
                # w_parent = {parent: w[parent] for parent in sfm.graph.predecessors(u)}
                # new_val = f(w_parent)
                new_val = f(w)
                COUNT += 1
                if new_val != w_ref[u]:
                    w[u] = new_val
                    changed[u] = True
                else:
                    w[u] = w_ref[u]
        else:   # don't recompute
            w[u] = w_ref[u]
    print(f"Contrastive evaluations: {COUNT}/{len(sfm.endo_nodes)}")
    return w


def main():
    a, b, c = np.random.rand(3, 10)
    d1 = {x: y for x, y in zip(a, b)}
    d2 = {x: y for x, y in zip(d1, c)}
    print(contrast_decode(contrast_encode(d1, w_ref=d2), w_ref=d2) == d1)


if __name__ == '__main__':
    main()
