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


def forward_infer(sfm: SFM, w_exo, w_ref=None):
    assert all(u in w_exo for u in sfm.exo_nodes),\
        "All exogenous nodes must be present in the valuation"
    assert nx.is_directed_acyclic_graph(sfm.graph),\
        "Forward inference is only allowed in directed acyclic graphs"
    if w_ref is None:  # use vanilla forward inference
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
        return w
    else:
        raise NotImplementedError("Implement contrastive forward inference")


def main():
    a, b, c = np.random.rand(3, 10)
    d1 = {x: y for x, y in zip(a, b)}
    d2 = {x: y for x, y in zip(d1, c)}
    print(contrast_decode(contrast_encode(d1, w_ref=d2), w_ref=d2) == d1)


if __name__ == '__main__':
    main()
