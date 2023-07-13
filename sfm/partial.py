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


def partial_cfi(sfm: SFM, w0: dict, w1_changed_exo: dict, target_nodes: set):
    target_nodes = set(target_nodes)
    assert sfm.is_directed_acyclic_graph, \
        "Forward inference is only allowed in directed acyclic graphs"
    if any(not sfm.is_exo_node(u) for u in w1_changed_exo):
        raise ValueError("w1_changed_exo must only contain exo-nodes in the SFM")
    if any(not sfm.graph.has_node(u) for u in target_nodes):
        raise ValueError("all target nodes should be in the SFM")

    # changed[u]==True means changed (confirmed)
    # changed[u]==False means not changed
    # u not in changed means we don't know
    changed = {u: w1_changed_exo[u] != w0[u] for u in w1_changed_exo}
    w1_c = {u: w1_changed_exo[u] for u in w1_changed_exo if changed[u]}

    stack = list(target_nodes)

    COUNT = 0

    while stack:
        node = stack.pop()  # Pop a node from the stack
        if node not in changed:     # don't know
            if sfm.is_exo_node(node):
                # all changed exo-nodes have been initialized in the beginning
                changed[node] = False
            else:
                unknown_parents = [p for p in sfm.parents(node) if p not in changed]
                if unknown_parents:
                    stack.append(node)
                    stack.extend(unknown_parents)
                else:
                    w_parents = {}
                    any_parent_changed = False
                    for p in sfm.parents(node):
                        if changed[p]:
                            any_parent_changed = True
                            w_parents[p] = w1_c[p]
                        else:
                            w_parents[p] = w0[p]
                    if any_parent_changed:
                        COUNT += 1
                        value = sfm.functions[node](w_parents)
                        if value != w0[node]:
                            changed[node] = True
                            w1_c[node] = value
                        else:
                            changed[node] = False
                    else:
                        # same parents, same child
                        changed[node] = False
    print(f"partial cfi evaluations: {COUNT}/{len(sfm.endo_nodes)}")
    return {u: w1_c[u] if changed[u] else w0[u] for u in target_nodes}

