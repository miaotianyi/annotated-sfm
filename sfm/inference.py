import numpy as np
import networkx as nx


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


def main():
    a, b, c = np.random.rand(3, 10)
    d1 = {x: y for x, y in zip(a, b)}
    d2 = {x: y for x, y in zip(d1, c)}
    print(contrast_decode(contrast_encode(d1, w_ref=d2), w_ref=d2) == d1)


if __name__ == '__main__':
    main()
