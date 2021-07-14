import math
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pytest
import toolz
from numpy.typing import NDArray

from sgkit.stats.ld import _maximal_independent_set as numba_mis


def to_vertex_ids(g: nx.Graph) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    g = np.array(sorted(g.edges))
    return g[:, 0], g[:, 1]


def plink_mis(
    idi: NDArray[np.int_], idj: NDArray[np.int_], cmp: Optional[NDArray[np.int_]] = None
) -> List[int]:
    # Direct port of https://groups.google.com/forum/#!msg/plink2-users/w5TuZo2fgsQ/WbNnE16_xDIJ
    if cmp is None:
        cmp = np.zeros(len(idi))
    lost = set()
    grps = toolz.groupby(lambda p: p[0], list(zip(idi, idj, cmp)))
    for i in sorted(grps.keys()):
        if i in lost:
            continue
        for t in sorted(grps[i]):
            j, c = t[1:]
            if j <= i:
                continue
            if c < 0:
                lost.add(i)
                break
            else:
                lost.add(j)
    return sorted(lost)


mis_fns = [numba_mis, plink_mis]


@pytest.mark.parametrize("mis", mis_fns)
@pytest.mark.parametrize("n", [2, 5, 25])
def test_star_graph(mis, n):
    # There are n+1 nodes in the resulting graph
    idi, idj = to_vertex_ids(nx.star_graph(n))
    # Favoring non-center node (which is the first)
    # results in only middle node lost
    idx = mis(idi, idj, cmp=np.full(n, -1))
    assert len(idx) == 1
    # Favoring center node results in all others lost
    idx = mis(idi, idj, cmp=np.full(n, 1))
    assert len(idx) == n


@pytest.mark.parametrize("mis", mis_fns)
def test_path_graph(mis):
    # Graph is 3 nodes with A-B and B-C
    idi, idj = to_vertex_ids(nx.path_graph(3))
    # First and third should be kept with no comparison
    idx = mis(idi, idj)
    assert idx == [1]
    # With comparisons favoring later nodes, only third is kept
    idx = mis(idi, idj, cmp=np.array([-1, -1]))
    assert idx == [0, 1]
    # With comparisons favoring earlier nodes, middle node is lost
    idx = mis(idi, idj, cmp=np.array([1, 1]))
    assert idx == [1]
    # With middle node largest, first and third are lost
    idx = mis(idi, idj, cmp=np.array([-1, 1]))
    assert idx == [0, 2]


@pytest.mark.parametrize("mis", mis_fns)
def test_disconnected_graph(mis):
    # Node 2 is connected to 3 but 0, 1, and 4 have no edges
    idi, idj = np.array([0, 1, 2, 2, 3, 4]), np.array([0, 1, 2, 3, 3, 4])
    idx = mis(idi, idj)
    assert idx == [3]


@pytest.mark.parametrize(
    "gfn",
    [
        nx.ladder_graph,
        nx.circular_ladder_graph,
        nx.binomial_tree,
        nx.wheel_graph,
        nx.complete_graph,
    ],
)
@pytest.mark.parametrize("n", [2, 10, 25])
def test_random_graphs(gfn, n):
    # For several more complex graph types, make sure
    # the plink algo is equal to the unrolled numba version
    if gfn == nx.binomial_tree:
        n = int(math.log(n, 2))
    idi, idj = to_vertex_ids(gfn(n))
    idx1 = numba_mis(idi, idj)
    idx2 = plink_mis(idi, idj)
    assert idx1 == idx2


def test_unsorted_edges():
    idi, idj = to_vertex_ids(nx.complete_graph(10))
    idi = idi[::-1]
    with pytest.raises(ValueError):
        numba_mis(idi, idj)
