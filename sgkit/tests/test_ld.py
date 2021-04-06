import numpy as np

from sgkit.stats.ld import rogers_huff_r2_between, rogers_huff_r_between


def test_rogers_huff_r_between():
    gna = np.array([0, 1, 2])
    gnb = np.array([0, 1, 2])
    assert rogers_huff_r_between(gna, gnb) == 1.0
    assert rogers_huff_r2_between(gna, gnb) == 1.0

    gna = np.array([0, 1, 2])
    gnb = np.array([2, 1, 0])
    assert rogers_huff_r_between(gna, gnb) == -1.0
    assert rogers_huff_r2_between(gna, gnb) == 1.0

    gna = np.array([0, 0, 0])
    gnb = np.array([1, 1, 1])
    assert np.isnan(rogers_huff_r_between(gna, gnb))
    assert np.isnan(rogers_huff_r2_between(gna, gnb))
