import allel
import numpy as np
import numpy.testing as npt

from sgkit.stats.ld import rogers_huff_r2_between, rogers_huff_r_between


def test_rogers_huff_r_between():
    gna = np.array([[0, 1, 2]])
    gnb = np.array([[0, 1, 2]])
    npt.assert_allclose(rogers_huff_r_between(gna[0], gnb[0]), 1.0, rtol=1e-06)
    npt.assert_allclose(rogers_huff_r2_between(gna[0], gnb[0]), 1.0, rtol=1e-06)
    npt.assert_allclose(
        allel.rogers_huff_r_between(gna, gnb),
        rogers_huff_r_between(gna[0], gnb[0]),
        rtol=1e-06,
    )

    gna = np.array([[0, 1, 2]])
    gnb = np.array([[2, 1, 0]])
    npt.assert_allclose(rogers_huff_r_between(gna[0], gnb[0]), -1.0, rtol=1e-06)
    npt.assert_allclose(rogers_huff_r2_between(gna[0], gnb[0]), 1.0, rtol=1e-06)
    npt.assert_allclose(
        allel.rogers_huff_r_between(gna, gnb),
        rogers_huff_r_between(gna[0], gnb[0]),
        rtol=1e-06,
    )

    gna = np.array([[0, 0, 0]])
    gnb = np.array([[1, 1, 1]])
    assert np.isnan(rogers_huff_r_between(gna[0], gnb[0]))
    assert np.isnan(rogers_huff_r2_between(gna[0], gnb[0]))
    assert np.isnan(allel.rogers_huff_r_between(gna, gnb))
