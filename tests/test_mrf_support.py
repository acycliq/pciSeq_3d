"""
Checks the mrf support / calc_mrf split did not change anything.

calc_mrf used to do the neighbour weighting and the beta multiplication in one
go. It got split so the cap can get at the support with no beta folded in yet.
This pins down that calc_mrf still comes out the same, and that the weights
behave the way the docstring says they do.
"""
import numpy as np


def test_calc_mrf_is_support_times_beta(minimal_varbayes):
    vb = minimal_varbayes
    vb.initialise_state()

    support = vb.cells.mrf_support()
    mrf = vb.cells.calc_mrf()

    np.testing.assert_allclose(mrf, support * vb.config["mrf_beta"],
                               rtol=0, atol=0)


def test_support_does_not_depend_on_beta(minimal_varbayes):
    """The support is beta-free, so changing beta must not move it."""
    vb = minimal_varbayes
    vb.initialise_state()

    support_before = vb.cells.mrf_support().copy()
    vb.config["mrf_beta"] = vb.config["mrf_beta"] * 7.0
    support_after = vb.cells.mrf_support()

    np.testing.assert_allclose(support_after, support_before, rtol=0, atol=0)


def test_weights_sum_to_nNeighbors_and_fall_off_with_distance():
    """The gaussian weights, on their own, straight out of the formula.

    Row sums to nNeighbors so the scale matches zeta, and a nearer neighbour
    always counts for more than a further one.
    """
    dist = np.array([[30.0, 36.0, 45.0, 52.0, 60.0]])
    nN = dist.shape[1]

    sigma = np.median(dist, axis=1, keepdims=True)
    w = np.exp(-(dist / sigma) ** 2)
    w = w / w.sum(axis=1, keepdims=True) * nN

    np.testing.assert_allclose(w.sum(axis=1), nN, rtol=1e-12)
    assert (np.diff(w[0]) < 0).all(), "weights must drop off as distance grows"
    assert w.max() < nN, "no single neighbour may take the whole row"


def test_similarity_pair_pools_the_support(minimal_varbayes):
    """A pair gives both sister classes the sum of their support, the rest stay put."""
    vb = minimal_varbayes
    vb.initialise_state()

    names = list(vb.cells.class_names)
    ia, ib = names.index("Type_A"), names.index("Type_B")

    plain = vb.cells.mrf_support().copy()
    vb.config["similarity_pairs"] = [("Type_A", "Type_B")]
    pooled = vb.cells.mrf_support()

    summed = plain[:, ia] + plain[:, ib]
    np.testing.assert_allclose(pooled[:, ia], summed, rtol=1e-12)
    np.testing.assert_allclose(pooled[:, ib], summed, rtol=1e-12)

    others = [k for k in range(len(names)) if k not in (ia, ib)]
    np.testing.assert_allclose(pooled[:, others], plain[:, others], rtol=0, atol=0)


def test_similarity_pair_with_a_typo_fails(minimal_varbayes):
    import pytest

    vb = minimal_varbayes
    vb.initialise_state()
    vb.config["similarity_pairs"] = [("Type_A", "Type_Typo")]
    with pytest.raises(ValueError, match="Type_Typo"):
        vb.cells.mrf_support()
