"""
Checks the starting gamma and the starting theta against the code they had
before.

init_gamma used to fill four cells x genes x classes arrays. Now it fills one,
gamma_bar, and leaves the other three empty until gamma_upd makes them in the
first pass of the loop. init_theta used to run an einsum over the whole
gamma_bar. gamma_bar is one number everywhere at that point, so now it takes
that number out of the sum. The older code is kept below, in
_init_gamma_before and _init_theta_before.

gamma_bar must be the very same array. theta cannot be bit for bit the same,
the numbers are added up in a different order, so it is checked to float32
rounding.
"""
import numpy as np
import scipy.special


def _init_gamma_before(a, b, dim):
    """NOTE: this is Spots.init_gamma before it was cut down to gamma_bar only.
    It is not used anywhere in pciSeq anymore, it lives here only to check the
    new code against. Returns post_shape, post_rate, gamma_bar, log_gamma_bar."""
    post_shape = np.ones(dim, dtype=np.float32) * a
    post_rate = np.ones(dim, dtype=np.float32) * b
    gamma_bar = np.ones(dim, dtype=np.float32) * (a / b)
    log_gamma_bar = np.ones(dim, dtype=np.float32) * (scipy.special.psi(a) - np.log(b))
    return post_shape, post_rate, gamma_bar, log_gamma_bar


def _init_theta_before(vb, gamma_bar):
    """NOTE: this is the rate of theta as VarBayes.init_theta worked it out
    before, with the einsum over the whole gamma_bar. Kept here only to check
    the new code against."""
    mu = vb.single_cell.mean_expression_adj + vb.config['SpotReg']
    area_factor = vb.cells.ini_cell_props['area_factor']
    eta_bar = vb.genes.eta_bar
    return np.einsum('c, cgk, g, gk -> ck', area_factor, gamma_bar, eta_bar, mu) + vb.config['rTheta']


def test_gamma_bar_is_the_same_array(minimal_varbayes):
    vb = minimal_varbayes
    a = b = vb.config['rSpot']
    dim = [vb.nC, vb.nG, vb.nK]
    _, _, gamma_bar_0, _ = _init_gamma_before(a, b, dim)

    vb.spots.init_gamma(a, b, dim)
    assert vb.spots.gamma_bar.dtype == np.float32
    assert np.array_equal(vb.spots.gamma_bar, gamma_bar_0)


def test_gamma_bar_is_the_same_when_shape_and_rate_differ(minimal_varbayes):
    # a/b is 1 in a real run, so try one where it is not
    vb = minimal_varbayes
    dim = [vb.nC, vb.nG, vb.nK]
    _, _, gamma_bar_0, _ = _init_gamma_before(3.0, 7.0, dim)
    vb.spots.init_gamma(3.0, 7.0, dim)
    assert np.array_equal(vb.spots.gamma_bar, gamma_bar_0)


def test_the_other_three_start_empty(minimal_varbayes):
    vb = minimal_varbayes
    vb.spots.init_gamma(2.0, 2.0, [vb.nC, vb.nG, vb.nK])
    assert vb.spots._post_shape is None
    assert vb.spots._post_rate is None
    assert vb.spots.log_gamma_bar is None


def test_gamma_upd_makes_all_four(minimal_varbayes):
    # the reason the three can start empty: the first gamma_upd of the loop
    # makes them, and nothing before it in the loop asks for them
    vb = minimal_varbayes
    vb.initialise_state()
    vb._step_times = {}
    vb.iter_num = 0
    for name in ('geneCount_upd_numba', 'rho_upd', 'eta_upd_numba', 'theta_upd_numba', 'gamma_upd_numba'):
        getattr(vb, name)()
    shape = (vb.nC, vb.nG, vb.nK)
    assert vb.spots._post_rate.shape == shape
    assert vb.spots.gamma_bar.shape == shape
    assert vb.spots.log_gamma_bar.shape == shape
    assert vb.spots._post_shape is not None


def _theta_before_and_now(vb):
    vb.initialise_state()
    theta_now = vb.cells.theta_bar.copy()

    alpha = vb.cells.ini_gene_counts + vb.config['rTheta'] - 1
    vb.cells.calc_theta(alpha, _init_theta_before(vb, vb.spots.gamma_bar))
    theta_before = vb.cells.theta_bar.copy()
    return theta_before, theta_now


def test_theta_starts_the_same(minimal_varbayes):
    theta_before, theta_now = _theta_before_and_now(minimal_varbayes)
    assert theta_now.dtype == theta_before.dtype == np.float32
    assert theta_now.shape == theta_before.shape
    np.testing.assert_allclose(theta_now, theta_before, rtol=1e-6)


def test_theta_starts_the_same_when_gamma_does_not_start_at_one(minimal_varbayes, monkeypatch):
    # make init_gamma start gamma at 3/7, so the test fails if init_theta
    # forgets to carry the starting value of gamma into the sum
    vb = minimal_varbayes
    init_gamma = vb.spots.init_gamma
    monkeypatch.setattr(vb.spots, 'init_gamma', lambda a, b, dim: init_gamma(3.0, 7.0, dim))
    theta_before, theta_now = _theta_before_and_now(vb)
    assert np.allclose(vb.spots.gamma_bar, 3.0 / 7.0)
    np.testing.assert_allclose(theta_now, theta_before, rtol=1e-6)
