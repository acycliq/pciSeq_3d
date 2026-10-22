"""
The Zero class has no scale factors.

The Zero column of the reference is all zeros, so theta and gamma have
nothing to scale there. They used to come out above 1 anyway (gamma as
(rSpot + reads) / rSpot, theta a bit above 1 on cells with a few dozen reads)
and the spot step read that as a bonus for a Zero cell holding reads. Now the
Zero column of theta is pinned at 1 and the Zero column of gamma at 1 (log 0),
where they are computed, so the spot step needs no special case.
"""
import numpy as np


def test_zero_column_of_theta_and_gamma_is_one(minimal_varbayes):
    vb = minimal_varbayes
    vb.initialise_state()

    # straight after the initialisation
    assert (vb.cells.theta_bar[:, -1] == 1.0).all()

    # and after a full round of the updates, with reads on the cells
    vb.geneCount_upd()
    vb.theta_upd()
    vb.gamma_upd()

    assert (vb.cells.theta_bar[:, -1] == 1.0).all()
    assert (vb.spots.gamma_bar[:, :, -1] == 1.0).all()
    assert (vb.spots.log_gamma_bar[:, :, -1] == 0.0).all()

    # the real classes are not touched by it
    assert not np.allclose(vb.cells.theta_bar[:, :-1], 1.0)
    assert not np.allclose(vb.spots.log_gamma_bar[:, :, :-1], 0.0)
