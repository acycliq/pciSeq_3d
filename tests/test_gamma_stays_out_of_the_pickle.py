"""
The big gamma array must not end up in the pickle.

gamma_bar is cells by genes by classes, about 790 MB on a dataset like espio, so
Spots drops it (and its log) when it gets pickled. That did not help for a long time:
gamma_upd also kept the very same array under a second name, my_gamma_bar, "only for
debugging", and that copy sailed straight into the file. Nothing ever read it.

So this checks the promise itself rather than one attribute name: after a pass of the
loop, nothing on the pickled spots has the shape of gamma_bar. The live object keeps
its gamma_bar, the loop and the diagnostics db both need it.
"""
import pickle

import numpy as np


def _one_pass(vb):
    vb.initialise_state()
    vb._step_times = {}
    vb.iter_num = 0
    for name in ('geneCount_upd', 'rho_upd', 'eta_upd', 'theta_upd', 'gamma_upd',
                 'cell_to_cellType'):
        vb._step(name, getattr(vb, name))


def test_nothing_shaped_like_gamma_bar_is_pickled(minimal_varbayes):
    vb = minimal_varbayes
    _one_pass(vb)
    shape = (vb.nC, vb.nG, vb.nK)
    assert vb.spots.gamma_bar.shape == shape

    back = pickle.loads(pickle.dumps(vb.spots))
    # _post_rate is the rate of the gamma posterior and is the same shape. It is kept
    # on purpose, the elbo reads it, so it is the one allowed exception.
    offenders = [k for k, v in vars(back).items()
                 if isinstance(v, np.ndarray) and v.shape == shape and k != '_post_rate']
    assert offenders == []


def test_the_live_object_keeps_its_gamma_bar(minimal_varbayes):
    vb = minimal_varbayes
    _one_pass(vb)
    pickle.dumps(vb.spots)
    assert vb.spots.gamma_bar is not None
    assert vb.spots.gamma_bar.shape == (vb.nC, vb.nG, vb.nK)


def test_the_old_debug_name_is_gone(minimal_varbayes):
    vb = minimal_varbayes
    _one_pass(vb)
    assert not hasattr(vb.spots, 'my_gamma_bar')
