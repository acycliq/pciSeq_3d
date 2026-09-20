"""
cell_typing_breakdown was taken out, this makes sure it stays out and that nothing
else went with it.

It was a public method on VarBayes that could not run any more: it called
cellTypes.ini_alpha(), which had been removed from CellClass, so it died with an
AttributeError. Its docstring also still described the old cell_type_weights, the
Dirichlet ones with the 'default' key. Nothing used it, not the package, the tests,
the docs, the viewer or the run scripts, so it went rather than get rewritten.

What stays: check_cell and check_spot, the two that do get used, and
calculate_genes_log_likelihood_contr, which the breakdown used to call but which is a
method in its own right.
"""
from pciSeq.src.core.main import VarBayes
from pciSeq.src.core.utils import inspection


def test_the_method_is_gone():
    assert not hasattr(VarBayes, 'cell_typing_breakdown')
    assert not hasattr(inspection, 'cell_typing_breakdown')


def test_its_helpers_went_with_it():
    assert not hasattr(inspection, '_plot_classification_steps')


def test_the_ones_that_are_used_are_still_there():
    for name in ('check_cell', 'check_spot', 'calculate_genes_log_likelihood_contr'):
        assert callable(getattr(VarBayes, name)), name
    assert callable(inspection.check_cell)
    assert callable(inspection.check_spot)


def test_the_likelihood_helper_still_runs(minimal_varbayes):
    vb = minimal_varbayes
    vb.initialise_state()
    vb._step_times = {}
    vb.iter_num = 0
    for name in ('geneCount_upd', 'rho_upd', 'eta_upd', 'theta_upd', 'gamma_upd',
                 'cell_to_cellType'):
        vb._step(name, getattr(vb, name))
    contr, *_ = vb.calculate_genes_log_likelihood_contr(1)
    assert contr.shape == (vb.nG, vb.nK)
