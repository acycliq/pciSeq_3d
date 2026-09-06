"""
scRNAseq is required.

pciSeq used to accept scRNAseq=None and fall back on estimating the class
definitions from the spots alone. That path never worked: mu_upd wrote the
estimate to attributes nothing read, so the diagonal starting reference was
used unchanged for the whole run. The path is gone and the input is compulsory.
"""
import pytest

from pciSeq.src.validation.validator import Validator


def test_missing_scrnaseq_is_rejected(base_opts):
    v = Validator(spots=None, coo=None, scdata=None, config=dict(base_opts))
    with pytest.raises(ValueError, match='scRNAseq is required'):
        v.validate_all()


def test_the_no_reference_machinery_is_gone():
    """Nothing should be left that only existed to serve that path."""
    from pciSeq.src.core.datatypes.singleCell import SingleCell
    from pciSeq.src.core.datatypes.cells import Cells
    from pciSeq.src.core.main import VarBayes
    from pciSeq import config

    assert not hasattr(SingleCell, '_diag')
    assert not hasattr(SingleCell, '_gene_expressions')
    assert not hasattr(VarBayes, 'mu_upd')
    assert 'mean_gene_counts_per_class' not in config.DEFAULT
    assert 'mean_gene_counts_per_cell' not in config.DEFAULT
