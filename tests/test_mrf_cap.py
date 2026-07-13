"""
Regression tests for the mrf cap gate in VarBayes._capped_mrf.

The cap is only meant to fire for cells whose OWN data (gene loglik + class
prior, ignoring the neighbours) already puts Zero on top. For any other cell
the winner is a real class, so there is no Zero label to protect and the
neighbours should keep their full pull (full beta) to clean the cell into a
real class.

The bug these tests guard against: testing D < 0 per class instead of over the
whole row. D < 0 per class only says "Zero beats this one class", which is true
for lots of cells whose actual winner is a real class (some other class beats
Zero). If someone rewrites the gate as a plain elementwise `where(D < 0, ...)`
again, test_real_winner_cell_keeps_full_beta will fail. The winner test must
also include the prior, not the gene loglik alone, which is what
test_prior_counts_towards_the_winner pins down.
"""

import types

import numpy as np

from pciSeq.src.core.main import VarBayes


def _run_capped_mrf(contr, support, log_prior, beta=1.0, tol=0.1):
    """Call VarBayes._capped_mrf with a stub self holding only what it reads.

    Returns (mrf_term, effective_beta). effective_beta is the per (cell, class)
    coupling actually used (0 = neighbours silenced, beta = full pull); mrf_term
    is effective_beta * support, the value added to the class log-score.
    """
    contr = np.asarray(contr, dtype=float)
    support = np.asarray(support, dtype=float)
    log_prior = np.asarray(log_prior, dtype=float)
    nK = contr.shape[1]

    cells = types.SimpleNamespace(mrf_support=lambda: support, effective_beta=None)
    cellTypes = types.SimpleNamespace(log_prior=log_prior)
    stub = types.SimpleNamespace(
        nK=nK,
        config={"mrf_beta": beta, "SpotReg": tol},
        cells=cells,
        cellTypes=cellTypes,
    )
    mrf = VarBayes._capped_mrf(stub, contr)
    return mrf, cells.effective_beta


# class layout used throughout: [real_A, real_B, Zero], Zero is the last column
A, B, ZERO = 0, 1, 2


def test_real_winner_cell_keeps_full_beta():
    """A cell whose data+prior winner is a real class must never be capped, even
    on a real class that on its own scores below Zero. This is the 22786 case:
    real_A wins the data, real_B loses to Zero, neighbours push real_B. The old
    per-class cap zeroed real_B; the gate must leave it at full beta."""
    beta = 1.0
    # data_score = contr + log_prior = [2.0, 0.3, 0.5]  -> argmax is real_A, not Zero.
    # real_B (0.3) sits below Zero (0.5), so per-class D[real_B] < 0.
    contr = [[2.0, 0.3, 0.0]]
    log_prior = [0.0, 0.0, 0.5]
    support = [[1.0, 6.0, 2.0]]  # neighbours mostly real_B

    _, eff = _run_capped_mrf(contr, support, log_prior, beta=beta)

    # no real class may be capped for this cell
    assert eff[0, A] == beta
    assert eff[0, B] == beta, "real_B was capped even though the cell's winner is a real class (the bug)"


def test_zero_winner_cell_is_capped():
    """A cell whose data+prior winner IS Zero must still be protected: a real
    class the neighbours push gets its coupling capped, and Zero keeps winning by
    at least tol once the (capped) mrf is added."""
    beta, tol = 1.0, 0.1
    # data_score = [0.0, 0.0, 0.5] -> Zero wins.
    contr = [[0.0, 0.0, 0.0]]
    log_prior = [0.0, 0.0, 0.5]
    support = [[6.0, 1.0, 0.0]]  # neighbours push real_A

    mrf, eff = _run_capped_mrf(contr, support, log_prior, beta=beta, tol=tol)

    # real_A must be capped below full beta
    assert eff[0, A] < beta
    # and Zero must still be the winner by at least tol after adding the mrf.
    # D[real_A] = contr - contr[Zero] + log_prior - log_prior[Zero] = 0 + (-0.5) = -0.5
    d_realA = -0.5
    full_realA = d_realA + mrf[0, A]  # score of real_A relative to Zero (which is 0)
    assert full_realA <= -tol + 1e-9, "capped mrf let a Zero-winner cell drift off Zero"


def test_prior_counts_towards_the_winner():
    """The winner test must use data PLUS prior, not the gene loglik alone. Here
    the gene loglik alone would call real_A the winner (so a loglik-only gate
    would not cap), but the prior tips Zero on top, so the cap must fire."""
    beta = 1.0
    # contr argmax is real_A (0.3 > 0), but contr + log_prior argmax is Zero (0.5 > 0.3).
    contr = [[0.3, 0.0, 0.0]]
    log_prior = [0.0, 0.0, 0.5]
    support = [[6.0, 1.0, 0.0]]  # neighbours push real_A

    _, eff = _run_capped_mrf(contr, support, log_prior, beta=beta)

    assert eff[0, A] < beta, "gate ignored the prior: Zero is the data+prior winner so real_A should be capped"


def test_zero_column_is_never_promoted():
    """The neighbours can never push a cell into Zero, for any cell, so the Zero
    column of the coupling is always 0."""
    beta = 1.0
    contr = [[2.0, 0.3, 0.0],   # real winner
             [0.0, 0.0, 0.0]]   # Zero winner
    log_prior = [0.0, 0.0, 0.5]
    support = [[1.0, 6.0, 3.0],
               [6.0, 1.0, 3.0]]

    mrf, eff = _run_capped_mrf(contr, support, log_prior, beta=beta)

    assert np.all(eff[:, ZERO] == 0.0)
    assert np.all(mrf[:, ZERO] == 0.0)


def test_zero_in_second_place_does_not_trigger_cap():
    """The exact cell 22786 shape: data+prior orders the classes

        CA1-ProS Glut  >  Zero  >  L6 CT CTX Glut

    so the winner is a real class (CA1-ProS Glut), Zero is only second, and the
    neighbour class (L6 CT CTX Glut) is third, below Zero. The neighbours push
    L6 CT CTX Glut. The old per-class cap fired on L6 CT CTX Glut just because it
    sits below Zero, which silenced the neighbours and let CA1-ProS Glut pop back
    in, and the cell oscillated. The gate must leave L6 CT CTX Glut at full beta
    so the neighbours clean the cell to L6 CT CTX Glut and it stays there."""
    ca1, l6ct, zero_ = 0, 1, 2
    beta, tol = 1.0, 0.1
    # data_score = contr + log_prior = [2.0, 0.0, 0.5]:  CA1 > Zero > L6CT
    contr = [[2.0, 0.0, 0.0]]
    log_prior = [0.0, 0.0, 0.5]
    support = [[1.0, 6.0, 0.0]]  # neighbours push L6 CT CTX Glut

    # confirm the scenario really has Zero squarely in the middle
    data_score = np.asarray(contr[0]) + np.asarray(log_prior)
    assert data_score[ca1] > data_score[zero_] > data_score[l6ct]

    mrf, eff = _run_capped_mrf(contr, support, log_prior, beta=beta, tol=tol)

    # L6 CT CTX Glut must not be capped
    assert eff[0, l6ct] == beta, "L6 CT CTX Glut got capped although Zero is not the winner"

    # and with the mrf added the cell must land on L6 CT CTX Glut (the neighbour
    # class) and stay, not flip back to the data winner CA1-ProS Glut. Scores are
    # relative to Zero, whose score is 0.
    d = (np.asarray(contr[0]) - contr[0][zero_]) + (np.asarray(log_prior) - log_prior[zero_])
    full = d + mrf[0]
    assert full.argmax() == l6ct, "cell did not settle on the neighbour class (this is the oscillation)"


def test_mixed_batch_gates_per_cell():
    """Two cells in one call: a real-winner cell keeps full beta while a
    Zero-winner cell gets capped. Guards that the gate is decided per cell, not
    globally for the whole batch."""
    beta = 1.0
    contr = [[2.0, 0.3, 0.0],   # cell 0: real_A winner
             [0.0, 0.0, 0.0]]   # cell 1: Zero winner
    log_prior = [0.0, 0.0, 0.5]
    support = [[1.0, 6.0, 0.0],
               [6.0, 1.0, 0.0]]

    _, eff = _run_capped_mrf(contr, support, log_prior, beta=beta)

    assert eff[0, B] == beta          # real-winner cell: uncapped
    assert eff[1, A] < beta           # Zero-winner cell: capped
