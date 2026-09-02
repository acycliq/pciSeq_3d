"""The capped MRF term for the cell-class update, see calc_capped_mrf."""
import numpy as np


def calc_capped_mrf(contr, support, log_prior, beta, tol):
    """MRF term, capped so the neighbours can never flip a cell out of Zero.

    For each real class k, D[c,k] is the cell's own score of k against
    Zero (NB loglik + log prior, relative to Zero) and the MRF adds
    beta * S[c,k] on top, with S the neighbour support. Where the data
    alone makes Zero the winner (D < 0 for every real class), beta is
    capped at beta* = -(D + tol) / S, the coupling that leaves Zero ahead
    by tol. Everywhere else the full beta is kept: there is no Zero label
    to protect and the neighbours are free to clean the cell into a real
    class. The Zero class itself gets no coupling: Zero membership is
    decided by the data alone, the neighbours can never push a cell into
    Zero.

    Parameters
    ----------
    contr : np.ndarray
        (nC, nK) negative-binomial log-likelihood per cell and class,
        already summed over genes.
    support : np.ndarray
        (nC, nK) beta-free neighbour support, from cells.mrf_support().
    log_prior : np.ndarray
        (nK,) log of the class prior.
    beta : float
        The user's mrf_beta, the coupling the cap is allowed to reduce.
    tol : float
        How far ahead Zero must stay, in log-odds, wherever the cap fires.
        SpotReg doubles as tol, so there is no extra knob.

    Returns
    -------
    tuple of np.ndarray
        (mrf_term, effective_beta), both (nC, nK). The first is what gets
        added to the cell-class log-score, the second is the per (cell, class)
        coupling actually used, which the caller keeps for diagnostics.
    """
    nK = contr.shape[1]
    zero = nK - 1  # Zero is the last class

    # data + prior score of each class relative to Zero (Zero column is 0)
    D = (contr - contr[:, [zero]]) + (log_prior - log_prior[zero])

    # beta* solves D + beta*S = -tol. Where S ~ 0 the MRF term is ~0
    # anyway, so the inf there is clamped away below; errstate keeps
    # numpy quiet about the division.
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        beta_star = -(D + tol) / support

    # Cap only cells whose own data picks Zero, ie every real class scores
    # below it; any other row keeps full beta. Deciding this per class
    # instead of per cell is what made boundary cells oscillate
    # (regression cover: tests/test_mrf_cap.py).
    zero_is_winner = (D[:, :zero] < 0).all(axis=1)

    # clip covers the band -tol <= D < 0, where no non-negative beta can
    # give Zero its margin, by dropping the MRF there entirely
    capped = np.where(zero_is_winner[:, None] & (D < 0),
                      np.clip(beta_star, 0.0, beta),
                      beta)
    capped[:, zero] = 0.0  # neighbours never push a cell into Zero

    return capped * support, capped
