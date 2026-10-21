"""Numba kernels for the variational-Bayes loop.

These are the fast compiled inner loops, kept out of main.py so the algorithm
code stays readable. The plain-numpy reference version of each kernel lives next
to the method in VarBayes that calls it.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def spots_to_cell_numba_kernel(parent, gene_id, classProb, log_gamma_bar, log_theta,
                               expected_counts, logeta_bar, nNb,
                               wSpotCell, attention, expr_fluct, cell_ineff, gene_ineff):
    """Computes the three spot-to-cell weight terms for every spot and its nearest cells.

    For each spot s and each of its nNb nearest cells, adds up the three terms below.
    It reads values straight from the arrays instead of first copying out the rows, so
    it never builds the large temporary [nS, nK] arrays the numpy version does, and it
    runs over spots on all cores (numba).
      term_1 = sum_k          expected_counts[s,k] * classProb[cell,k]
      term_2 = sum_{k != Zero} classProb[cell,k]   * log_gamma_bar[cell, gene_s, k]
      term_3 = sum_{k != Zero} classProb[cell,k]   * log_theta[cell,k]
      eta    = (1 - classProb[cell, Zero]) * logeta_bar[s]
    Zero is the last column; it gets no gamma, theta or eta term, see the Notes in
    VarBayes.spots_to_cell. mvn_loglik and the misread column are handled by the
    caller.
    """
    nS = parent.shape[0]
    nK = classProb.shape[1]
    kZero = nK - 1
    for s in prange(nS):
        g = gene_id[s]
        le = logeta_bar[s]
        for n in range(nNb):
            cell = parent[s, n]
            t1 = 0.0
            t2 = 0.0
            t3 = 0.0
            for k in range(nK):
                t1 += expected_counts[s, k] * classProb[cell, k]
            for k in range(kZero):
                cpk = classProb[cell, k]
                t2 += cpk * log_gamma_bar[cell, g, k]
                t3 += cpk * log_theta[cell, k]
            le_cell = le * (1.0 - classProb[cell, kZero])
            wSpotCell[s, n] = t1 + t2 + t3 + le_cell
            attention[s, n] = t1
            expr_fluct[s, n] = t2
            cell_ineff[s, n] = t3
            gene_ineff[s, n] = le_cell
