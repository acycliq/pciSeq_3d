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
      term_1 = sum_k expected_counts[s,k] * classProb[cell,k]
      term_2 = sum_k classProb[cell,k]   * log_gamma_bar[cell, gene_s, k]
      term_3 = sum_k classProb[cell,k]   * log_theta[cell,k]
    mvn_loglik and the misread column are handled by the caller.
    """
    nS = parent.shape[0]
    nK = classProb.shape[1]
    for s in prange(nS):
        g = gene_id[s]
        le = logeta_bar[s]
        for n in range(nNb):
            cell = parent[s, n]
            t1 = 0.0
            t2 = 0.0
            t3 = 0.0
            for k in range(nK):
                cpk = classProb[cell, k]
                t1 += expected_counts[s, k] * cpk
                t2 += cpk * log_gamma_bar[cell, g, k]
                t3 += cpk * log_theta[cell, k]
            wSpotCell[s, n] = t1 + t2 + t3 + le
            attention[s, n] = t1
            expr_fluct[s, n] = t2
            cell_ineff[s, n] = t3
            gene_ineff[s, n] = le


@njit(parallel=True, cache=True)
def gamma_bar_kernel(scaled_exp, eta_bar, theta_bar, rho, psi_rho, rSpot,
                     gamma_bar, log_gamma_bar, beta):
    """Fills the three big [nC, nG, nK] arrays that gamma_upd needs.

    This does the same element-by-element maths as the numpy version in
    gamma_upd, just spread over all cores (numba) with one pass instead of
    building several [nC, nG, nK] temporaries. For every cell c, gene g and
    class k it writes:
      beta          = scaled_exp * eta_bar * theta_bar + rSpot
      gamma_bar     = rho / beta          (the Gamma mean)
      log_gamma_bar = psi(rho) - log(beta)   (E[log gamma])
    psi(rho) is the digamma of rho; it does not depend on k, so the caller
    works it out once (as psi_rho, shape [nC, nG]) and passes it in.
    """
    C, G, K = scaled_exp.shape
    for c in prange(C):
        for g in range(G):
            r = rho[c, g]
            pr = psi_rho[c, g]
            for k in range(K):
                b = scaled_exp[c, g, k] * eta_bar[g] * theta_bar[c, k] + rSpot
                beta[c, g, k] = b
                inv = np.float32(1.0) / b
                gamma_bar[c, g, k] = r * inv
                log_gamma_bar[c, g, k] = pr - np.log(b)


@njit(parallel=True, cache=True)
def gene_loglik_kernel(scaled_exp, eta_bar, theta_bar, geneCount, rSpot, SpotReg, contr):
    """Fills the [nC, nG, nK] negative-binomial log-likelihood matrix.

    Same maths as compute_gene_loglikelihood_matrix, done over all cores.
    For every cell c, gene g and class k:
      ScaledExp = scaled_exp * eta_bar * theta_bar + SpotReg
      q         = ScaledExp / (rSpot + ScaledExp)
      contr     = geneCount * log(q) + rSpot * log(1 - q)
    The caller still sums this over genes with numpy, so the per-gene sum stays
    bit-for-bit the same as before; only the element-wise part is parallelised.
    """
    C, G, K = scaled_exp.shape
    for c in prange(C):
        for g in range(G):
            gc = geneCount[c, g]
            for k in range(K):
                se = scaled_exp[c, g, k] * eta_bar[g] * theta_bar[c, k] + SpotReg
                q = se / (rSpot + se)
                contr[c, g, k] = gc * np.log(q) + rSpot * np.log(np.float32(1.0) - q)


@njit(parallel=True, cache=True)
def mvn_loglik_kernel(xyz, parent, centroids, eig_vals, eig_vecs, nNb, out):
    """Multivariate-normal log-pdf of every spot under its nearest cells (3D).

    This is the numba twin of Spots.mvn_loglik / multiple_logpdfs for the 3D
    case. For each spot s and each of its nNb nearest cells it evaluates the
    Gaussian log-density of the spot position under that cell's fitted 3D
    Gaussian, using the cell's eigen-decomposed covariance (eig_vals, eig_vecs).
    Because the covariance is fixed once and never re-fit in this version, this
    is pure geometry; running it over spots on all cores replaces the old
    per-neighbour numpy loop.
    """
    nS = xyz.shape[0]
    log2pi = np.log(2.0 * np.pi)
    for s in prange(nS):
        for n in range(nNb):
            c = parent[s, n]
            # d = spot position minus this cell's centroid (3-vector)
            d0 = xyz[s, 0] - centroids[c, 0]
            d1 = xyz[s, 1] - centroids[c, 1]
            d2 = xyz[s, 2] - centroids[c, 2]
            maha = 0.0
            logdet = 0.0
            for j in range(3):
                v = eig_vals[c, j]
                logdet += np.log(v)
                s_j = np.sqrt(1.0 / v)
                # project d onto eigenvector j and scale by 1/sqrt(eigenvalue)
                du = (d0 * eig_vecs[c, 0, j] + d1 * eig_vecs[c, 1, j] + d2 * eig_vecs[c, 2, j]) * s_j
                maha += du * du
            out[s, n] = -0.5 * (3.0 * log2pi + maha + logdet)