"""Statistical calculation utilities."""
import numpy as np
import pandas as pd
import numba
import numpy_groupies as npg
from typing import Tuple, Optional, Any, Union
import logging
import opt_einsum as oe

# Optional GPU support via CuPy. The module imports cleanly even without it;
# b_upd_gpu raises a clear error if called without a working GPU.
try:
    import cupy as _cp
    _cp.cuda.runtime.getDeviceCount()  # raises if no driver / no GPU visible
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False

# ... existing code ...

def b_upd_naive(b, mu, Ac, eta, theta, gamma, precision, counts):
    """
    Newton-Raphson update for the expression bias term 'b'.
    This is a naive implementation based on Section 4.2 of the pciSeq notes.
    Used for mathematical verification and clarity.

    Parameters
    ----------
    b : np.ndarray
        Array of shape (nC, nG, nK) containing the log-expression bias.
    mu : np.ndarray
        Array of shape (nG, nK) containing the mean expression values.
    Ac : np.ndarray
        Array of shape (nC,) containing the cell area factors.
    eta : np.ndarray
        Array of shape (nG,) containing the gene efficiency terms.
    theta : np.ndarray
        Array of shape (nC, nK) containing the cell inefficiency terms.
    gamma : np.ndarray
        Array of shape (nC, nG, nK) containing the spot-cell scale factors.
    precision : np.ndarray
        Array of shape (nK, nG, nG) containing the precision matrices (inverse covariance).
    counts : np.ndarray
        Array of shape (nC, nG) containing the observed gene counts per cell.

    Returns
    -------
    np.ndarray
        Updated expression bias array 'b' of shape (nC, nG, nK).
    """
    nC, nG, nK = b.shape
    b_out = b.copy()
    for c in range(nC):
        for k in range(nK):
            # Λ_c|k (Expected counts based on current class)
            Lambda = mu[:, k] * Ac[c] * eta * theta[c, k] * gamma[c, :, k]
            
            # term_1 = Λ ⊙ e^b
            term_1 = Lambda * np.exp(b_out[c, :, k])
            
            # term_2 = Σ^-1 b
            term_2 = precision[k] @ b_out[c, :, k]
            
            # Gradient ∇L = N - term_1 - term_2
            grad = counts[c] - term_1 - term_2
            
            # Hessian H = -diag(term_1) - Σ^-1
            H = -np.diag(term_1) - precision[k]
            
            # Newton-Raphson update: b = b - H^-1 @ grad
            step = np.linalg.solve(H, grad)
            b_out[c, :, k] -= step
    return b_out


@numba.njit(parallel=True, fastmath=True)
def b_upd_optimized(b, mu, Ac, eta, theta, gamma, precision, counts, class_prob, tol=1e-3, max_iter=20):
    """
    Fast implementation of b_upd using Sparse PCG and Numba.
    Optimizes each cell's expression bias by solving the Newton step using
    Conjugate Gradient, parallelized across all CPU cores.

    Parameters
    ----------
    b : np.ndarray
        Array of shape (nC, nG, nK) containing current log-expression bias values.
    mu : np.ndarray
        Mean expression array of shape (nG, nK).
    Ac : np.ndarray
        Cell area factors array of shape (nC,).
    eta : np.ndarray
        Gene efficiency terms array of shape (nG,).
    theta : np.ndarray
        Cell inefficiency terms array of shape (nC, nK).
    gamma : np.ndarray
        Spot-cell scale factors array of shape (nC, nG, nK).
    precision : np.ndarray
        Precision matrices array of shape (nK, nG, nG).
    counts : np.ndarray
        Observed gene counts array of shape (nC, nG).
    class_prob : np.ndarray
        Class assignment probabilities array of shape (nC, nK).
    tol : float, optional
        Threshold for class probability. Classes below this are skipped. Default 1e-3.
    max_iter : int, optional
        Maximum number of PCG iterations per solve. Default 20.

    Returns
    -------
    np.ndarray
        Updated expression bias array 'b' of shape (nC, nG, nK).
    """
    nC, nG, nK = b.shape
    f4 = np.float32
    for c in numba.prange(nC):
        Ac_c = Ac[c]
        gc_c = counts[c]
        for k in range(nK):
            if class_prob[c, k] < tol:
                continue
                
            Pk = precision[k]
            diag_Pk = np.diag(Pk)
            theta_ck = theta[c, k]
            
            b_ck = np.ascontiguousarray(b[c, :, k]).astype(f4)
            gamma_ck = gamma[c, :, k].astype(f4)
            
            Dk = np.empty(nG, dtype=f4)
            rk = np.empty(nG, dtype=f4)
            
            term_2 = np.dot(Pk, b_ck)
            
            for g in range(nG):
                lk = mu[g, k] * Ac_c * eta[g] * theta_ck * gamma_ck[g]
                Dk[g] = lk * np.exp(b_ck[g])
                rk[g] = gc_c[g] - Dk[g] - term_2[g]
                
            M_inv = f4(1.0) / (Dk + diag_Pk)
            zk = rk * M_inv
            pk = zk.copy()
            rz_old = f4(np.sum(rk * zk))
            
            xk = np.zeros(nG, dtype=f4)
            
            for i in range(max_iter):
                Apk = np.dot(Pk, pk) + Dk * pk
                pAp = f4(np.sum(pk * Apk))
                if pAp == 0: break
                
                alpha = rz_old / pAp
                xk += alpha * pk
                rk -= alpha * Apk
                
                zk = rk * M_inv
                rz_new = f4(np.sum(rk * zk))
                beta = rz_new / (rz_old + f4(1e-12))
                pk = zk + beta * pk
                rz_old = rz_new
                
            for g in range(nG):
                b[c, g, k] += xk[g]
    return b


def b_upd_gpu(b, mu, Ac, eta, theta, gamma, precision, counts, class_prob,
              tol=1e-3, max_iter=20):
    """
    GPU implementation of b_upd. Same Jacobi-PCG algorithm as b_upd_optimized
    above, but vectorised across all active cells of each class so the
    dominant work becomes batched cuBLAS sgemm. Drop-in replacement: same
    signature, same defaults, same in-place + return contract.

    Relation to the other implementations
    -------------------------------------
    - `b_upd_naive`: gold-standard reference. One full dense `np.linalg.solve`
       per (c, k). Exact to float64 rounding. O(nC * nK * nG^3); too slow for
       production but used as the verification oracle in tests/test_b_upd.py.
    - `b_upd_optimized`: production CPU. Same Jacobi-PCG inner loop as this
       function, but loops cell-by-cell inside `numba.prange` and uses one
       sgemv per cell per iteration.
    - `b_upd_gpu`: same algorithm again, but batches all active cells of
       each class into a single (nG, n_a) matrix so each PCG iteration
       collapses to one sgemm `Pk @ pk_matrix`. Runs on GPU via CuPy.

    Numerical equivalence
    ---------------------
    Bit-for-bit equivalence to b_upd_optimized is not guaranteed because the
    sums in the per-cell dot products execute in different orders, but
    agreement is within float32 rounding (~1e-4). Both implementations meet
    the existing 1e-3 tolerance in tests/test_b_upd.py.

    Inputs
    ------
    Accepts either numpy or cupy arrays for every argument.
    - numpy in: the function uploads inputs to GPU, runs, and copies the
      updated `b` back to the host array in place. True drop-in.
    - cupy in (recommended for outer loops): runs entirely on GPU, no PCIe
      transfer per call. Caller uploads inputs once before the outer loop.

    Performance, RTX 2060, production scale (nC=24456, nG=207, nK=39)
    -----------------------------------------------------------------
    With cupy inputs (no per-call transfer), median wall time per call:
      * iter 3+ (lucky=2,  ~50 K active pairs):  ~250 ms      [9.4x vs CPU]
      * iter 2  (lucky=5, ~120 K active pairs):  ~480 ms
      * iter 1  (lucky=39, ~950 K active):       ~3.5 s

    With numpy inputs the per-call PCIe upload of `b` and `gamma` (~1.6 GB)
    adds ~150-300 ms. Still faster end-to-end than the CPU production path.

    Parameters
    ----------
    Same as `b_upd_optimized`. See its docstring above.

    Raises
    ------
    RuntimeError
        If CuPy is not installed or no compatible GPU is visible. Use
        `b_upd_optimized` instead in that case.
    """
    if not _HAS_CUPY:
        raise RuntimeError(
            "CuPy/GPU not available. "
            "Install GPU dependencies (see pciSeq_3d/setup.py) or fall back "
            "to b_upd_optimized."
        )
    cp = _cp

    # Detect input array module so we can return numpy results to a numpy caller.
    is_numpy_in = isinstance(b, np.ndarray)
    if is_numpy_in:
        b_g           = cp.asarray(b)
        mu_g          = cp.asarray(mu)
        Ac_g          = cp.asarray(Ac)
        eta_g         = cp.asarray(eta)
        theta_g       = cp.asarray(theta)
        gamma_g       = cp.asarray(gamma)
        precision_g   = cp.asarray(precision)
        counts_g      = cp.asarray(counts)
        class_prob_g  = cp.asarray(class_prob)
    else:
        b_g, mu_g, Ac_g       = b, mu, Ac
        eta_g, theta_g        = eta, theta
        gamma_g, precision_g  = gamma, precision
        counts_g, class_prob_g = counts, class_prob

    nC, nG, nK = b_g.shape
    f4 = cp.float32

    for k in range(nK):
        active = cp.where(class_prob_g[:, k] >= tol)[0]
        n_a = int(active.size)
        if n_a == 0:
            continue

        Pk = precision_g[k]
        diag_Pk = cp.diag(Pk).astype(f4)

        # Stack all active cells of class k as columns of (nG, n_a) matrices.
        b_k       = cp.ascontiguousarray(b_g[active, :, k].T).astype(f4)
        gamma_k   = cp.ascontiguousarray(gamma_g[active, :, k].T).astype(f4)
        counts_a  = cp.ascontiguousarray(counts_g[active, :].T).astype(f4)
        scale     = (Ac_g[active] * theta_g[active, k]).astype(f4)
        Lambda    = ((mu_g[:, k] * eta_g).astype(f4))[:, None] * scale[None, :] * gamma_k
        Dk        = Lambda * cp.exp(b_k)

        # Initial residual r0 = grad(b) = counts - D - Pk @ b.
        rk = counts_a - Dk - Pk @ b_k

        # Jacobi preconditioner M = diag(D + diag(Pk)). Same as b_upd_optimized.
        M_inv = (f4(1.0) / (Dk + diag_Pk[:, None])).astype(f4)

        zk = rk * M_inv
        pk = zk.copy()
        rz_old = (rk * zk).sum(axis=0).astype(f4)
        xk = cp.zeros((nG, n_a), dtype=f4)

        for _ in range(max_iter):
            Apk = Pk @ pk + Dk * pk
            pAp = (pk * Apk).sum(axis=0).astype(f4)
            safe_pAp = cp.where(pAp != 0, pAp, f4(1.0))
            alpha = (rz_old / safe_pAp).astype(f4)
            xk += alpha[None, :] * pk
            rk -= alpha[None, :] * Apk
            zk = rk * M_inv
            rz_new = (rk * zk).sum(axis=0).astype(f4)
            beta = (rz_new / (rz_old + f4(1e-12))).astype(f4)
            pk = zk + beta[None, :] * pk
            rz_old = rz_new

        b_g[active, :, k] = (b_k + xk).T

    if is_numpy_in:
        # Copy result back into the caller's numpy array, in place.
        b[:] = cp.asnumpy(b_g)
    return b


def b_upd_fixed_point(b, mu, Ac, eta, theta, gamma, precision, counts, class_prob,
                      tol=1e-3, max_iter=3):
    """
    Fixed-point splitting Newton-step solver. Drop-in signature with the other
    b_upd_* functions. Provided for experimentation and accuracy/speed
    comparison; **not recommended as a production replacement for
    b_upd_optimized / b_upd_gpu** because of the convergence caveat below.

    Algorithm
    ---------
    The Newton system per (c, k) is `(Pk + D_c) x_c = grad_c`. We rewrite as

        Pk x_new = grad_c - D_c x_old

    and iterate from x_0 = 0. After `max_iter` steps,

        x ≈ (Pk + D)^(-1) grad

    Each iteration applies `Pk^{-1}` (precomputed once per class per call)
    as a single matvec. With a strong prior (Pk^{-1} small relative to D)
    this converges in ~3 iterations for our problem; otherwise it diverges.

    Convergence condition
    ---------------------
    The error contracts by `||Pk^(-1) D||_2` per iteration. In the pciSeq
    operating regime (heavy-tailed `mu` with max ≥ 50, weak prior at
    initialisation with cov=I), this norm is **above 1 for ~22% of (c, k)
    pairs** measured on a representative pickle (see
    `notes/b_upd_methods.md` for the diagnostic). For those pairs the
    iteration diverges. PCG with Jacobi preconditioning (the production
    `b_upd_optimized`) is unconditionally stable and should be preferred
    for production use.

    When this function is fast and accurate
    ---------------------------------------
    - Strong prior (lambda_min(Pk) >> max(D))
    - Safe-regime data (e.g. early VB iterations after the prior tightens,
      or synthetic with `Pk = A A.T + 10 I` and `mu = U(0,1)` style inputs).
    In that regime: ~3 iterations to reach 3e-5 max abs error vs naive,
    versus ~20-25 PCG iterations for the same accuracy. Substantially faster.

    Parameters
    ----------
    Same as `b_upd_optimized`. See its docstring above.
    Note: `max_iter` defaults to 3 (vs 20 for PCG) because each fixed-point
    iteration is a single Pk^(-1) matvec and convergence (when it occurs)
    is geometric with a small contraction factor.

    Inputs
    ------
    Accepts numpy or cupy arrays. Same auto-detection as `b_upd_gpu`.
    """
    is_cupy_in = (_HAS_CUPY and isinstance(b, _cp.ndarray))
    xp = _cp if is_cupy_in else np

    nC, nG, nK = b.shape
    f4 = xp.float32

    # Precompute Pk^(-1) once per class. (Mathematically the per-class
    # covariance, since precision = inv(cov).)
    P_inv = xp.stack([xp.linalg.inv(precision[k]) for k in range(nK)])

    for k in range(nK):
        active = xp.where(class_prob[:, k] >= tol)[0]
        n_a = int(active.size)
        if n_a == 0:
            continue

        Pk = precision[k]
        Pk_inv = P_inv[k]

        b_k       = xp.ascontiguousarray(b[active, :, k].T).astype(f4)
        gamma_k   = xp.ascontiguousarray(gamma[active, :, k].T).astype(f4)
        counts_a  = xp.ascontiguousarray(counts[active, :].T).astype(f4)
        scale     = (Ac[active] * theta[active, k]).astype(f4)
        Lambda    = ((mu[:, k] * eta).astype(f4))[:, None] * scale[None, :] * gamma_k
        Dk        = Lambda * xp.exp(b_k)

        grad = counts_a - Dk - Pk @ b_k

        # Fixed-point split: Pk x_new = grad - D x_old, starting x_0 = 0.
        xk = Pk_inv @ grad
        for _ in range(max_iter - 1):
            xk = Pk_inv @ (grad - Dk * xk)

        b[active, :, k] = (b_k + xk).T

    return b


from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.special import psi, softmax

# Configure logging
logger = logging.getLogger(__name__)


def expected_covariance(scale_matrix, dof):
    """
        Calculate the expected covariance matrix from a scale matrix and degrees of freedom.

        Parameters
        ----------
        scale_matrix : np.ndarray
            Scale matrix of shape (C, d, d) where d must be 2 or 3
        dof : np.ndarray
            Degrees of freedom,shape (C,).
            Values will be automatically adjusted if below d + 2

        Returns
        -------
        np.ndarray
            Expected covariance matrix of same shape as input scale_matrix

        Raises
        ------
        ValueError
            If matrix dimensions are invalid or don't match
    """
    # Get the last two dimensions
    *_, d1, d2 = scale_matrix.shape

    # Check square
    if d1 != d2:
        raise ValueError(f"scale_matrix must be square, got shape {scale_matrix}")

    # Check dimension is 2 or 3
    if d1 not in (2, 3):
        raise ValueError(f"scale_matrix dimension must be 2 or 3, got {d1}")

    # Adjust degrees of freedom if needed, maybe I should drop a warning?
    min_dof = d1 + 1
    dof[dof <= min_dof] = min_dof + 1

    return scale_matrix / (dof[:, None, None] - d1 - 1)


def negative_binomial_loglikelihood(x: np.ndarray, r: float, q: np.ndarray) -> np.ndarray:
    """Calculate the Negative Binomial log-likelihood for given parameters.

    The Negative Binomial distribution models the number of failures (x) before
    observing the r-th success, with failure probability q. The PMF is:
        P(X = x) = C(x + r - 1, x) * q^x * (1 - q)^r

    Here we compute only the terms that depend on q and r:
        log-likelihood = x * log(q) + r * log(1 - q)

    Args:
        x: Array of observed failure counts (non-negative floats).
        r: Number of successes until stopping (dispersion parameter, positive).
        q: Array of failure probabilities (each between 0 and 1).

    Returns:
        Array of log-likelihood values, broadcast over x and q.

    Raises:
        ValueError: If any q is outside (0, 1) or if x has negative values.
    """
    try:
        x = x[:, :, None]  # Add dimension for broadcasting

        # Compute the log-likelihood of seeing x failures before the r-th success,
        # if the failure probability is q.
        # In our context, x is the cell gene counts, q is derived from the single cell data
        # count data and r is a hyperparameter (set by default = 2.0).
        # Scipy's nbinom object has logpmf(k, n, p) where p is the prob of success, ie p = 1-q
        # and k, n is what is denoted here by x, r respectively. Also logpmf includes the
        # combinatorial factor. Finally logpmf will drop an exception if the counts k are not
        # integers
        log_likelihood = x * np.log(q) + r * np.log(1 - q)

        return log_likelihood

    except Exception as e:
        logger.error(f"Error calculating negative binomial log-likelihood: {str(e)}")
        raise ValueError("Failed to compute log-likelihood. Check input dimensions and values.")


def compute_gene_loglikelihood_matrix(obj) -> np.ndarray:
    """
    Compute the full gene log-likelihood contribution matrix for all cells and cell types.

    This function performs the core computation shared between cell_to_cellType and
    calculate_genes_log_likelihood_contr, eliminating code duplication and improving performance.

    Args:
        obj: VarBayes object containing the following attributes:
            - scaled_exp: A delayed or computed array of scaled expression values (shape: nC x nG x nK)
            - genes.eta_bar: Gene efficiency (shape: nG)
            - cells.theta_bar: Cell inefficiency (shape: nC)
            - config['SpotReg']: Regularization parameter for spot-level noise
            - config['rSpot']: Dispersion parameter for the negative binomial distribution
            - cells.geneCount: Observed gene counts for all cells (shape: nC x nG)

    Returns:
        np.ndarray: Log-likelihood contributions matrix of shape (nC, nG, nK)
                   where element [c,g,k] is the log-likelihood contribution of
                   gene g in cell c for cell type k
    """
    # Compute scaled expression (expensive operation done once)
    scaled_means = obj.scaled_exp.compute()

    # Calculate scaled expression adjusted by gene efficiency and regularization
    ScaledExp = np.einsum('cgk,cgk,g,ck->cgk', scaled_means, np.exp(obj.cells.b), obj.genes.eta_bar, obj.cells.theta_bar) + obj.config['SpotReg']

    # Calculate negative binomial probabilities
    pNegBin = ScaledExp / (obj.config['rSpot'] + ScaledExp)

    # Get gene counts for all cells
    cgc = obj.cells.geneCount

    # Calculate log-likelihood contributions for all cells
    contr = negative_binomial_loglikelihood(cgc, obj.config['rSpot'], pNegBin)

    return contr


def calculate_genes_log_likelihood_contr(obj, label: int) -> Tuple[DataFrame, Series, DataFrame]:
    """
    Calculate the log-likelihood contributions, gene counts, and scaled expression values
    for a specific cell.

    This function computes:
        1. The genes' log-likelihood contributions (`contr`) for the specified cell under a
           negative binomial distribution.
        2. The gene counts (`cgc`) for the specified cell.
        3. The scaled expression values (`scaled_means`) for the specified cell.

    Args:
        obj: An object containing the following attributes:
            - scaled_exp: A delayed or computed array of scaled expression values (shape: nC x nG x nK).
            - genes.eta_bar: Gene efficiency parameters (shape: nG).
            - config['SpotReg']: Regularization parameter for spot-level noise.
            - config['rSpot']: Dispersion parameter for the negative binomial distribution.
            - cells.geneCount: Observed gene counts for all cells (shape: nC x nG).
        label (int): The index of the cell for which to compute the values.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - contr: The log-likelihood contributions for the specified cell (shape: nG x nK).
            - cgc: The gene counts for the specified cell (shape: nG).
            - scaled_means: The scaled expression values for the specified cell (shape: nG x nK).
    """
    # If original labels have been renumbered find the label it's been mapped to.
    if obj.config['label_map']:
        label = obj.config['label_map'][label]

    # Get the full log-likelihood matrix using shared computation
    contr = compute_gene_loglikelihood_matrix(obj)

    # Get scaled expression and gene counts
    scaled_means = obj.scaled_exp.compute()
    cgc = obj.cells.geneCount

    # Return values for the specified cell
    contr_df = pd.DataFrame(contr[label], columns=obj.cells.class_names).set_index(obj.genes.gene_panel)
    gene_counts = pd.Series(cgc[label], index=obj.genes.gene_panel)
    scaled_means_df = pd.DataFrame(scaled_means[label], columns=obj.cells.class_names).set_index(obj.genes.gene_panel)
    return contr_df, gene_counts, scaled_means_df


# def plot_loglik_contr(df):
#     """
#     Create a scatter plot of the first column vs the second column in a DataFrame,
#     with tooltips from the index, and add a diagonal line (y = x).
#
#     Args:
#         df (pd.DataFrame): The DataFrame containing the data.
#     """
#     # Ensure the DataFrame has at least two columns
#     if len(df.columns) < 2:
#         raise ValueError("The DataFrame must have at least two columns.")
#
#     # Reset the index to include it as a column for tooltips
#     df = df.reset_index()
#
#     # Get the names of the first and second columns
#     x_col = df.columns[1]  # First column (after resetting the index)
#     y_col = df.columns[2]   # Second column (after resetting the index)
#
#     # Create the scatter plot with tooltips
#     fig = px.scatter(
#         df,
#         x=x_col,
#         y=y_col,
#         hover_data=['index'],  # Include the index as a tooltip
#         title=f"Scatter Plot: {x_col} vs {y_col}"
#     )
#
#     # Add a diagonal line (y = x)
#     min_val = min(df[x_col].min(), df[y_col].min())  # Minimum value across both axes
#     max_val = max(df[x_col].max(), df[y_col].max())  # Maximum value across both axes
#
#     diagonal_line = go.Scatter(
#         x=[min_val, max_val],  # X values for the line (y = x)
#         y=[min_val, max_val],  # Y values for the line (y = x)
#         mode='lines',  # Draw a line
#         name='Diagonal Line (y = x)',  # Label for the line
#         line=dict(color='red', dash='dash')  # Customize line color and style
#     )
#
#     # Add the diagonal line to the figure
#     fig.add_trace(diagonal_line)
#
#     # Show the plot
#     fig.show()


# def visualize_fit(gene_counts, scaled_means):
#     """
#     Visualize the fit between gene_counts and scaled_means using Plotly.
#
#     Args:
#         gene_counts (pd.Series): Observed gene counts for a cell.
#         scaled_means_df (pd.DataFrame): Scaled expected gene expression values for the cell.
#     """
#     # Ensure gene_counts and scaled_means_df have the same index (gene names)
#     if not gene_counts.index.equals(scaled_means.index):
#         raise ValueError("gene_counts and scaled_means_df must have the same index.")
#
#     for column in scaled_means.columns:
#         # Create a scatter plot
#         fig = go.Figure()
#
#         # Add scatter plot: gene_counts vs. scaled_means
#         scatter_trace = go.Scatter(
#             x=scaled_means[column],
#             y=gene_counts,
#             mode='markers',
#             marker=dict(opacity=0.6),
#             text=gene_counts.index,  # Tooltip: gene names
#             name='Scatter Plot'
#         )
#         fig.add_trace(scatter_trace)
#
#         # Add a true diagonal line (y = x)
#         min_val = min(scaled_means[column].min(), gene_counts.min())  # Minimum value across both axes
#         max_val = max(scaled_means[column].max(), gene_counts.max())  # Maximum value across both axes
#
#         diagonal_line = go.Scatter(
#             x=[min_val, max_val],  # X values for the line (y = x)
#             y=[min_val, max_val],  # Y values for the line (y = x)
#             mode='lines',
#             line=dict(color='red', dash='dash'),
#             name='y = x'
#         )
#         fig.add_trace(diagonal_line)
#
#         # Update layout
#         fig.update_layout(
#             title=f'Gene Counts vs. Scaled Means ({column})',
#             xaxis_title=f'Scaled Means ({column})',
#             yaxis_title='Gene Counts',
#             showlegend=True
#         )
#
#         # Calculate correlation
#         correlation = gene_counts.corr(scaled_means[column])
#
#         # Calculate residuals and their sum
#         residuals = gene_counts - scaled_means[column]
#         sum_residuals = residuals.sum()
#
#         # Print correlation and sum of residuals
#         print(f"Correlation between gene_counts and {column}: {correlation:.3f}")
#         print(f"Sum of residuals for {column}: {sum_residuals:.3f}")
#
#         # Show the plot
#         fig.show()


def check_cell(obj, label, user_class, top_n=10, show_plot=True):
    """
    Compare gene expression likelihoods between two classes for a specific cell.

    Parameters:
        label (int): The cell number to analyze.
        user_class (str): The user-specified class to compare against.
        top_n (int): Number of top and bottom genes to retrieve (default: 10).

    Returns:
        gene_expression_data (pd.DataFrame): A DataFrame with columns:
            - (Cells typed as X, mean counts): Population-level. Average gene counts across ALL cells
              currently assigned to class X.
            - (Cells typed as X, NB expected): Cell-specific. What the NB model predicts THIS particular
              cell should have for each gene if it belonged to class X, accounting for this cell's theta
              (cell efficiency) and each gene's eta (gene efficiency). This is what actually drives the
              log-likelihood, not the population mean.
            - (This cell, counts): The actual observed gene counts for this cell.
        my_contr_df (pd.DataFrame): Per-gene log-likelihood contributions for the two classes.
        fig: The matplotlib figure (or None if show_plot=False).
    """

    # If original labels have been renumbered find the label it's been mapped to.
    if obj.config['label_map']:
        pciSeq_label = obj.config['label_map'][label]
    else:
        pciSeq_label = label

    # Step 1: Calculate gene log-likelihood contributions
    contr_df, gene_counts, scaled_means_df = obj.calculate_genes_log_likelihood_contr(label)

    # Step 2: Get the cell's class from cellData
    pciSeq_class = obj.cells.class_names[obj.cells.classProb[pciSeq_label].argmax()]

    # Step 3: Check if classes exist in contr_df
    if pciSeq_class not in contr_df.columns or user_class not in contr_df.columns:
        raise ValueError(f"One or both classes ({pciSeq_class}, {user_class}) not found in contr_df.")

    # Step 4: Calculate differences and get top/bottom genes
    my_contr_df = contr_df[[pciSeq_class, user_class]].copy()
    my_contr_df['diff'] = my_contr_df[pciSeq_class] - my_contr_df[user_class]

    top_genes = my_contr_df.nlargest(top_n, 'diff').index.values
    bottom_genes = my_contr_df.nsmallest(top_n, 'diff').index.values

    # Step 5: Combine top and bottom genes
    selected_genes = np.append(top_genes, bottom_genes)

    # Step 6: Retrieve mean expression and gene counts
    # gene_expression_data = obj.single_cell.mean_expression.loc[selected_genes, [pciSeq_class, user_class]]
    # gene_expression_data = gene_expression_data.merge(
    #     gene_counts[selected_genes].rename('Cell Gene Counts'),
    #     left_index=True,
    #     right_index=True
    # )

    # Step 6: Retrieve mean expression and gene counts
    gene_expression_data = pd.DataFrame(
        obj.cells.mean_gene_reads_per_class(),
        columns=obj.cells.class_names
    ).set_index(obj.genes.gene_panel)

    # Filter rows and columns
    gene_expression_data = gene_expression_data.loc[selected_genes, [pciSeq_class, user_class]]

    # Merge with gene_counts and expected counts (scaled_means)
    expected_counts = scaled_means_df.loc[selected_genes, [pciSeq_class, user_class]]
    gene_expression_data = gene_expression_data.merge(
        expected_counts, left_index=True, right_index=True, suffixes=('_mean', '_expected')
    )
    gene_expression_data = gene_expression_data.merge(
        gene_counts[selected_genes].rename('Cell Gene Counts'),
        left_index=True,
        right_index=True
    )

    # Add the MultiIndex header
    new_columns = pd.MultiIndex.from_tuples([
        (f'Cells typed as {pciSeq_class}', 'mean counts'),
        (f'Cells typed as {user_class}', 'mean counts'),
        (f'Cell {label} NB prediction', f'as {pciSeq_class}'),
        (f'Cell {label} NB prediction', f'as {user_class}'),
        (f'This cell: ({label})', 'observed')
    ])
    gene_expression_data.columns = new_columns

    # Step 7: Compute the prior and MRF terms for the two classes
    class_names = list(obj.cells.class_names)
    pciSeq_idx = class_names.index(pciSeq_class)
    user_idx = class_names.index(user_class)

    log_prior = obj.cellTypes.log_prior
    mrf = obj.cells.mrf

    gene_loglik_pciSeq = my_contr_df[pciSeq_class].sum()
    gene_loglik_user = my_contr_df[user_class].sum()
    log_prior_pciSeq = log_prior[pciSeq_idx]
    log_prior_user = log_prior[user_idx]
    mrf_pciSeq = mrf[pciSeq_label, pciSeq_idx]
    mrf_user = mrf[pciSeq_label, user_idx]

    # Full posterior over ALL classes, reconstructed from the same 3 components
    gene_loglik_all = contr_df.sum(axis=0).reindex(class_names).values
    log_post_all = gene_loglik_all + log_prior + mrf[pciSeq_label, :]
    full_post = softmax(log_post_all)
    full_pciSeq = full_post[pciSeq_idx]
    full_user = full_post[user_idx]

    fig = None
    if show_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # --- Top row: gene-level log-likelihood differences (unchanged) ---
        top_contribution_sum = my_contr_df.loc[top_genes, 'diff'].sum()
        bottom_contribution_sum = my_contr_df.loc[bottom_genes, 'diff'].sum()

        my_contr_df.loc[top_genes, 'diff'].plot.bar(ax=axes[0, 0], color='skyblue',
                                                    title=f'Cell: {label} - Top {top_n} contr for class: {pciSeq_class} (Sum: {top_contribution_sum:.2f})')
        axes[0, 0].set_ylabel('Log-Likelihood Difference')
        axes[0, 0].set_xlabel('Genes')

        my_contr_df.loc[bottom_genes, 'diff'].plot.bar(ax=axes[0, 1], color='lightcoral',
                                                       title=f'Cell: {label} - Top {top_n} contr for class: {user_class} (Sum: {bottom_contribution_sum:.2f})')
        axes[0, 1].set_ylabel('Log-Likelihood Difference')
        axes[0, 1].set_xlabel('Genes')

        # --- Bottom-left: grouped bar chart of log-posterior components ---
        x = np.arange(3)
        width = 0.35
        vals_pciSeq = [gene_loglik_pciSeq, log_prior_pciSeq, mrf_pciSeq]
        vals_user = [gene_loglik_user, log_prior_user, mrf_user]

        axes[1, 0].bar(x - width/2, vals_pciSeq, width, label=pciSeq_class, color='skyblue')
        axes[1, 0].bar(x + width/2, vals_user, width, label=user_class, color='lightcoral')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(['Gene LogLik', 'Log Prior', 'MRF'])
        axes[1, 0].set_ylabel('Log-scale value')
        axes[1, 0].set_title(f'Cell: {label} - Log-posterior components')
        axes[1, 0].legend()
        axes[1, 0].axhline(y=0, color='grey', linestyle='--', linewidth=0.5)

        # --- Bottom-right: posterior probabilities ---
        axes[1, 1].bar([pciSeq_class, user_class],
                       [full_pciSeq * 100, full_user * 100],
                       color=['skyblue', 'lightcoral'])
        axes[1, 1].set_ylabel('Posterior Probability (%)')
        axes[1, 1].set_title(f'Cell: {label} - Posterior probabilities')

        plt.tight_layout()
        plt.show()

    return gene_expression_data, my_contr_df, (fig if show_plot else None)


def cell_typing_breakdown(obj, label, weights=None, show_plot=True):
    """
    Follow cell-typing step-by-step for a given cell and assuming spot assignment is known

    Parameters:
        obj: The VarBayes object
        label (int): The cell label to analyze
        weights: Optional override for initial Dirichlet alpha.
            - dict: Same semantics as config['cell_type_weights']
              {'default': value_1, 'Class_1': value_2, ..., 'Class_n': value_n}.
              Unknown class keys are ignored with a warning. Values map by name
              to the order in obj.cellTypes.names.
            - 1D array-like: Explicit alpha vector of length K matching
              obj.cellTypes.names order. When using an array, you must include
              the entry for the 'Zero' class yourself.
        show_plot (bool): Whether to display plots (default: True)

    Returns:
        dict: Contains all intermediate values and final probabilities
    """

    # Get configuration
    prior_mode = obj.config.get('cell_type_prior', 'uniform')

    if prior_mode != 'weighted':
        logger.warning(
            f"Function available only for 'weighted' cell type prior mode."
        )
        return dict()

    # Step 1: Get initial alpha (from config weights) or override
    def _build_alpha_from_dict(dct, names):
        """Build alpha vector following the same logic as cell_type_weights.

        - Start from default=1 (or provided)
        - Override per-class entries when present
        - Ignore unknown keys with a warning
        """
        default_val = dct.get('default', 1)
        # Initialize with defaults
        vals = {name: default_val for name in names}
        # Apply overrides
        for key, val in dct.items():
            if key == 'default':
                continue
            if key not in names:
                logger.warning(
                    f"Cell type '{key}' in weights dict not found in cell type names. Ignoring.")
                continue
            vals[key] = val
        # Handle Zero if not explicitly provided
        # if 'Zero' not in dct:
        #     non_zero_names = [n for n in names if n != 'Zero']
        #     vals['Zero'] = float(np.sum([vals[n] for n in non_zero_names]))
        # Return in the exact order of names
        return np.array([float(vals[n]) for n in names], dtype=float)

    names = obj.cellTypes.names
    nK = obj.nK # number of classes (aka cell types) including 'Zero'


    if weights is not None:
        if isinstance(weights, dict):
            ini_alpha = _build_alpha_from_dict(weights, names)
            alpha_source = 'override'
        else:
            ini_alpha = np.asarray(weights, dtype=float)
            if ini_alpha.shape != (nK,):
                raise ValueError(f"weights must have shape ({nK},), got {ini_alpha.shape}")
            alpha_source = 'override'
    else:
        ini_alpha = obj.cellTypes.ini_alpha()
        alpha_source = 'default'

    # Step 2: Get observed class sizes (zeta). This is basically the number of cells in each class.
    zeta = obj.cells.classProb.sum(axis=0)

    # WARNING: DUPLICATED CODE. Steps 3 and 4 below are already in cellClass.
    # If I change something in CellClass, I need to change it here too.
    # It is OK for now, but If we develop cellClass any further this will be a problem.

    # Step 3: Updated alpha (what dalpha_upd does)
    updated_alpha = zeta + ini_alpha

    # Step 4: Compute log_prior from updated alpha
    if obj.single_cell.isMissing or prior_mode == 'weighted':
        log_prior = psi(updated_alpha) - psi(updated_alpha.sum())
    else:
        prior = updated_alpha / updated_alpha.sum()
        log_prior = np.log(prior)

    # Step 5: Get gene log-likelihood for this cell
    contr_df, _, _ = calculate_genes_log_likelihood_contr(obj, label)
    gene_loglik = contr_df.sum(axis=0).values  # Sum over genes

    # Step 6: Compute log posterior
    log_posterior = gene_loglik + log_prior

    # Step 7: Apply softmax to get final probabilities
    posterior_probs = softmax(log_posterior)

    # Store results
    out = {
        'label': label,
        'cell_type_names': obj.cellTypes.names,
        'ini_alpha': ini_alpha,
        'zeta': zeta,
        'updated_alpha': updated_alpha,
        'log_prior': log_prior,
        'gene_loglik': gene_loglik,
        'log_posterior': log_posterior,
        'posterior_probs': posterior_probs,
        'prior_mode': prior_mode,
        'alpha_source': alpha_source,
        'predicted_class': obj.cellTypes.names[np.argmax(posterior_probs)],
        'predicted_prob': np.max(posterior_probs)
    }

    if show_plot:
        _plot_classification_steps(out)

    return out


def _plot_classification_steps(data):
    """Helper function to plot the classification trace."""

    from plotly.subplots import make_subplots

    cell_type_names = data['cell_type_names']
    n_types = len(cell_type_names)

    # Compute cell class prior (softmax of log_prior)
    cell_class_prior = softmax(data['log_prior'])

    # Create subplots: 4 rows x 2 columns (leave last slot empty)
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            '<b>Step 1: Initial Alpha</b><br><sub>(from config weights)</sub>',
            '<b>Step 2: Updated Alpha</b><br><sub>(ini_alpha + zeta)</sub>',
            '<b>Step 3: Cell Class Log Prior</b><br><sub>(from updated alpha)</sub>',
            '<b>Step 4: Cell Class Prior</b><br><sub>(softmax of log prior)</sub>',
            '<b>Step 5: Cell Class Log-Likelihood</b><br><sub>(from gene expression data)</sub>',
            '<b>Step 6: Cell Class Log Posterior</b><br><sub>(log-likelihood + log prior)</sub>',
            '<b>Step 7: Cell Class Posterior</b><br><sub>(softmax of log posterior)</sub>',
            ''  # Empty placeholder
        ),
        # Reduce spacing to make each subplot taller (same overall size)
        vertical_spacing=0.08,
        # Slightly increase space between left and right columns
        horizontal_spacing=0.12
    )

    # Color scheme
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    bar_colors = [colors[i % len(colors)] for i in range(n_types)]

    # Plot 1: Initial alpha (Row 1, Col 1)
    fig.add_trace(go.Bar(
        x=cell_type_names,
        y=data['ini_alpha'],
        marker_color=bar_colors,
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>ini_alpha: %{y:.2f}<extra></extra>'
    ), row=1, col=1)

    # Plot 2: Updated alpha (Row 1, Col 2)
    fig.add_trace(go.Bar(
        x=cell_type_names,
        y=data['updated_alpha'],
        marker_color=bar_colors,
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>updated_alpha: %{y:.2f}<extra></extra>'
    ), row=1, col=2)

    # Plot 3: Cell Class Log Prior (Row 2, Col 1)
    fig.add_trace(go.Bar(
        x=cell_type_names,
        y=data['log_prior'],
        marker_color=bar_colors,
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>log_prior: %{y:.3f}<extra></extra>'
    ), row=2, col=1)

    # Plot 4: Cell Class Prior (Row 2, Col 2)
    fig.add_trace(go.Bar(
        x=cell_type_names,
        y=cell_class_prior * 100,
        marker_color=bar_colors,
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>prior: %{y:.2f}%<extra></extra>'
    ), row=2, col=2)

    # Plot 5: Cell Class Log-Likelihood (Row 3, Col 1)
    fig.add_trace(go.Bar(
        x=cell_type_names,
        y=data['gene_loglik'],
        marker_color=bar_colors,
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>log_likelihood: %{y:.1f}<extra></extra>'
    ), row=3, col=1)

    # Plot 6: Cell Class Log Posterior (Row 3, Col 2)
    fig.add_trace(go.Bar(
        x=cell_type_names,
        y=data['log_posterior'],
        marker_color=bar_colors,
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>log_posterior: %{y:.1f}<extra></extra>'
    ), row=3, col=2)

    # Plot 7: Cell Class Posterior (Row 4, Col 1) with highlight for winner
    max_idx = np.argmax(data['posterior_probs'])
    final_colors = [colors[i % len(colors)] if i != max_idx else '#e74c3c'
                   for i in range(n_types)]

    fig.add_trace(go.Bar(
        x=cell_type_names,
        y=data['posterior_probs'] * 100,
        marker_color=final_colors,
        showlegend=False,
        hovertemplate='<b>%{x}</b><br>posterior: %{y:.1f}%<extra></extra>'
    ), row=4, col=1)

    # Target subplot size based on provided screenshot dimensions (426x369 px)
    target_subplot_w = 426
    target_subplot_h = 369

    # Compute overall figure size to approximate per-subplot dimensions
    # Note: Plotly spacing is fractional, so this is an approximation.
    fig_width = target_subplot_w * 2 + 160  # margins/padding
    fig_height = target_subplot_h * 4 + 240  # margins/padding

    # Update layout using computed figure size
    alpha_note = "custom" if data.get('alpha_source') == 'override' else "default"
    fig.update_layout(
        height=fig_height,
        width=fig_width,
        title_text=(
            f"<span style='font-size:18px'><b>Cell {data['label']}: Classification Trace</b></span><br>"
            f"<span style='font-size:12px'>Predicted: {data['predicted_class']} ({data['predicted_prob'] * 100:.1f}%) | "
            f"Mode: {data['prior_mode']} | Alpha: {alpha_note}</span>"
        ),
        title_x=0.5,
        title_y=0.98,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=11),
        # Increase top margin to add padding between title and top row
        margin=dict(l=80, r=40, t=130, b=60)
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="ini_alpha", row=1, col=1)
    fig.update_yaxes(title_text="ini_alpha + zeta", row=1, col=2)
    fig.update_yaxes(title_text="Log Prior", row=2, col=1)
    fig.update_yaxes(title_text="Prior (%)", row=2, col=2)
    fig.update_yaxes(title_text="Log-Likelihood", row=3, col=1)
    fig.update_yaxes(title_text="Log Posterior", row=3, col=2)
    fig.update_yaxes(title_text="Posterior (%)", row=4, col=1)

    # Update x-axes
    for row in [1, 2, 3, 4]:
        for col in [1, 2]:
            fig.update_xaxes(tickangle=-45, row=row, col=col)

    fig.show()


def read_tsv(filepath):
    """
    Convenience function to read the tsv files generated by pciSeq
    """
    data = pd.read_csv(filepath, sep='\t')
    data = data.map(
        lambda x: eval(x) if isinstance(x, str) and x.strip().startswith(('{', '[', '(')) else x)
    return data


# def softmax(X: np.ndarray, theta: float = 1.0, axis: Optional[int] = None) -> np.ndarray:
#     """Compute the softmax of each element along an axis of X.
#
#     Args:
#         X: Input array (should be floats)
#         theta: Multiplier prior to exponentiation (default: 1.0)
#         axis: Axis to compute values along (default: first non-singleton axis)
#
#     Returns:
#         Array same size as X, normalized along the specified axis
#
#     Notes:
#         From https://nolanbconaway.github.io/blog/2017/softmax-numpy
#     """
#     # Make X at least 2d
#     y = np.atleast_2d(X)
#
#     # Find axis if not specified
#     if axis is None:
#         axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
#
#     # Multiply y against the theta parameter
#     y = y * float(theta)
#
#     # Subtract the max for numerical stability
#     y = y - np.expand_dims(np.max(y, axis=axis), axis)
#
#     # Exponentiate y
#     y = np.exp(y)
#
#     # Take the sum along the specified axis
#     ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
#
#     # Finally: divide elementwise
#     p = y / ax_sum
#
#     # Flatten if X was 1D
#     if len(X.shape) == 1:
#         p = p.flatten()
#
#     return p


def has_converged(
        spots: Any,
        p0: Optional[np.ndarray],
        tol: float
) -> Tuple[bool, float]:
    """Check if probability assignments have converged.

    Args:
        spots: Spot data object containing parent_cell_prob
        p0: Previous probability matrix (None for first iteration)
        tol: Convergence tolerance threshold

    Returns:
        Tuple containing:
            - bool: True if converged, False otherwise
            - float: Maximum absolute difference between iterations

    Raises:
        Exception: If convergence check fails
    """
    p1 = spots.parent_cell_prob
    if p0 is None:
        p0 = np.zeros_like(p1)

    try:
        delta = np.max(np.abs(p1 - p0))
        converged = (delta < tol)
        return converged, delta
    except Exception as e:
        logger.error(f"Convergence check failed: {str(e)}")
        raise


def scaled_exp(cell_area_factor: np.ndarray,
               sc_mean_expressions: np.ndarray) -> np.ndarray:
    """Calculate scaled expression values.

    Args:
        cell_area_factor: Cell area scaling factors
        sc_mean_expressions: Single cell mean expression values

    Returns:
        Scaled expression array
    """
    subscripts = 'c,gk->cgk'
    operands = [cell_area_factor, sc_mean_expressions]

    return oe.contract(subscripts, *operands, optimize='optimal')


def empirical_mean(spots, cells):

    # get the total gene counts per cell
    N_c = cells.total_counts

    xyz_spots = spots.xyz_coords
    prob = spots.parent_cell_prob
    n = cells.config['nNeighbors'] + 1

    # multiply the x coord of the spots by the cell prob
    a = np.tile(xyz_spots[:, 0], (n, 1)).T * prob

    # multiply the y coord of the spots by the cell prob
    b = np.tile(xyz_spots[:, 1], (n, 1)).T * prob

    # multiply the z coord of the spots by the cell prob
    c = np.tile(xyz_spots[:, 2], (n, 1)).T * prob

    # aggregated x and y coordinate
    idx = spots.parent_cell_id
    x_agg = npg.aggregate(idx.ravel(), a.ravel(), size=len(N_c))
    y_agg = npg.aggregate(idx.ravel(), b.ravel(), size=len(N_c))
    z_agg = npg.aggregate(idx.ravel(), c.ravel(), size=len(N_c))

    # get the estimated cell centers
    x_bar = np.nan * np.ones(N_c.shape)
    y_bar = np.nan * np.ones(N_c.shape)
    z_bar = np.nan * np.ones(N_c.shape)

    x_bar[N_c > 0] = x_agg[N_c > 0] / N_c[N_c > 0]
    y_bar[N_c > 0] = y_agg[N_c > 0] / N_c[N_c > 0]
    z_bar[N_c > 0] = z_agg[N_c > 0] / N_c[N_c > 0]

    # cells with N_c = 0 will end up with x_bar = y_bar = np.nan
    xyz_bar_fitted = np.array(list(zip(x_bar.T, y_bar.T, z_bar.T)))

    # if you have a value for the estimated centroid use that, otherwise
    # use the initial (starting values) centroids
    ini_cent = cells.ini_centroids()
    xyz_bar = np.array(tuple(zip(*[ini_cent['x'], ini_cent['y'], ini_cent['z']])))

    # # sanity check. NaNs or Infs should appear together
    # assert np.all(np.isfinite(x_bar) == np.isfinite(y_bar))
    # use the fitted centroids where possible otherwise use the initial ones
    xyz_bar[np.isfinite(x_bar)] = xyz_bar_fitted[np.isfinite(x_bar)]
    return pd.DataFrame(xyz_bar, columns=['x', 'y', 'z'], dtype=np.float32)
