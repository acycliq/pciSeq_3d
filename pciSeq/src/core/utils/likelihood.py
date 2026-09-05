"""How well a cell's gene counts match a class, under the negative binomial."""

import logging
import numpy as np
import pandas as pd
import opt_einsum as oe
from typing import Tuple
from pandas import DataFrame, Series

logger = logging.getLogger(__name__)


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
    scaled_means = obj.scaled_exp

    # Calculate scaled expression adjusted by gene efficiency and regularization
    ScaledExp = np.einsum('cgk,g,ck->cgk', scaled_means, obj.genes.eta_bar, obj.cells.theta_bar) + obj.config['SpotReg']

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
    scaled_means = obj.scaled_exp
    cgc = obj.cells.geneCount

    # Return values for the specified cell
    contr_df = pd.DataFrame(contr[label], columns=obj.cells.class_names).set_index(obj.genes.gene_panel)
    gene_counts = pd.Series(cgc[label], index=obj.genes.gene_panel)
    scaled_means_df = pd.DataFrame(scaled_means[label], columns=obj.cells.class_names).set_index(obj.genes.gene_panel)
    return contr_df, gene_counts, scaled_means_df


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
