# Standard library imports
import logging
from typing import Tuple, Dict

# Third party imports
import numpy as np
import pandas as pd
import scipy
from natsort import natsort_keygen

# Local imports
from ..utils.cell_utils import read_image_objects, keep_labels_unique

logger = logging.getLogger(__name__)

# name of the class with no reference, see config['unclassified_class']
UNCLASSIFIED = 'Unclassified'


def unclassified_start(values: np.ndarray) -> Tuple[np.ndarray, int]:
    """Starting values of the Unclassified class, from the genes x classes
    array of the reference.

    For a given k the candidate is, gene by gene, the mean of the k highest
    values of that gene over the classes. k=1 is the max, k = number of classes
    is the plain mean. Every k is tried and the one kept is the candidate whose
    closest class is the furthest away (lowest cosine to its nearest class). A
    start that looks like one of the classes would take that class's cells.

    Returns the chosen column and its k.
    """
    values = np.asarray(values, dtype=np.float64)
    nK = values.shape[1]
    unit = values / np.linalg.norm(values, axis=0, keepdims=True)
    # column j is the mean of the j+1 highest values of each gene
    candidates = np.cumsum(-np.sort(-values, axis=1), axis=1) / np.arange(1, nK + 1)
    candidates_unit = candidates / np.linalg.norm(candidates, axis=0, keepdims=True)
    nearest = np.einsum('gj, gk -> jk', candidates_unit, unit).max(axis=1)
    j = int(np.argmin(nearest))
    return candidates[:, j], j + 1


class SingleCell(object):
    """
    Handles single-cell RNA sequencing reference data, including mean
    expression levels per cell type. Supports integration of single-cell
    data into broader analyses.

    Attributes:
        config (dict): Configuration parameters for single-cell data.
        _mean_expression (pd.DataFrame): Mean expression levels.
        _log_mean_expression (pd.DataFrame): Log mean expression levels.
    """

    def __init__(self, scdata: pd.DataFrame, genes: np.ndarray, config: Dict):
        """
        Initializes the SingleCell object with single-cell data and configuration.

        Parameters:
            scdata (pd.DataFrame): Single-cell data.
            genes (np.array): Array of gene names.
            config (dict): Configuration parameters for single-cell data.
        """
        # otherwise, if they are unknown, this will be set to True and the algorithm will
        # try to estimate them
        # self.raw_data = self._raw_data(scdata, genes)
        self.config = config
        self._mean_expression_adj, self._log_mean_expression_adj = self._setup(scdata, genes, self.config)

    def _setup(self, scdata: pd.DataFrame, genes: np.ndarray, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Sets up the mean and log mean expression levels.

        Parameters:
            scdata (pd.DataFrame): Single-cell data.
            genes (np.array): Array of gene names.
            config (dict): Configuration parameters.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Mean and log mean expression levels.
        """
        expr = self._raw_data(scdata, genes)
        self.raw_data = expr

        # get the mean (and log-mean) expression data by cell type.
        # Figures have been scaled by the gene inefficiency
        me_adj, lme_adj = self._helper(expr.copy())

        assert me_adj.columns[-1] == 'Zero', "Last column should be the Zero class"
        assert lme_adj.columns[-1] == 'Zero', "Last column should be the Zero class"
        return me_adj.astype(np.float32), lme_adj.astype(np.float32)

    # -------- PROPERTIES -------- #
    @property
    def mean_expression_adj(self):
        """Returns the mean expression levels adjusted by the initial gene inefficiency."""
        assert self._mean_expression_adj.columns[-1] == 'Zero', "Last column should be the Zero class"
        return self._mean_expression_adj

    @property
    def log_mean_expression(self):
        """Returns the log mean expression levels adjusted by the initial gene inefficiency."""
        assert self._log_mean_expression_adj.columns[-1] == 'Zero', "Last column should be the Zero class"
        return self._log_mean_expression_adj

    @property
    def mean_expression(self):
        """Returns the mean gene counts per cell class"""
        return self.mean_expression_adj / self.config['Inefficiency']

    @property
    def genes(self):
        """
        Returns the gene names.
        WARNING: you can get the gene names from the Genes object and its gene_panel property.
        They should be the same but is there any value having two places for the same thing?
        """
        return self.mean_expression_adj.index.values

    @property
    def classes(self):
        """Returns the class names."""
        return self.mean_expression_adj.columns.values

    @property
    def unclassified_idx(self):
        """Column of the Unclassified class, None when the run has no such class."""
        cols = list(self._mean_expression_adj.columns)
        return cols.index(UNCLASSIFIED) if UNCLASSIFIED in cols else None

    def set_unclassified(self, mu_adj: np.ndarray, log_mu: np.ndarray) -> None:
        """Overwrites the Unclassified column of the two tables with a new
        estimate, one value per gene, in the order of the gene panel.

        mu_adj is on the scale of mean_expression_adj, so the steps that add
        SpotReg to the table get the mean of the class back. log_mu goes into
        the log table as it is. For every other column the log table is
        log(mean_expression_adj + SpotReg); for this one it is the expected
        log of an unknown, which is lower than the log of its mean, so it is
        handed in and not worked out from mu_adj.
        """
        self._mean_expression_adj[UNCLASSIFIED] = np.asarray(mu_adj, dtype=np.float32)
        self._log_mean_expression_adj[UNCLASSIFIED] = np.asarray(log_mu, dtype=np.float32)

    # Helper functions #
    def _set_axes(self, df):
        """
        Sets the axes labels for a DataFrame.
        """
        df = df.rename_axis("class_name", axis="columns").rename_axis('gene_name')
        return df

    def _remove_zero_cols(self, df):
        """
        Removes columns with all zero values from a DataFrame.
        """
        out = df.loc[:, (df != 0).any(axis=0)]
        return out

    def _helper(self, expr):
        """
        Helper function to process expression data.

        Parameters:
            expr (pd.DataFrame): Expression data.

        Returns:
            tuple: Processed mean and log-mean expression data.
        """

        # The Unclassified class, when asked for. It has no reference, so it
        # starts from the reference of the other classes (see unclassified_start)
        # and VarBayes.unclassified_upd re-estimates it on every pass. It goes in
        # before the sort, so it sits among the other classes by name and Zero
        # stays the last column.
        if self.config.get('unclassified_class', False):
            if UNCLASSIFIED in expr.columns:
                raise ValueError("The cell type definitions already have a class named '%s'. "
                                 "Rename it or set unclassified_class to False." % UNCLASSIFIED)
            start, k = unclassified_start(expr.values)
            logger.info("Unclassified class starts as the mean of the top %d of %d classes per gene"
                        % (k, expr.shape[1]))
            expr[UNCLASSIFIED] = start

        # order by column name
        expr = expr.copy().sort_index(axis=0).sort_index(axis=1, key=natsort_keygen(key=lambda y: y.str.lower()))

        # append at the end the Zero class
        expr['Zero'] = np.zeros([expr.shape[0], 1])
        me = expr.rename_axis('gene_name').rename_axis("class_name", axis="columns")

        # apply the inefficiency
        me = me * self.config['Inefficiency']

        # log mean expression
        lme = np.log(me + self.config['SpotReg'])
        return me, lme
    def _raw_data(self, scdata: pd.DataFrame, genes: np.ndarray) -> pd.DataFrame:
        """
        Processes raw single-cell data, filtering out any genes outside the gene panel and grouping by cell type.

        Parameters:
            scdata (pd.DataFrame): Single-cell data.
            genes (np.array): Array of gene names.

        Returns:
            pd.DataFrame: Processed single-cell data.
        """
        assert np.all(scdata >= 0), "Single cell dataframe has negative values"
        logger.info('Single cell data passed-in have %d genes and %d cells' % (scdata.shape[0], scdata.shape[1]))
        logger.info('Single cell data: Keeping counts for the gene panel of %d only' % len(genes))
        df = scdata.loc[genes]

        # set the axes labels
        df = self._set_axes(df)

        # remove any rows with the same gene label
        df = keep_labels_unique(df)

        df = self._remove_zero_cols(df.copy())
        dfT = df.T

        logger.info('Single cell data: Grouping gene counts by cell type. Aggregating function is the mean.')
        out = dfT.groupby(dfT.index.values).agg('mean').T
        logger.info('Grouped single cell data have %d genes and %d cell types' % (out.shape[0], out.shape[1]))
        return out
