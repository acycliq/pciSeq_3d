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
