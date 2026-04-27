"""
Core Algorithm Implementation Module for pciSeq

This module implements the main Variational Bayes algorithm for spatial transcriptomics
analysis, primarily through the VarBayes class. The algorithm iteratively:
1. Assigns spots to cells
2. Determines cell types
3. Estimates gene expression patterns
4. Updates model parameters

Key Components:
-------------
VarBayes:
    Main class implementing the iterative algorithm with methods for:
    - Gene count updates
    - Cell type assignment
    - Spot-to-cell assignment
    - Parameter estimation (eta, gamma, covariance)
    - Model convergence checking

Algorithm Steps:
--------------
1. Initialization:
   - Set prior probabilities
   - Initialize cell assignments
   - Set gene efficiency parameters

2. Iterative Updates:
   - Update expected gene counts
   - Calculate gamma expectations
   - Update gaussian parameters
   - Assign cells to types
   - Assign spots to cells
   - Update gene efficiency
   - Update Dirichlet parameters
   - Update single-cell reference

3. Convergence:
   - Check for convergence after each iteration
   - Return results when converged or max iterations reached

Notes:
-----
- Uses Redis for optional diagnostic monitoring
- Implements equations from the pciSeq paper
- Handles missing single-cell reference data
- Supports parallel processing via numpy operations

Dependencies:
-----------
- numpy: For numerical computations
- pandas: For data management
- scipy: For statistical operations
- numpy_groupies: For group operations
- dask: For delayed computations
"""
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import sys
import numpy as np
import numpy_groupies as npg
import pandas as pd
from dask.delayed import delayed
from scipy.special import softmax
from sklearn.cluster import DBSCAN
import opt_einsum as oe

# Local imports
from .datatypes.cells import Cells
from .datatypes.genes import Genes
from .datatypes.spots import Spots
from .datatypes.singleCell import SingleCell
from .datatypes.cellClass import CellClass
from .summary import collect_data
from .utils.elbo import calc_elbo
# from .analysis import CellExplorer
from .utils import ops_utils as utils
from .utils import visualisation
from ...src.diagnostics.controller.diagnostic_controller import DiagnosticController
import joblib

# Configure logging
logger = logging.getLogger(__name__)


class VarBayes:
    """
    Implements Variational Bayes algorithm for spatial transcriptomics analysis.

    This class performs cell type assignment and spot-to-cell mapping using a
    probabilistic model with variational inference.

    Args:
        cells_df: DataFrame containing cell information
        spots_df: DataFrame containing spot information
        scRNAseq: Single-cell RNA sequencing reference data
        config: Configuration dictionary containing algorithm parameters
    """

    def __init__(self,
                 cells_df: pd.DataFrame,
                 spots_df: pd.DataFrame,
                 scRNAseq: pd.DataFrame,
                 config: Dict[str, Any]) -> None:
        """Initialize components and setup."""
        # Explicitly declare important instance attributes
        self.diagnostic_controller: Optional[DiagnosticController] = None  # For real-time diagnostics
        self.config = None
        self.iter_num = None
        self.iter_delta = []
        self.has_converged = False
        self.on_iteration_callback = None  # Optional callback for real-time visualization

        # Birth-debug tracking: filled in by birth_cells the first time it
        # actually creates cells. _birth_debug_dump uses these to log the
        # evolution of those specific spots/cells every iteration.
        self._birth_debug_spots: Optional[np.ndarray] = None
        self._birth_debug_cells: Optional[np.ndarray] = None
        self._birth_debug_iter: Optional[int] = None

        # Initialize components
        self._validate_config(config)
        self.config = config
        self._setup_diagnostics()
        self._setup_components(cells_df, spots_df, scRNAseq)
        self._setup_dimensions()

        # Placeholder for other attributes
        self._scaled_exp = None
        # self._cell_explorer: Optional[CellExplorer] = None

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Check for required config parameters."""
        required = ['exclude_genes', 'max_iter', 'CellCallTolerance',
                    'rGene', 'Inefficiency', 'InsideCellBonus', 'MisreadDensity',
                    'cell_centroid_prior', 'cell_cov_prior', 'SpotReg', 'nNeighbors', 'rSpot',
                    'save_data', 'output_path', 'launch_viewer', 'launch_diagnostics',
                    'is_redis_running', 'cell_radius', 'cell_type_prior', 'is3D',
                    'mean_gene_counts_per_class', 'mean_gene_counts_per_cell']
        missing = [param for param in required if param not in config]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

    def _setup_diagnostics(self) -> None:
        """Initialize diagnostics controller if enabled in config."""
        self.diagnostic_controller = None
        if not self.config.get('launch_diagnostics', False):
            return

        try:
            self.diagnostic_controller = DiagnosticController()
            if not self.diagnostic_controller.launch_dashboard():
                logger.warning("Failed to launch diagnostics dashboard")
                self.diagnostic_controller = None
        except Exception as e:
            logger.warning(f"Failed to initialize diagnostics: {e}")
            self.diagnostic_controller = None

    def _setup_components(self, cells_df, spots_df, scRNAseq) -> None:
        """Set up the core data components needed for the algorithm."""
        self.cells = Cells(cells_df, self.config)
        self.spots = Spots(spots_df, self.config)
        self.genes = Genes(self.spots, self.config)
        self.single_cell = SingleCell(scRNAseq, self.genes.gene_panel, self.config)
        self.cellTypes = CellClass(self.single_cell, self.config)
        self.cells.class_names = self.single_cell.classes

    def _setup_dimensions(self) -> None:
        """Set up core dimensions."""
        self.nC = self.cells.nC  # cells
        self.nG = self.genes.nG  # genes
        self.nK = self.cellTypes.nK  # classes
        self.nS = self.spots.nS  # spots
        self.nN = self.config['nNeighbors'] + 1  # neighbors + background

    def initialise_state(self) -> None:
        """Initialises the starting state of the objects
        4-Feb-2025: Note that inefficiency (denoted by eta) follows a Gamma(rGene, rGene).
        Keep in mind also that there is also the config['Inefficiency'] parameter that has
        been applied directly to the expression data from scRNAseq
        """
        self.cellTypes.ini_prior()
        self.cells.nbrs = self.cells.nearest_neighbours()
        self.cells.classProb = np.tile(self.cellTypes.prior, (self.nC, 1))
        self.genes.init_eta(self.config['rGene'], self.config['rGene'])
        self.spots.parent_cell_id = self.spots.cells_nearby(self.cells)[0]
        self.spots.parent_cell_prob = self.spots.ini_cellProb(self.spots.parent_cell_id, self.config)
        self.cells._ini_gene_counts = np.bincount(self.spots.data.label.values, minlength=self.nC)
        self.genes._misread_density = self.genes.calc_misread_density()
        A_total = self.config['img_dim']['w'] * self.config['img_dim']['h'] * self.config['img_dim']['n_planes']
        self.genes.init_rho(self.config['MisreadDensity']['default'], A_total)
        self.spots.init_gamma(self.config['rSpot'], self.config['rSpot'], [self.nC, self.nG, self.nK])
        self.init_theta()

    def init_theta(self) -> None:
        geneCounts = self.cells.ini_gene_counts
        alpha = geneCounts + self.config['rTheta'] - 1



        mu = self.single_cell.mean_expression_adj + self.config['SpotReg']
        area_factor = self.cells.ini_cell_props['area_factor']
        gamma_bar = self.spots.gamma_bar.compute()
        eta_bar = self.genes.eta_bar

        beta = np.einsum('c, cgk, g, gk -> ck',
                         area_factor,
                         gamma_bar,
                         eta_bar,
                         mu) + self.config['rTheta']

        self.cells.calc_theta(alpha, beta)


    def __getstate__(self):
        """
        Get state for pickling.
        Removes diagnostics-related attributes to enable pickling and reduce file size.
        """
        attributes = self.__dict__.copy()
        if 'diagnostic_controller' in attributes:
            del attributes['diagnostic_controller']
        if 'on_iteration_callback' in attributes:
            del attributes['on_iteration_callback']
        return attributes

    @property
    def scaled_exp(self):
        """
        Get scaled expression values.

        Returns:
            delayed: Dask delayed object containing scaled expression computation
        """
        return self._scaled_exp


    # -------------------------------------------------------------------- #
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.initialise_state()
        cell_df, gene_df = self.main_loop()
        return cell_df, gene_df

    # -------------------------------------------------------------------- #
    def main_loop(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main algorithm loop with diagnostic updates."""
        """
        Executes the main Variational Bayes algorithm loop.

        Iteratively updates:
            1. Gene counts per cell
            2. Gamma parameters
            3. Gaussian parameters (if needed)
            4. Cell type assignments
            5. Spot-to-cell assignments
            6. Gene efficiency parameters
            7. Dirichlet parameters (if needed)
            8. Expression means (if needed)

        The loop continues until either:
            - Convergence is reached (change in probabilities below tolerance)
            - Maximum iterations are reached

        Args:
            None

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - Cell dataframe with final assignments and probabilities
                - Gene dataframe with expression statistics

        Note:
            Progress is published to Redis if enabled.
        """
        p0 = None
        cell_df = None
        gene_df = None
        max_iter = self.config['max_iter']

        try:
            for i in range(max_iter):
                self.iter_num = i

                # 1. For each cell, calc the expected gene counts
                self.geneCount_upd()

                # 2. update gene-specific misread density
                self.rho_upd()

                # 3. calc the gene inefficiency
                self.eta_upd()

                # 4. calc the cell inefficiency
                self.theta_upd()

                # 5. calc expected gamma
                self.gamma_upd()

                logger.info("gaussian_upd step has been removed in this version of the software")
                # 3 update correlation matrix and variance of the gaussian distribution
                # if self.single_cell.isMissing or (self.config['InsideCellBonus'] is False) or (self.config['is3D']):
                #     self.gaussian_upd()

                # 6. assign cells to cell types
                self.cell_to_cellType()

                # 7. update the dirichlet distribution
                if self.single_cell.isMissing or (self.config['cell_type_prior'] == 'weighted'):
                    self.dalpha_upd()

                # 8. Update single cell data
                if self.single_cell.isMissing:
                    self.mu_upd()

                # 9. assign spots to cells
                self.spots_to_cell()

                # 10. birth move: instantiate new cells over blobs of
                # background-dominated spots (see detect_birth_candidates /
                # birth_cells). Gated on config + warmup + cadence.
                if self.config.get('birth_enabled', False) \
                        and i >= self.config.get('birth_warmup_iters', 5) \
                        and (i - self.config.get('birth_warmup_iters', 5)) \
                            % self.config.get('birth_every', 5) == 0:
                    candidate_spots = self.detect_birth_candidates()
                    if candidate_spots.size > 0:
                        self.birth_cells(candidate_spots)

                # Per-iter trace of tracked birthed cells/spots (no-op if
                # nothing has been birthed yet)
                self._birth_debug_dump()

                # # Calculate ELBO
                # elbo = calc_elbo(self)
                # logger.info('Iteration %d, ELBO: %f' % (i, elbo))

                self.has_converged, delta = utils.has_converged(
                    self.spots, p0, self.config['CellCallTolerance']
                )
                logger.info('Iteration %d, mean prob change %f' % (i, delta))
                # --- SMART LOGGING --- 
                if delta > 0:
                    p1 = self.spots.parent_cell_prob
                    p0_val = p0 if p0 is not None else np.zeros_like(p1)
                    diffs = np.abs(p1 - p0_val)
                    max_idx = np.unravel_index(np.argmax(diffs), diffs.shape)
                    spot_idx = max_idx[0]
                    col_idx = max_idx[1]
                    gene_name = self.spots.data.gene_name.iloc[spot_idx]
                    cell_id = self.spots.parent_cell_id[spot_idx, col_idx]
                    old_prob = p0_val[spot_idx, col_idx]
                    new_prob = p1[spot_idx, col_idx]
                    logger.info(f"DIAGNOSTIC: Spot {spot_idx} (Gene: {gene_name}) changed by {delta:.6f}")
                    logger.info(f"DIAGNOSTIC: Cell {cell_id} Prob: {old_prob:.4f} -> {new_prob:.4f}")

                # Update diagnostics using controller
                self.diagnostics_upd()

                # Call real-time viewer callback if provided
                if self.on_iteration_callback is not None:
                    try:
                        self.on_iteration_callback(self.cells.classProb, i, delta)
                    except Exception as e:
                        logger.warning(f"Real-time viewer callback failed: {e}")

                # keep track of the deltas
                self.iter_delta.append(delta)

                # replace p0 with the latest probabilities
                p0 = self.spots.parent_cell_prob

                if self.has_converged:
                    # self.cell_analysis(35975)
                    cell_df, gene_df = collect_data(self.cells, self.spots, self.genes, self.config['is3D'])
                    break

                if i == max_iter - 1:
                    logger.info('Loop exhausted. Exiting with convergence status: %s' % self.has_converged)
                    cell_df, gene_df = collect_data(self.cells, self.spots, self.genes, self.config['is3D'])
                    break
        finally:
            # Ensure diagnostics are properly shut down
            if self.diagnostic_controller is not None:
                try:
                    self.diagnostic_controller.shutdown()
                except Exception as e:
                    logger.warning(f"Failed to shutdown diagnostics: {e}")

        return cell_df, gene_df

    # -------------------------------------------------------------------- #
    def geneCount_upd(self) -> None:
        """
        Updates the gene count matrix for each cell.

        Produces a matrix numCells-by-numGenes where element at position (c,g) keeps
        the expected counts of gene g in cell c. The first row corresponds to the
        background counts (spots not assigned to any cell).

        Note:
            The sum of background spots and cell gene counts should equal
            the total number of spots.
        """
        # make an array nS-by-nN and fill it with the spots id
        gene_ids = np.tile(self.spots.gene_id, (self.nN, 1)).T

        # flatten it
        gene_ids = gene_ids.ravel()

        # make corresponding arrays for cell_id and probs
        cell_ids = self.spots.parent_cell_id.ravel()
        probs = self.spots.parent_cell_prob.ravel()

        # make the array to be used as index in the group-by operation
        group_idx = np.vstack((cell_ids, gene_ids))

        # For each cell aggregate the number of spots from the same gene.
        # It will produce an array of size nC-by-nG where the entry at (c,g)
        # is the gene counts of gene g within cell c
        N_cg = npg.aggregate(group_idx, probs, size=(self.nC, self.nG))

        # assert N_cg.sum() == self.spots.data.shape[0], \
        #     "The sum of the background spots and the cell gene counts should be equal to the total number of spots"

        # make output. This part needs to be rewritten
        out = np.zeros([self.nC, self.nG], dtype=np.float32)
        out[1:, :] = N_cg[1:, :]

        # cell at position zero is the background
        self.cells.background_counts = N_cg[0, :]
        # Actual cells are on non-zero positions
        self.cells.geneCount = out

    # -------------------------------------------------------------------- #
    def gamma_upd(self) -> None:
        """
        Updates gamma parameters for the negative binomial distribution.

        Implements equation (3) of the Qian paper. Calculates the expected gamma
        values using scaled expression and spot regularization parameters.

        Updates:
            - self._scaled_exp: Delayed computation of scaled expression
            - self.spots._log_gamma_bar: Log of expected gamma values
            - self.spots._gamma_bar: Expected gamma values
        """
        cells = self.cells
        cfg = self.config

        self._scaled_exp = delayed(utils.scaled_exp(cells.ini_cell_props['area_factor'],
                                                    self.single_cell.mean_expression_adj.values))

        beta = self.scaled_exp.compute() * self.genes.eta_bar[:, None] * self.cells.theta_bar[:,None, :]+ cfg['rSpot']
        rho = cfg['rSpot'] + cells.geneCount

        self.spots._post_shape = rho
        self.spots._post_rate = beta
        self.spots._log_gamma_bar = delayed(self.spots.logGammaExpectation(rho, beta))
        self.spots._gamma_bar = delayed(self.spots.gammaExpectation(rho, beta))
        self.spots.my_gamma_bar = self.spots._gamma_bar.compute()

    # -------------------------------------------------------------------- #
    def cell_to_cellType(self) -> None:
        """
        Updates cell type assignment probabilities.

        Implements equation (2) of the Qian paper. Returns an array of size
        numCells-by-numCellTypes where element in position [i,j] keeps the
        probability that cell i has cell type j.

        The computation combines:
            1. Negative binomial log-likelihood for gene expression
            2. Cell type priors
            3. Softmax normalization for final probabilities
        """

        # Get the full log-likelihood matrix using shared computation
        contr = utils.compute_gene_loglikelihood_matrix(self)

        label_map = self.config['label_map']
        inv_label_map = {v:k for k,v in label_map.items()}

        df_list = [pd.DataFrame(d, columns=self.cells.class_names) for d in contr]

        # populate the genes' contributions to the negative loglik. Property 'nb_contr' is only useful
        # for debugging, safe to remove in the future
        self.cells.nb_contr = contr
        contr = np.sum(contr, axis=1)
        mrf = self.cells.calc_mrf()
        # stash mrf for debugging (same pattern as nb_contr above)
        self.cells.mrf = mrf
        # mrf = self.cells.classProb[self.cells.nbrs].sum(axis=1)
        wCellClass = contr + self.cellTypes.log_prior + mrf
        pCellClass = softmax(wCellClass, axis=1)

        # # save the data to a tmp dir
        # if (self.iter_num < 10) or (self.iter_num > 70):
        #     from pathlib import Path
        #     out_dir = Path("/tmp/pciSeq/data/flatfiles") / f"iter_{self.iter_num}"
        #     out_dir.mkdir(parents=True, exist_ok=True)
        #
        #     for i, d in enumerate(df_list):
        #         d.to_csv(out_dir / f"contr_{i}.csv", index=False)
        #
        #     pd.DataFrame(mrf, columns=self.cells.class_names).to_csv(out_dir / "mrf.csv")
        #
        #     # if log_prior is (K,) make it a single row; if it's already (1,K) or (N,K) this also works if you adjust
        #     pd.DataFrame([self.cellTypes.log_prior], columns=self.cells.class_names).to_csv(out_dir / "log_prior.csv")
        #     logger.info(f"[iter {self.iter_num}] Saving debug CSVs to: {out_dir}")

        self.cells.classProb = pCellClass

    # -------------------------------------------------------------------- #
    def spots_to_cell(self) -> None:
        """
        Updates spot-to-cell assignment probabilities.

        Implements equation (4) of the Qian paper. For each spot, calculates the
        probability of it belonging to each nearby cell or being a misread.

        The computation includes:
            1. Expected gene expression for each cell type
            2. Gamma parameter contributions
            3. Gene efficiency factors
            4. Spatial distance likelihood
            5. Inside-cell bonus for spots within cell boundaries
            6. Misread probability for background noise

        Note:
            Updates spot-cell assignments and triggers gene count update.
        """
        nN = self.nN
        nS = self.spots.data.gene_name.shape[0]

        wSpotCell = np.zeros([nS, nN], dtype=np.float64)
        gn = self.spots.data.gene_name.values
        expected_counts = self.single_cell.log_mean_expression.loc[gn].values
        logeta_bar = self.genes.logeta_bar[self.spots.gene_id]

        # Gene-specific misread density (learned per gene)
        log_rho = self.genes.log_rho_bar[self.spots.gene_id]

        # pre-populate last column
        wSpotCell[:, -1] = log_rho
        mvn_loglik_arr = np.zeros(wSpotCell.shape)
        attention = np.zeros(wSpotCell.shape)
        expr_fluctuations = np.zeros(wSpotCell.shape)
        cell_inefficiency = np.zeros(wSpotCell.shape)
        gene_inefficiency = np.zeros(wSpotCell.shape)

        # Materialize once before the loop (same for all neighbors)
        log_gamma_bar_arr = self.spots.log_gamma_bar.compute()

        # loop over the first nN-1 closest cells. The nN-th column is reserved for the misreads
        for n in range(nN - 1):
            # get the spots' nth-closest cell
            sn = self.spots.parent_cell_id[:, n]

            # get the respective cell type probabilities
            cp = self.cells.classProb[sn]
            log_theta_bar = np.log(self.cells.theta_bar[sn])
            # multiply and sum over cells. In practice this means that when high expected counts
            # are aligned with high cell class probs this term will be high
            term_1 = np.einsum('ij, ij -> i', expected_counts, cp)

            log_gamma_bar = log_gamma_bar_arr[self.spots.parent_cell_id[:, n], self.spots.gene_id]

            term_2 = np.einsum('ij, ij -> i', cp, log_gamma_bar)

            term_3 = np.einsum('ij, ij -> i', cp, log_theta_bar)

            # wSpotCell[:, n] = term_1 + term_2 + logeta_bar + loglik[:, n]
            mvn_loglik = self.spots.mvn_loglik(self.spots.xyz_coords, sn, self.cells, self.config['is3D'])
            wSpotCell[:, n] = term_1 + term_2 + term_3 + logeta_bar + mvn_loglik
            mvn_loglik_arr[:, n] = mvn_loglik
            attention[:, n] = term_1
            expr_fluctuations[:, n] = term_2
            cell_inefficiency[:, n] = term_3
            gene_inefficiency[:, n] = logeta_bar

        # apply inside cell bonus
        bonus_mask = self.spots.bonus_mask * self.config['InsideCellBonus']
        wSpotCell += bonus_mask

        # update the prob a spot belongs to a neighboring cell
        self.spots.parent_cell_prob = softmax(wSpotCell, axis=1)
        self.spots.mvn_loglik_arr = mvn_loglik_arr
        self.spots.attention = attention
        self.spots.expr_fluctuations = expr_fluctuations
        self.spots.cell_inefficiency = cell_inefficiency
        self.spots.gene_inefficiency = gene_inefficiency

        # Since the spot-to-cell assignments changed you need to update the gene counts now.
        # However, this is commented out because it is computationally redundant;
        # the same operation is explicitly called as Step 1 at the top of the main_loop.
        # Note: If spots_to_cell ceases to be the final step of the loop, this MUST be uncommented.
        # self.geneCount_upd()

    # -------------------------------------------------------------------- #
    def spots_to_cell_par(self) -> None:
        """
        Updates spot-to-cell assignment probabilities.

        Implements equation (4) of the Qian paper. For each spot, calculates the
        probability of it belonging to each nearby cell or being a misread.

        Parallelized (multithreading) version of 'spots_to_cell'
        """
        nN = self.nN
        nS = self.nS

        wSpotCell = np.zeros([nS, nN], dtype=np.float64)
        gn = self.spots.data.gene_name.values
        expected_counts = self.single_cell.log_mean_expression.loc[gn].values

        # Pre-populate misread column
        misread = self.spots.misread_density(self.genes)
        wSpotCell[:, -1] = np.log(misread)

        log_gamma_bar = self.spots.log_gamma_bar.compute()

        def process_neighbor(n):
            sn = self.spots.parent_cell_id[:, n]
            cp = self.cells.classProb[sn]

            term_1 = oe.contract('ij, ij -> i', expected_counts, cp, optimize='optimal')

            current_log_gamma = log_gamma_bar[self.spots.parent_cell_id[:, n], self.spots.gene_id]
            term_2 = oe.contract('ij, ij -> i', cp, current_log_gamma, optimize='optimal')

            mvn_loglik = self.spots.mvn_loglik(self.spots.xyz_coords, sn, self.cells, self.config['is3D'])
            return n, term_1 + term_2 + mvn_loglik

        # Parallel processing
        results = joblib.Parallel(n_jobs=-1, backend='threading')(
            joblib.delayed(process_neighbor)(n) for n in range(nN - 1)
        )

        # Fill results back into wSpotCell
        for n, result in results:
            wSpotCell[:, n] = result

        # Apply inside cell bonus
        bonus_mask = self.spots.bonus_mask * self.config['InsideCellBonus']
        wSpotCell += bonus_mask

        # Update probabilities
        self.spots.parent_cell_prob = softmax(wSpotCell, axis=1)

        # Update gene counts
        # Commented out because it is computationally redundant;
        # the same operation is explicitly called as Step 1 at the top of the main_loop.
        # Note: If spots_to_cell_par ceases to be the final step of the loop, this MUST be uncommented.
        # self.geneCount_upd()

    # -------------------------------------------------------------------- #
    def rho_upd(self) -> None:
        """Updates gene-specific misread density (rho_g).

        Uses the expected number of background spots per gene to update
        the Gamma posterior for each gene's misread density.
        """
        background_counts = np.bincount(
            self.spots.gene_id,
            self.spots.parent_cell_prob[:, -1],
            minlength=self.nG
        )
        self.genes.calc_rho(background_counts)
        logger.info(f"rho_upd: bg_counts min/max={background_counts.min():.1f}/{background_counts.max():.1f}, "
                     f"rho_bar min/max={self.genes.rho_bar.min():.2e}/{self.genes.rho_bar.max():.2e}, "
                     f"log_rho min/max={self.genes.log_rho_bar.min():.4f}/{self.genes.log_rho_bar.max():.4f}")

    # -------------------------------------------------------------------- #
    def eta_upd(self) -> None:
        """
        Updates gene efficiency parameters (eta).

        Implements equation (5) of the Qian paper. Calculates the expected eta values
        by combining:
            1. Total gene counts across cells
            2. Cell type probabilities
            3. Mean expression values
            4. Area factors and gamma values

        Note:
            The zero-expressing cell class is excluded from the computation.
        """
        # grand_total = self.cells.background_counts.sum() + self.cells.total_counts.sum()
        # assert round(grand_total) == self.spots.data.shape[0], \
        #     'The sum of the background spots and the total gene counts should be equal to the number of spots'

        classProb = self.cells.classProb
        mu = self.single_cell.mean_expression_adj + self.config['SpotReg']
        area_factor = self.cells.ini_cell_props['area_factor']
        gamma_bar = self.spots.gamma_bar.compute()
        theta_bar = self.cells.theta_bar

        zero_prob = classProb[:, -1]  # probability a cell being a zero expressing cell
        zero_class_counts = self.spots.zero_class_counts(self.spots.gene_id, zero_prob)
        # zero_class_counts = oe.contract('c, cg -> g', classProb[:, -1], self.cells.geneCount, optimize='optimal')

        # Calcs the sum in the Gamma distribution (equation 5). The zero class
        # is excluded from the sum, hence the arrays in the einsum below stop at :-1
        # Note. We should exclude the "cell" that is meant to keep the
        # misreads, ie exclude the background, hence the relevant indexing below
        # starts at 1
        class_total_counts = oe.contract('ck, gk, c, cgk, ck -> g',
                                         classProb[:, :-1],
                                         mu.values[:, :-1],
                                         area_factor,
                                         gamma_bar[:, :, :-1],
                                         theta_bar[:,:-1], optimize='optimal')
        # background_counts = self.cells.background_counts
        background_counts = np.bincount(self.spots.gene_id, self.spots.parent_cell_prob[:, -1], minlength=self.nG)

        # observed (ie actual) gene reads per gene
        observed = self.config['rGene'] + self.spots.counts_per_gene - background_counts - zero_class_counts

        # expected (ie predicted) gene reads per gene
        expected = self.config['rGene'] + class_total_counts

        # Finally, update gene_gamma. It will basically divide observed by expected
        # and gene inefficiency will eventually express how well a gene is detected.
        self.genes.calc_eta(observed, expected)

    # -------------------------------------------------------------------- #
    def gaussian_upd(self) -> None:
        """
        Updates Gaussian distribution (centroids and covariance matrices) for cells
        """
        self.centroid_upd()
        self.cov_upd()

    # -------------------------------------------------------------------- #
    def centroid_upd(self) -> None:
        """
        Updates the centroid (mean) for each cell based on the posterior distribution.

        The posterior centroid is calculated as a weighted average of:
        - The prior centroid (mu_0), weighted by the prior pseudo-sample size (k_0)
        - The empirical (sample) mean (x_bar), weighted by the observed sample size (n)
        """

        # Get the prior weight (pseudo-sample size for the centroid)
        k_0 = self.config['cell_centroid_prior']['default']

        # Prior centroid (mu_0)
        prior_centroid = self.cells.ini_centroids()

        # 1. Calculate the empirical (sample) mean
        sample_mean = utils.empirical_mean(spots=self.spots, cells=self.cells)

        # 2. Get the observed sample size (gene counts per cell)
        sample_size = self.cells.total_counts

        # 3. Calculate the posterior centroid as a weighted average
        numerator = k_0 * prior_centroid + sample_size[:, None] * sample_mean
        denominator = k_0 + sample_size[:, None]
        posterior_centroid = numerator / denominator

        # 4. Handle the background (index 0)
        # Reset the background centroid to a default large value (e.g., max int)
        posterior_centroid.iloc[0, :] = -sys.maxsize

        # Update the centroids
        self.cells.centroid = posterior_centroid

    # -------------------------------------------------------------------- #
    def cov_upd(self) -> None:
        """
        Updates the covariance matrix for each cell based on the posterior distribution.

        The posterior covariance is computed using:
        - The scatter matrix (data-driven scale matrix)
        - The prior covariance (weighted by prior hyperparameters)
        - An adjustment term based on the difference between sample means and prior means
        """

        # Hyperparameters
        k_0 = self.config['cell_centroid_prior']['default']  # Prior for centroids
        nu_0 = self.config['cell_cov_prior']['default']  # Prior degrees of freedom for covariance

        # 1. Calculate the scatter matrix (data-driven scale matrix)
        scatter_matrix = self.cells.scatter_matrix(self.spots)

        # 2. Calculate the prior scale matrix
        prior_cov = self.cells.ini_cov()
        prior_scale_matrix = nu_0 * prior_cov

        # 3. Calculate the adjustment term
        # Difference between current centroids and prior centroids (x_bar - mu_0)
        mean_diff = self.cells.centroid - self.cells.ini_centroids()
        mean_outer_product = oe.contract('rk, rn -> rkn', mean_diff, mean_diff,
                                         optimize='optimal')  # (x_bar - mu_0)(x_bar - mu_0)^T

        # Multiplier for the adjustment term
        multiplier = (k_0 * self.cells.total_counts) / (k_0 + self.cells.total_counts)

        # Avoid warnings by setting background (index 0) adjustment to zero
        mean_outer_product[0] = np.zeros_like(mean_outer_product[0])
        adjustment_term = multiplier[:, None, None] * mean_outer_product

        # 4. Calculate the updated scale matrix
        scale_matrix_upd = scatter_matrix + prior_scale_matrix + adjustment_term

        # 5. Calculate the updated degrees of freedom
        nu_upd = nu_0 + self.cells.total_counts + 1

        # 6. Compute the expected covariance
        covariance_upd = utils.expected_covariance(scale_matrix_upd, nu_upd)

        # Handle background (index 0) by mapping it to the prior covariance
        covariance_upd[0] = prior_cov[0]

        # 7. Update the cell covariance attribute. Get the eigenvals/eigenvectors too
        self.cells.cov = covariance_upd
        self.cells.eig_vals, self.cells.eig_vecs = np.linalg.eigh(covariance_upd)

    # -------------------------------------------------------------------- #
    def mu_upd(self) -> None:
        """
        Updates mean expression values when single-cell reference is missing.

        Estimates mean expression for each gene and cell type using:
            1. Current cell type assignments
            2. Observed gene counts
            3. Cell area factors
            4. Current gamma and eta values

        Updates:
            - single_cell._mean_expression: Updated mean expression values
            - single_cell._log_mean_expression: Log of mean expression values
        """
        classProb = self.cells.classProb[1:, :-1].copy()
        geneCount = self.cells.geneCount[1:, :].copy()
        gamma_bar = self.spots.gamma_bar.compute()[1:, :, :-1]
        area_factor = self.cells.ini_cell_props['area_factor'][1:]

        numer = oe.contract('ck, cg -> gk', classProb, geneCount, optimize='optimal')
        denom = oe.contract('ck, c, cgk, g -> gk', classProb, area_factor, gamma_bar, self.genes.eta_bar,
                            optimize='optimal')

        me, lme = self.single_cell._gene_expressions(numer, denom)
        self.single_cell._mean_expression = me
        self.single_cell._log_mean_expression = lme

    # -------------------------------------------------------------------- #
    def dalpha_upd(self) -> None:
        """
        Updates cell type prior distribution parameters.

        Adjusts Dirichlet parameters based on:
            1. Current cell type assignments
            2. Initial alpha values
        """
        # logger.info('Update cell type (marginal) distribution')
        # Only update real classes (exclude Zero, which is the last column)
        zeta_real = self.cells.classProb[:, :-1].sum(axis=0)
        alpha_0 = np.ones(self.cellTypes.nK - 1, dtype=np.float32)
        self.cellTypes.alpha = zeta_real + alpha_0

    # -------------------------------------------------------------------- #
    def theta_upd(self):
        geneCounts = self.cells.geneCount.sum(axis=1)
        alpha = geneCounts + self.config['rTheta'] - 1

        mu = self.single_cell.mean_expression_adj + self.config['SpotReg']
        area_factor = self.cells.ini_cell_props['area_factor']
        gamma_bar = self.spots.gamma_bar.compute()
        eta_bar = self.genes.eta_bar

        beta = np.einsum('c, cgk, g, gk -> ck',
                         area_factor,
                         gamma_bar,
                         eta_bar,
                         mu) + self.config['rTheta']

        self.cells.calc_theta(alpha, beta)
        print('ok')

    # -------------------------------------------------------------------- #
    def detect_birth_candidates(self) -> np.ndarray:
        """Find background-dominated spots that cluster into dense regions.

        A spot is "background-dominated" if the misread column wins argmax
        across its nN candidates (no fixed threshold on bg prob). DBSCAN then
        filters out isolated spots: only spots inside a cluster of
        >= birth_min_blob_size survive.

        Returns a 1D array of qualifying spot indices. Each one will become
        a new cell centered at its own xyz coords. Overlap between newly
        birthed cells is expected and will be resolved by a later merge move.
        """
        pcp = self.spots.parent_cell_prob
        mask = np.argmax(pcp, axis=1) == (self.nN - 1)

        min_samples = self.config.get('birth_min_blob_size', 5)
        if mask.sum() < min_samples:
            return np.empty(0, dtype=np.int64)

        members_global = np.where(mask)[0]
        coords_full = self.spots.xyz_coords[mask]
        # 2D data has constant z, so drop it for clustering distance
        coords = coords_full if self.config['is3D'] else coords_full[:, :2]

        eps = self.config.get('birth_eps', float(self.cells.mcr))
        labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(coords)

        # Keep only spots that landed inside a cluster (drop noise label -1)
        in_cluster = labels >= 0
        qualifying = members_global[in_cluster]

        if len(qualifying) > 0:
            n_clusters = len(np.unique(labels[in_cluster]))
            logger.info(f"birth: {len(qualifying)} spots in {n_clusters} cluster(s) "
                        f"out of {mask.sum()} background-dominated spots")

        # ------------------------------------------------------------------
        # TEMP DEBUG GATE: replace the bg/DBSCAN-derived `qualifying` with
        # ALL spots inside a sphere of radius cell_radius around a fixed
        # centre. This widens the sample beyond bg-dominated Pvalb-only
        # spots so birthed cells start with multi-gene evidence and a
        # chance to push classProb away from the prior. Remove once the
        # real cluster-quality filtering is in place.
        # ------------------------------------------------------------------
        _centre = np.array([4550.0, 3240.0, 175.0], dtype=np.float32)
        _radius = float(self.cells.mcr)
        _spot_xyz = self.spots.xyz_coords
        if self.config['is3D']:
            _dist = np.linalg.norm(_spot_xyz - _centre, axis=1)
        else:
            _dist = np.linalg.norm(_spot_xyz[:, :2] - _centre[:2], axis=1)
        qualifying = np.where(_dist <= _radius)[0].astype(np.int64)
        n_genes = self.spots.data.gene_name.iloc[qualifying].nunique() if len(qualifying) else 0
        logger.info(
            f"birth: TEMP sphere filter (r={_radius:.1f} @ {_centre.tolist()}) "
            f"→ {len(qualifying)} spots across {n_genes} unique gene(s)"
        )
        return qualifying

    # -------------------------------------------------------------------- #
    def birth_cells(self, spot_indices: np.ndarray) -> None:
        """Birth one cell per qualifying spot, centered at the spot's xyz.

        Cells are 1-to-1 with spot_indices: new cell i is centered exactly
        at spot spot_indices[i] and inherits that spot's background prob mass.
        Overlapping cells (nearby spots in the same DBSCAN cluster) are
        expected, and a future merge move will collapse them.

        Steps:
          1. Extend Cells (one new cell per spot, centroid = spot xyz)
          2. Extend spots._gamma_bar with prior 1.0 (gamma_upd will refresh
             properly at step 5 of the next iteration)
          3. Rebuild parent_cell_id for the blob spots via fresh kNN against
             the extended cell set, and reset parent_cell_prob to a uniform
             1/nN over all slots, same convention as ini_cellProb at iter 0.
             A blob spot now sees its own birthed cell (closest, distance 0)
             plus the next nN-2 closest cells (typically other birthed cells
             from the same blob, then some real ones), letting them compete
             for each other's spots and accumulate enough counts to escape
             the "1-spot cell looks like Zero class" trap.
          4. Sync self.nC

        Non-blob spots are NOT touched.
        """
        spot_indices = np.asarray(spot_indices)
        if spot_indices.size == 0:
            return

        B = spot_indices.size
        # one cell per spot, centered at that spot's own coords
        centroids = self.spots.xyz_coords[spot_indices].astype(np.float32)

        # 1. Extend Cells
        new_ids = self.cells.add_cells(centroids, self.cellTypes.prior)

        # 2. Extend spots._gamma_bar with prior expectation E[gamma]=1.
        # Only read once at step 3 (eta_upd) of the next iter; gamma_upd at
        # step 5 then rebuilds it with proper posterior values.
        gb = self.spots.gamma_bar.compute()
        new_gb = np.ones((B, self.nG, self.nK), dtype=gb.dtype)
        gb_ext = np.concatenate([gb, new_gb], axis=0)
        self.spots._gamma_bar = delayed(gb_ext)

        # 3. Rebuild parent_cell_id for the blob spots via kNN against the
        # extended cell set. Use a BINARY inside-cell rule for parent_cell_prob:
        # for each blob spot, set prob=1.0 for every cell whose centroid is
        # within cell_radius of the spot (i.e. the spot is "inside" that cell).
        # If the spot lies inside no cell, send all its mass to background.
        # This intentionally double-counts: a spot inside K overlapping cells
        # contributes prob=1.0 to each of them, so per-cell counts scale up
        # to ~(blob size) instead of ~1, giving cells enough mass to push
        # classProb away from the uniform/Zero prior. The over-counting is
        # one-shot: spots_to_cell at the end of the next iteration restores
        # a proper softmax. The merge move (TBD) will collapse the overlapping
        # cells once they've established a class.
        nbrs = self.cells.nn()
        spot_zyx = self.spots.data[['z', 'y', 'x']].values[spot_indices]
        distances, neighbor_ids = nbrs.kneighbors(spot_zyx)
        neighbor_ids[:, -1] = 0  # last slot reserved for background

        radius = float(self.cells.mcr)
        inside = distances < radius           # (B, nN) bool
        inside[:, -1] = False                 # bg slot is never "inside"

        new_prob = np.zeros_like(distances, dtype=np.float32)
        new_prob[inside] = 1.0
        # spots inside no cell → fully background
        no_cell = ~inside.any(axis=1)
        new_prob[no_cell, -1] = 1.0

        self.spots.parent_cell_id[spot_indices] = neighbor_ids.astype(
            self.spots.parent_cell_id.dtype
        )
        self.spots.parent_cell_prob[spot_indices] = new_prob

        n_in = int(inside.sum())
        n_per_spot = inside.sum(axis=1)
        logger.info(
            f"birth: binary rule: {n_in} (cell, spot) inside-pairs over "
            f"{B} spots (avg {n_in / max(B, 1):.1f} cells/spot, "
            f"max {int(n_per_spot.max()) if B else 0}, "
            f"{int(no_cell.sum())} spot(s) inside nothing → bg)"
        )

        # 4. Sync VarBayes nC cache
        self.nC = self.cells.nC

        logger.info(f"birth: instantiated {B} cell(s); nC={self.nC}")

        # Capture the first birth batch for debug tracking
        if self._birth_debug_spots is None:
            self._birth_debug_spots = np.asarray(spot_indices).copy()
            self._birth_debug_cells = new_ids.copy()
            self._birth_debug_iter = self.iter_num
            gene_names = self.spots.data.gene_name.iloc[self._birth_debug_spots].values
            xyz = self.spots.xyz_coords[self._birth_debug_spots]
            logger.info("=" * 80)
            logger.info(f"BIRTH-DEBUG: tracking {B} spot/cell pair(s) born at iter {self.iter_num}")
            for i, (s, c, g, p) in enumerate(zip(self._birth_debug_spots,
                                                 self._birth_debug_cells,
                                                 gene_names, xyz)):
                logger.info(f"  [{i}] spot={s} gene={g:<10} "
                            f"xyz=({p[0]:.1f},{p[1]:.1f},{p[2]:.1f}) → cell={c}")
            logger.info("=" * 80)

    # -------------------------------------------------------------------- #
    def _birth_debug_dump(self) -> None:
        """Per-iteration trace of the tracked birthed cells and source spots.

        Logs, for each tracked cell:
          - total geneCount, count of nonzero genes, top-3 contributing genes
          - top class (name + prob), and its closest competitor

        And for each tracked spot:
          - bg prob, prob assigned to its OWN birthed cell, slot index
          - top non-self neighbor + prob (in case the spot has migrated away)
        """
        if self._birth_debug_spots is None:
            return

        spots = self._birth_debug_spots
        cells = self._birth_debug_cells
        iters_since = self.iter_num - self._birth_debug_iter

        # ----- per-cell view -----
        gc = self.cells.geneCount[cells]                 # (B, nG)
        cp = self.cells.classProb[cells]                 # (B, nK)
        gene_panel = np.asarray(self.genes.gene_panel)
        class_names = np.asarray(self.cells.class_names)

        logger.info(f"BIRTH-DEBUG iter={self.iter_num} (+{iters_since}): "
                    f"tracked {len(cells)} cells")
        for i, (s, c) in enumerate(zip(spots, cells)):
            total = float(gc[i].sum())
            nz = int((gc[i] > 0).sum())
            # top-3 genes (counts, then names)
            top_g = np.argsort(gc[i])[::-1][:3]
            top_g = [(gene_panel[g], float(gc[i, g])) for g in top_g if gc[i, g] > 0]
            # top-2 classes
            top_k = np.argsort(cp[i])[::-1][:2]
            top_k = [(class_names[k], float(cp[i, k])) for k in top_k]

            # spot-side: where is this spot now?
            pcid_row = self.spots.parent_cell_id[s]
            pcp_row = self.spots.parent_cell_prob[s]
            bg_prob = float(pcp_row[-1])
            self_slot = np.where(pcid_row == c)[0]
            self_prob = float(pcp_row[self_slot[0]]) if self_slot.size else float('nan')
            # best non-bg, non-self neighbor
            mask_other = (pcid_row != c) & (np.arange(self.nN) != self.nN - 1)
            if mask_other.any():
                other_idx = np.argmax(np.where(mask_other, pcp_row, -np.inf))
                other_cell = int(pcid_row[other_idx])
                other_prob = float(pcp_row[other_idx])
            else:
                other_cell, other_prob = -1, float('nan')

            top_g_str = ", ".join(f"{n}={v:.2f}" for n, v in top_g) or "(none)"
            top_k_str = ", ".join(f"{n}={v:.3f}" for n, v in top_k)
            logger.info(
                f"  [{i}] cell={c}: counts={total:.2f} "
                f"(nz={nz}) top_genes=[{top_g_str}] top_classes=[{top_k_str}]"
            )
            logger.info(
                f"      spot={s}: bg={bg_prob:.3f} "
                f"self={self_prob:.3f} (slot={self_slot.tolist()}) "
                f"other_top=cell={other_cell} p={other_prob:.3f}"
            )

    # -------------------------------------------------------------------- #
    def diagnostics_upd(self) -> None:
        """Update diagnostic visualization if controller is available."""
        if self.diagnostic_controller is None:
            return

        try:
            self.diagnostic_controller.update_diagnostics(
                algorithm_model=self,
                iteration=self.iter_num,
                has_converged=self.has_converged
            )
        except Exception as e:
            logger.warning(f"Failed to update diagnostics: {e}")


    # -------------------------------------------------------------------- #
    def heatmap_counts_per_class(self):
        """Display the interactive heatmap."""
        return visualisation.heatmap_counts_per_class(self)

    def calculate_genes_log_likelihood_contr(self, label):
        return utils.calculate_genes_log_likelihood_contr(self, label)

    # def plot_loglik_contr(self, df):
    #     return utils.plot_loglik_contr(df)

    # def visualize_fit(self, gene_counts, scaled_means):
    #     return utils.visualize_fit(gene_counts, scaled_means)

    def check_cell(self, my_label, user_class, top_n=10, show_plot=True):
        return utils.check_cell(self, my_label, user_class, top_n, show_plot)

    def check_spot(self, spot_id):
        return visualisation.check_spot(self, spot_id)

    def read_tsv(self, filepath):
        return utils.read_tsv(filepath)

    def cell_typing_breakdown(self, label, weights=None, show_plot=True):
        return utils.cell_typing_breakdown(self, label, weights, show_plot)

    # def trellis_plot(self, label, flatfile_folder):
    #     return visualisation.trellis_plot(self, label, flatfile_folder)
