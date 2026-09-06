"""Take one cell or one spot apart and see why it got called the way it did."""

import logging
import numpy as np
import pandas as pd
from scipy.special import softmax
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.special import psi

from .likelihood import calculate_genes_log_likelihood_contr
from .visualisation import spot_to_cell_prob_plot, spot_to_cell_score_plot

logger = logging.getLogger(__name__)


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

    # Merge with gene_counts and the NB prediction (the expected counts the model
    # actually uses in the likelihood). scaled_means_df on its own is area_factor * mu;
    # the NB mean that drives the log-likelihood is scaled_means * eta_g * theta_ck + SpotReg,
    # so we have to put eta (gene efficiency) and theta (this cell's efficiency) back in.
    # This matches compute_gene_loglikelihood_matrix and the JS viewer's diagnostics.js.
    nb_values = np.einsum('gk, g, k -> gk',
                          scaled_means_df.values,
                          obj.genes.eta_bar,
                          obj.cells.theta_bar[pciSeq_label]) + obj.config['SpotReg']
    nb_prediction = pd.DataFrame(nb_values, index=scaled_means_df.index, columns=scaled_means_df.columns)
    expected_counts = nb_prediction.loc[selected_genes, [pciSeq_class, user_class]]
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
    mrf = obj.cells.calc_mrf()

    gene_loglik_pciSeq = my_contr_df[pciSeq_class].sum()
    gene_loglik_user = my_contr_df[user_class].sum()
    log_prior_pciSeq = log_prior[pciSeq_idx]
    log_prior_user = log_prior[user_idx]
    mrf_pciSeq = mrf[pciSeq_label, pciSeq_idx]
    mrf_user = mrf[pciSeq_label, user_idx]

    # Log-posterior for the two classes
    log_post_pciSeq = gene_loglik_pciSeq + log_prior_pciSeq + mrf_pciSeq
    log_post_user = gene_loglik_user + log_prior_user + mrf_user

    # Posterior probabilities (softmax over just these two classes)
    log_posts = np.array([log_post_pciSeq, log_post_user])
    posterior_probs = softmax(log_posts)

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
                       [posterior_probs[0] * 100, posterior_probs[1] * 100],
                       color=['skyblue', 'lightcoral'])
        axes[1, 1].set_ylabel('Posterior Probability (%)')
        axes[1, 1].set_title(f'Cell: {label} - Posterior probabilities')

        plt.tight_layout()
        plt.show()

    return gene_expression_data, my_contr_df, (fig if show_plot else None)


def check_spot(self, spot_id):
    """
    Analyze a spot by creating visualization charts and returning score/probability arrays.

    Parameters:
    spot_id (int): The ID of the spot to analyze

    Returns:
    tuple: (scores_array, probabilities_array)s
    """
    # Get data for the specified spot
    # First find the row position of the spot_id
    row_pos = self.spots.data.index.get_loc(spot_id)

    gene_name = self.spots.data.iloc[row_pos].gene_name # I could have used loc[spot_id] here too
    x = self.spots.data.iloc[row_pos].x.astype(np.int32).tolist()
    y = self.spots.data.iloc[row_pos].y.astype(np.int32).tolist()
    z = self.spots.data.iloc[row_pos].z.astype(np.int32).tolist()
    n_cells = len(self.spots.parent_cell_id[row_pos]) - 1  # Exclude background
    cell_ids = self.spots.parent_cell_id[row_pos][:-1]
    mvn_loglik = self.spots.mvn_loglik_arr[row_pos][:-1]
    attention = self.spots.attention[row_pos][:-1]
    expr_fluct = self.spots.expr_fluctuations[row_pos][:-1]
    cell_inefficiency = self.spots.cell_inefficiency[row_pos][:-1]
    gene_inefficiency = self.spots.gene_inefficiency[row_pos][:-1]
    gene_idx = np.where(self.genes.gene_panel == gene_name)[0][0]
    misread = self.genes.log_rho_bar[gene_idx]
    # the inside-cell bonus the model adds before the softmax in spots_to_cell. it is
    # nonzero only for the cell whose boundary the spot sits in, and zero for background.
    bonus = self.spots.bonus_mask[row_pos][:-1] * self.config['InsideCellBonus']

    # Calculate scores and probabilities
    scores = mvn_loglik + attention + expr_fluct + cell_inefficiency + gene_inefficiency + bonus
    scores = np.append(scores, misread)
    probabilities = softmax(scores)

    # Create labels. If the segmentation has been relabelled, map the labels back to the original ones.
    if self.config['label_map']:
        reverse_map = {v:k for k, v in self.config['label_map'].items()}
        cell_ids = [reverse_map[d] for d in cell_ids]

    labels = [f'Cell {cid}' for cid in cell_ids] + ['Misread']

    datadict = {
        'spot_id': spot_id,
        'gene_name': gene_name,
        'x': x,  # Already converted to list of int32
        'y': y,  # (same as above)
        'z': z,  # (same as above)
        'n_cells': n_cells,
        'cell_ids': cell_ids,
        'mvn_loglik': mvn_loglik,
        'attention': attention,
        'expr_fluct': expr_fluct,
        'cell_inefficiency': cell_inefficiency,
        'gene_inefficiency': gene_inefficiency,
        'bonus': bonus,
        'misread': float(misread),  # Convert numpy float to native Python float
        'score': scores,
        'prob': probabilities,
        'labels': labels
    }

    df = pd.DataFrame({
        'Name': labels[:-1],
        # 'internal_tag':self.spots.parent_cell_id[row_pos][:-1],
        'mvn_loglik': mvn_loglik,
        'attention': attention,
        'expr_fluct': expr_fluct,
        'cell_inefficiency': cell_inefficiency,
        'gene_inefficiency': gene_inefficiency,
        'bonus': bonus}).set_index(['Name'])
    df['misread'] = np.nan
    df['sum'] = df[['mvn_loglik', 'attention', 'expr_fluct', 'cell_inefficiency', 'gene_inefficiency', 'bonus']].sum(axis=1)
    df.loc['background'] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, misread, misread]

    spot_to_cell_score_plot(datadict)
    spot_to_cell_prob_plot(datadict)
    return df


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
    if prior_mode == 'weighted':
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
