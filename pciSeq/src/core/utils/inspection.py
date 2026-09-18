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


def check_cell(obj, label, user_class, top_n=10, show_plot=True, top_classes=5):
    """Implementation of VarBayes.check_cell. See that method for the full description."""

    # If original labels have been renumbered find the label it's been mapped to.
    if obj.config['label_map']:
        pciSeq_label = obj.config['label_map'][label]
    else:
        pciSeq_label = label

    # Step 1: per-gene log-likelihood contributions for this cell. Read the ones the model
    # kept from its last class update (cells.nb_contr), so the numbers match classProb and
    # we dont rebuild the whole nC x nG x nK matrix just to look at one row.
    gene_panel = obj.genes.gene_panel
    class_names = obj.cells.class_names
    contr_df = pd.DataFrame(obj.cells.nb_contr[pciSeq_label], columns=class_names, index=gene_panel)
    gene_counts = pd.Series(obj.cells.geneCount[pciSeq_label], index=gene_panel)
    scaled_means_df = pd.DataFrame(obj.scaled_exp[pciSeq_label], columns=class_names, index=gene_panel)

    # Step 2: Get the cell's class from cellData
    pciSeq_class = obj.cells.class_names[obj.cells.classProb[pciSeq_label].argmax()]

    # Step 3: Check if classes exist in contr_df
    if pciSeq_class not in contr_df.columns or user_class not in contr_df.columns:
        raise ValueError(f"One or both classes ({pciSeq_class}, {user_class}) not found in contr_df.")

    if user_class == pciSeq_class:
        raise ValueError(f"Cell {label} is already typed as {user_class}, pick a different "
                         f"class to compare it against.")

    # Step 4: Calculate differences and get top/bottom genes
    my_contr_df = contr_df[[pciSeq_class, user_class]].copy()
    my_contr_df['diff'] = my_contr_df[pciSeq_class] - my_contr_df[user_class]

    # only genes that actually favour a class. nlargest/nsmallest on their own always
    # return top_n genes, so on a near empty cell the plots filled up with genes whose
    # diff is 0. Now a side can have fewer than top_n genes, or none at all.
    diff = my_contr_df['diff']
    top_genes = diff[diff > 0].nlargest(top_n).index.values
    bottom_genes = diff[diff < 0].nsmallest(top_n).index.values

    # Step 5: Combine top and bottom genes. On a small panel the two lists can overlap,
    # so drop repeats, otherwise the merges below duplicate rows.
    selected_genes = pd.unique(np.append(top_genes, bottom_genes))

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
        # observed data first, then what the model predicts for this cell. The viewer
        # server reads these columns by position, so keep the order if you rename them
        (f'Cells typed as {pciSeq_class}', 'observed mean'),
        (f'Cells typed as {user_class}', 'observed mean'),
        (f'Model prediction for cell {label}', f'as {pciSeq_class}'),
        (f'Model prediction for cell {label}', f'as {user_class}'),
        (f'Cell {label}', 'observed')
    ])
    gene_expression_data.columns = new_columns

    # Step 7: Compute the prior and MRF terms for the two classes
    class_names = list(obj.cells.class_names)
    pciSeq_idx = class_names.index(pciSeq_class)
    user_idx = class_names.index(user_class)

    log_prior = obj.cellTypes.log_prior
    # the mrf the model actually used in its last class update
    mrf = obj.cells.mrf

    gene_loglik_pciSeq = my_contr_df[pciSeq_class].sum()
    gene_loglik_user = my_contr_df[user_class].sum()
    log_prior_pciSeq = log_prior[pciSeq_idx]
    log_prior_user = log_prior[user_idx]
    mrf_pciSeq = mrf[pciSeq_label, pciSeq_idx]
    mrf_user = mrf[pciSeq_label, user_idx]

    fig = None
    if show_plot:
        # constrained layout keeps the panels in a row the same size and lined up, even
        # when one has much longer tick labels than the other
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), layout='constrained')

        # --- Top row: gene-level log-likelihood differences ---
        for ax, genes, cls, color in [(axes[0, 0], top_genes, pciSeq_class, 'skyblue'),
                                      (axes[0, 1], bottom_genes, user_class, 'lightcoral')]:
            vals = my_contr_df.loc[genes, 'diff']
            # two lines, one line is wider than the panel once the class names get long
            ax.set_title(f'Cell: {label} - Top {len(genes)} contr for class:\n{cls} (Sum: {vals.sum():.2f})')
            if len(genes):
                vals.plot.bar(ax=ax, color=color)
            else:
                # pandas cant bar plot an empty series, so just say it
                ax.text(0.5, 0.5, f'No gene favours {cls}', transform=ax.transAxes,
                        ha='center', va='center', color='grey')
            ax.set_ylabel('Log-Likelihood Difference')
            ax.set_xlabel('Genes')

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
        axes[1, 0].set_title(f'Cell: {label} - Log-posterior components\n(higher is better)')
        axes[1, 0].legend()
        axes[1, 0].axhline(y=0, color='grey', linestyle='--', linewidth=0.5)

        # --- Bottom-right: the model posterior over all classes, top ones only ---
        # show the top_classes most likely classes, plus user_class if it didnt make
        # the cut, so you can always see both classes being compared.
        probs = obj.cells.classProb[pciSeq_label] * 100
        shown = list(np.argsort(probs)[::-1][:top_classes])
        if user_idx not in shown:
            shown.append(user_idx)
        n_hidden = len(class_names) - len(shown)
        # clipped at 0, otherwise float rounding can print -0.0%
        hidden_sum = max(0.0, probs.sum() - probs[shown].sum())

        bar_names = [class_names[i] for i in shown]
        bar_probs = [probs[i] for i in shown]
        colors = ['skyblue' if n == pciSeq_class else 'lightcoral' if n == user_class else 'lightgrey'
                  for n in bar_names]
        xpos = np.arange(len(shown))
        axes[1, 1].bar(xpos, bar_probs, color=colors)
        for xi, p in zip(xpos, bar_probs):
            axes[1, 1].text(xi, p, f'{p:.1f}', ha='center', va='bottom', fontsize='small')
        # 45 degrees, anchored at the right end so each name finishes under its bar
        axes[1, 1].set_xticks(xpos)
        axes[1, 1].set_xticklabels(bar_names, rotation=45, ha='right', rotation_mode='anchor')
        axes[1, 1].set_ylabel('Posterior probability (%)')
        # a bit of headroom above 100 so a 100% bar and its label dont touch the top,
        # the ticks still stop at 100
        axes[1, 1].set_ylim(0, 110)
        axes[1, 1].set_yticks(range(0, 101, 20))
        # extra pad so a 100% bar label does not run into the title
        axes[1, 1].set_title(f'Cell: {label} - Posterior over all {len(class_names)} classes', pad=14)
        axes[1, 1].text(0.98, 0.97, f'{n_hidden} classes not shown, summing to {hidden_sum:.1f}%',
                        transform=axes[1, 1].transAxes, ha='right', va='top', fontsize='small', color='grey')

        plt.show()

    return gene_expression_data, my_contr_df, (fig if show_plot else None)


def check_spot(self, spot_id, show_plot=True):
    """Implementation of VarBayes.check_spot. See that method for the full description."""
    # Get data for the specified spot
    # First find the row position of the spot_id
    row_pos = self.spots.data.index.get_loc(spot_id)

    gene_name = self.spots.data.iloc[row_pos].gene_name # I could have used loc[spot_id] here too
    x = round(float(self.spots.data.iloc[row_pos].x), 2)
    y = round(float(self.spots.data.iloc[row_pos].y), 2)
    z = round(float(self.spots.data.iloc[row_pos].z), 2)
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
        'x': x,
        'y': y,
        'z': z,
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
    # softmax of the sums, same order as the rows (cells first, background last)
    df['prob'] = probabilities

    if show_plot:
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
