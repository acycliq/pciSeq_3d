"""Take one cell or one spot apart and see why it got called the way it did."""

import logging
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt

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
