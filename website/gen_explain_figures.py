#!/usr/bin/env python3
"""
Makes the figures and tables for the "Explaining the calls" pages.

Loads two fitted Espio models, the same data and settings, one with the spatial prior
(mrf_beta=1.5) and one without (mrf_beta=0.0), and calls check_cell and check_spot on a
few hand picked cells and spots. The check_cell figures and the check_spot charts go to
docs/public/explaining-the-calls/ (the charts need kaleido for the plotly export), the
tables go to docs/explaining-the-calls/_tables/, where the pages pull them in with an
@include. The numbers the page text quotes are dumped to explain_numbers.json next to
this script, so the prose can be checked against them.

Run it from anywhere:  python website/gen_explain_figures.py
Rerun it whenever check_cell, check_spot or the model changes, otherwise the pages go
stale. The prose quotes numbers too, so reread it after (compare with the json).
"""

import gc
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # no window, we only save the figures
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
# use the pciSeq in this repo, not whatever copy is installed in site-packages
sys.path.insert(0, str(REPO))
import pciSeq  # noqa: E402,F401

FIG_DIR = HERE / "docs" / "public" / "explaining-the-calls"
TABLE_DIR = HERE / "docs" / "explaining-the-calls" / "_tables"
NUMBERS = HERE / "explain_numbers.json"

# the two fitted models, made by pciSeq_experiments/run_espio.py with only mrf_beta changed
RUNS = {
    "mrf": pathlib.Path.home() / "pciseq_runs/zero_boost/espio/pciSeq/data/debug/pciSeq.pickle",
    "nomrf": pathlib.Path.home() / "pciseq_runs/zero_boost/espio_noMRF/pciSeq/data/debug/pciSeq.pickle",
}

DG = "037 DG Glut"
L6CT = "030 L6 CT CTX Glut"
CA1 = "016 CA1-ProS Glut"

# (run, cell label, class to compare against, file name)
CELLS = [
    ("mrf", 7768, L6CT, "cell-7768-mrf"),      # the walkthrough: DG with the spatial prior
    ("nomrf", 7768, DG, "cell-7768-nomrf"),    # the same cell called L6 CT without it
]
# (run, spot id, file name): spots of cell 7768 that change hands between the runs
SPOTS = [
    ("nomrf", 2452812, "spot-2452812-nomrf"),  # Rprm, border with DG neighbours
    ("mrf", 2452812, "spot-2452812-mrf"),
    ("nomrf", 2665680, "spot-2665680-nomrf"),  # Glul, goes to background with the mrf
    ("mrf", 2665680, "spot-2665680-mrf"),
]

# the check_cell figure is 14 inches wide and gets shrunk to fit the page column, so
# bump the font sizes for the docs copy only
DOC_FONTS = {"font.size": 17, "axes.titlesize": 17, "axes.labelsize": 16,
             "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15}


def to_markdown(df, floatfmt="{:.3f}"):
    """Tiny markdown table writer, so we dont need tabulate just for this."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def cell_table_html(ged, label):
    """The check_cell table as html, with the same two level header and column order as the
    DataFrame check_cell returns. Markdown tables cant do a two level header."""
    top = [a for a, _ in ged.columns]
    sub = [b for _, b in ged.columns]
    # merge neighbouring columns that share the same top header into one spanning cell
    groups = []
    for t in top:
        if groups and groups[-1][0] == t:
            groups[-1][1] += 1
        else:
            groups.append([t, 1])
    head_top = "".join(f'<th colspan="{n}">{t}</th>' if n > 1 else f"<th>{t}</th>" for t, n in groups)
    head_sub = "".join(f"<th>{b}</th>" for b in sub)
    lines = ["<table>", "<thead>",
             f"<tr><th></th>{head_top}</tr>",
             f"<tr><th>gene</th>{head_sub}</tr>",
             "</thead>", "<tbody>"]
    for gene, row in zip(ged.index, ged.values):
        cells = "".join(f"<td>{v:.2f}</td>" for v in row)
        lines.append(f"<tr><td>{gene}</td>{cells}</tr>")
    lines += ["</tbody>", "</table>"]
    return "\n".join(lines) + "\n"


def cell_numbers(obj, label, user_class, contr, ged):
    """The numbers the page quotes for one check_cell call."""
    lm = obj.config["label_map"]
    row = lm[label] if lm else label
    names = list(obj.cells.class_names)
    assigned = names[int(obj.cells.classProb[row].argmax())]
    a, u = names.index(assigned), names.index(user_class)
    p = obj.cells.classProb[row]
    top = np.argsort(p)[::-1][:5]
    diff = contr["diff"]
    nbr_rows = obj.cells.nbrs["indices"][row]
    return {
        "assigned": assigned, "user_class": user_class, "mrf_beta": float(obj.config["mrf_beta"]),
        "n_spots": float(obj.cells.geneCount[row].sum()),
        "p_assigned": float(p[a]), "p_user": float(p[u]),
        "top5": [[names[k], float(p[k])] for k in top],
        "gene_ll": [float(contr[assigned].sum()), float(contr[user_class].sum())],
        "log_prior": [float(obj.cellTypes.log_prior[a]), float(obj.cellTypes.log_prior[u])],
        "mrf": [float(obj.cells.mrf[row, a]), float(obj.cells.mrf[row, u])],
        "top_genes_assigned": [[g, float(v)] for g, v in diff[diff > 0].nlargest(10).items()],
        "top_genes_user": [[g, float(v)] for g, v in diff[diff < 0].nsmallest(10).items()],
        "table": {g: [float(x) for x in r] for g, r in zip(ged.index, ged.values)},
        "rSpot": float(obj.config["rSpot"]),
        "neighbour_types": [names[int(obj.cells.classProb[n].argmax())] for n in nbr_rows],
    }


def gene_breakdown(obj, label, gene, type_names):
    """For one gene: how the model prediction for this cell is built, and, per type, the
    observed mean in cells of that type against the mean model prediction for them. The
    page uses it to show why the prediction and the observed mean differ."""
    cfg = obj.config
    lm = cfg["label_map"]
    row = lm[label] if lm else label
    names = list(obj.cells.class_names)
    g = list(obj.genes.gene_panel).index(gene)
    eta = float(obj.genes.eta_bar[g])
    out = {"gene": gene, "eta": eta, "Inefficiency": float(cfg["Inefficiency"]),
           "SpotReg": float(cfg["SpotReg"]), "types": {}}
    counts = obj.cells.geneCount[:, g]
    for t in type_names:
        k = names.index(t)
        p = obj.cells.classProb[:, k]
        pred = obj.scaled_exp[:, g, k] * eta * obj.cells.theta_bar[:, k] + cfg["SpotReg"]
        out["types"][t] = {
            "reference_mean": float(obj.single_cell.mean_expression.values[g, k]),
            "theta_this_cell": float(obj.cells.theta_bar[row, k]),
            "area_factor_this_cell": float(obj.cells.ini_cell_props["area_factor"][row]),
            "prediction_this_cell": float(pred[row]),
            # gamma as the model estimates it for this cell and gene (main.py gamma_upd):
            # (rSpot + observed) / (rSpot + prediction without SpotReg)
            "prediction_no_spotreg": float(pred[row] - cfg["SpotReg"]),
            "gamma_this_cell": float((cfg["rSpot"] + counts[row]) / (cfg["rSpot"] + pred[row] - cfg["SpotReg"])),
            "observed_mean": float((p * counts).sum() / p.sum()),
            "mean_prediction": float((p * pred).sum() / p.sum()),
            "mean_theta": float((p * obj.cells.theta_bar[:, k]).sum() / p.sum()),
        }
    out["this_cell_total"] = float(obj.cells.geneCount[row].sum())
    out["this_cell_observed"] = float(counts[row])
    out["rSpot"] = float(cfg["rSpot"])
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    numbers = {"cells": {}, "spots": {}}

    # check_spot draws its two plotly charts with fig.show() and does not hand the
    # figures back. So swap show() for something that just keeps them, then save them.
    shown = []
    go.Figure.show = lambda self, *args, **kwargs: shown.append(self)

    for run, path in RUNS.items():
        # one 5 GB model at a time
        obj = pd.read_pickle(path)

        for _, label, user_class, name in [c for c in CELLS if c[0] == run]:
            with matplotlib.rc_context(DOC_FONTS):
                ged, contr, fig = obj.check_cell(label, user_class)
                fig.savefig(FIG_DIR / f"{name}.png", dpi=90)
            plt.close(fig)
            numbers["cells"][name] = cell_numbers(obj, label, user_class, contr, ged)
            if name == "cell-7768-mrf":
                numbers["sema5a_breakdown"] = gene_breakdown(
                    obj, label, "Sema5a", [DG, L6CT, "017 CA3 Glut", CA1])
            (TABLE_DIR / f"{name}.md").write_text(cell_table_html(ged, label))
            print(f"wrote {name}.png and {name}.md")

        for _, spot_id, name in [s for s in SPOTS if s[0] == run]:
            shown.clear()
            df = obj.check_spot(spot_id, show_plot=True)
            score_fig, prob_fig = shown  # same order as check_spot draws them
            score_fig.write_image(FIG_DIR / f"{name}-scores.png", width=1000, height=550, scale=1)
            prob_fig.write_image(FIG_DIR / f"{name}-probs.png", width=1000, height=450, scale=1)
            (TABLE_DIR / f"{name}.md").write_text(to_markdown(df))
            numbers["spots"][name] = {"gene": str(obj.spots.data.loc[spot_id].gene_name),
                                      "table": json.loads(df.to_json(orient="index"))}
            print(f"wrote {name}-scores.png, {name}-probs.png and {name}.md")

        del obj
        gc.collect()

    NUMBERS.write_text(json.dumps(numbers, indent=1))
    print(f"wrote {NUMBERS}")


if __name__ == "__main__":
    main()
