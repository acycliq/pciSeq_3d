#!/usr/bin/env python3
"""
Makes the figures and tables for the "Explaining the calls" pages.

Runs pciSeq on the CA1 demo data that ships with the package, then calls check_cell
and check_spot on a few hand picked cells and spots. The check_cell figures go to
docs/public/explaining-the-calls/, the check_spot charts go there too (needs kaleido for
the plotly export), and the tables go to docs/explaining-the-calls/_tables/, where the
pages pull them in with an @include.

Run it from anywhere:  python website/gen_explain_figures.py
Rerun it whenever check_cell, check_spot or the model changes, otherwise the pages go
stale. The prose on the pages quotes a few of the numbers too, so reread it after.
"""

import pathlib
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")  # no window, we only save the figures

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.sparse import load_npz

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
# use the pciSeq in this repo, not whatever copy is installed in site-packages
sys.path.insert(0, str(REPO))
import pciSeq  # noqa: E402

DATA = REPO / "pciSeq" / "data" / "mouse" / "ca1"
FIG_DIR = HERE / "docs" / "public" / "explaining-the-calls"
TABLE_DIR = HERE / "docs" / "explaining-the-calls" / "_tables"

# the examples on the pages. Picked from the CA1 run so that each one shows a different
# reason for a call. If a rerun changes the calls, pick new ones and fix the prose.
CELLS = [
    # (cell label, class to compare against, file name)
    (1023, "Sst.Erbb4.Rgs10", "cell-1023"),  # clear call, the genes decide
    (430, "PC.Other2", "cell-430"),          # the neighbours decide
    (2979, "Zero", "cell-2979"),             # near empty cell, Zero competes
]
SPOTS = [
    (1, "spot-1"),    # goes to a cell
    (55, "spot-55"),  # goes to the background
    (70, "spot-70"),  # split between two cells
]


def load_ca1():
    coo = load_npz(DATA / "segmentation" / "label_image.coo.npz")
    spots = pd.read_csv(DATA / "iss" / "spots.csv")
    sc = pd.read_csv(DATA / "scRNA" / "scRNAseq.csv.gz", header=None, index_col=0,
                     compression="gzip", dtype=object)
    # first row holds the class of each single cell, use it as the column names
    sc = sc.rename(columns=sc.iloc[0]).iloc[1:].astype(float).astype(np.uint32)
    return spots, coo, sc


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


def cell_table_html(ged, assigned, user_class):
    """The check_cell table with a grouped header: each class name once on top, spanning
    its mean and expected columns. The real headers repeat the full class names in every
    column and the table gets too wide for the page. Markdown tables cant span columns,
    so this one is plain html."""
    # check_cell order is mean A, mean B, expected A, expected B, observed.
    # regroup it per class: mean A, expected A, mean B, expected B, observed
    vals = ged.values[:, [0, 2, 1, 3, 4]]
    lines = [
        "<table>",
        "<thead>",
        f'<tr><th></th><th colspan="2"><code>{assigned}</code></th>'
        f'<th colspan="2"><code>{user_class}</code></th><th>this cell</th></tr>',
        "<tr><th>gene</th><th>mean</th><th>expected</th><th>mean</th><th>expected</th><th>observed</th></tr>",
        "</thead>",
        "<tbody>",
    ]
    for gene, row in zip(ged.index, vals):
        cells = "".join(f"<td>{v:.2f}</td>" for v in row)
        lines.append(f"<tr><td>{gene}</td>{cells}</tr>")
    lines += ["</tbody>", "</table>"]
    return "\n".join(lines) + "\n"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    spots, coo, sc = load_ca1()
    with tempfile.TemporaryDirectory() as tmp:
        pciSeq.fit(spots=spots, coo=coo, scRNAseq=sc, opts={"output_path": tmp})
        obj = pd.read_pickle(pathlib.Path(tmp) / "pciSeq" / "data" / "debug" / "pciSeq.pickle")

    for label, user_class, name in CELLS:
        # the figure is 14 inches wide and gets shrunk to fit the page column, which
        # makes the default text tiny. Bump the font sizes for the docs copy only.
        with matplotlib.rc_context({"font.size": 17, "axes.titlesize": 17,
                                    "axes.labelsize": 16, "xtick.labelsize": 15,
                                    "ytick.labelsize": 15, "legend.fontsize": 15}):
            ged, _, fig = obj.check_cell(label, user_class)
            fig.savefig(FIG_DIR / f"{name}.png", dpi=90)
        assigned = ged.columns[0][0].replace("Cells typed as ", "")
        (TABLE_DIR / f"{name}.md").write_text(cell_table_html(ged, assigned, user_class))
        print(f"wrote {name}.png and {name}.md")

    # check_spot draws its two plotly charts with fig.show() and does not hand the
    # figures back. So swap show() for something that just keeps them, then save them.
    shown = []
    go.Figure.show = lambda self, *args, **kwargs: shown.append(self)
    for spot_id, name in SPOTS:
        shown.clear()
        df = obj.check_spot(spot_id, show_plot=True)
        score_fig, prob_fig = shown  # same order as check_spot draws them
        score_fig.write_image(FIG_DIR / f"{name}-scores.png", width=1000, height=550, scale=1)
        prob_fig.write_image(FIG_DIR / f"{name}-probs.png", width=1000, height=450, scale=1)
        (TABLE_DIR / f"{name}.md").write_text(to_markdown(df))
        print(f"wrote {name}-scores.png, {name}-probs.png and {name}.md")


if __name__ == "__main__":
    main()
