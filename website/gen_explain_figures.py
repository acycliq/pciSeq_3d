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

import io
import sqlite3

import matplotlib
matplotlib.use("Agg")  # no window, we only save the figures
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps

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

# the DAPI background of the same run, as map tiles, and the cell outlines per plane.
# both are made for the viewer, see pciSeq/src/tiling. Skipped if they are not around.
MBTILES = pathlib.Path.home() / "data/Christina/pciSeq_HC_silver/Espio/coppafisher_1100/dapi_Espio.mbtiles"
BOUNDARIES = RUNS["mrf"].parents[1] / "viewer_data/arrow_boundaries"

DG = "037 DG Glut"
L6CT = "030 L6 CT CTX Glut"
CA1 = "016 CA1-ProS Glut"

# (run, cell label, class to compare against, file name)
CELLS = [
    ("mrf", 18223, L6CT, "cell-18223-mrf"),    # the walkthrough: DG with the spatial prior
    ("nomrf", 18223, DG, "cell-18223-nomrf"),  # the same cell called L6 CT without it
]
# (run, spot id, file name): two spots of cell 18223 that change hands between the runs
SPOTS = [
    ("nomrf", 1642419, "spot-1642419-nomrf"),  # Synpr, the cell only gets it with the mrf
    ("mrf", 1642419, "spot-1642419-mrf"),
    ("nomrf", 1533144, "spot-1533144-nomrf"),  # Neurod6, the cell loses it with the mrf
    ("mrf", 1533144, "spot-1533144-mrf"),
]

# the tissue map: which cell to show and how the two panels look
MAP = {"cell": 18223, "name": "cell-18223-map", "zoom_w": 210, "zoom_h": 140,
       "panel_w": 1200, "panel_h": 800, "gutter": 16,
       "accent": (255, 96, 80), "faint": (120, 210, 255),
       "font": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"}

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


def _tile_window(con, plane, zoom, cx, cy, w_img, h_img, img_w, img_h, tile=256):
    """Stitch the tiles covering a w_img by h_img window of image pixels centred on (cx, cy).

    Returns the image, the image pixel to tile pixel scale, and the top left corner of the
    window in tile pixels, so boundaries can be drawn on top.
    """
    s = tile * 2 ** zoom / max(img_w, img_h)
    # keep the window inside the image, otherwise the panel gets a black band
    cx = min(max(cx, w_img / 2), img_w - w_img / 2)
    cy = min(max(cy, h_img / 2), img_h - h_img / 2)
    x0, y0 = (cx - w_img / 2) * s, (cy - h_img / 2) * s
    x1, y1 = (cx + w_img / 2) * s, (cy + h_img / 2) * s
    tx0, ty0, tx1, ty1 = int(x0 // tile), int(y0 // tile), int(x1 // tile), int(y1 // tile)
    canvas = Image.new("L", ((tx1 - tx0 + 1) * tile, (ty1 - ty0 + 1) * tile))
    rows = con.execute(
        "select tile_column, tile_row, tile_data from tiles where plane_id=? and zoom_level=? "
        "and tile_column between ? and ? and tile_row between ? and ?",
        (plane, zoom, tx0, tx1, ty0, ty1))
    for tx, ty, blob in rows:
        canvas.paste(Image.open(io.BytesIO(blob)).convert("L"), ((tx - tx0) * tile, (ty - ty0) * tile))
    im = canvas.crop((round(x0 - tx0 * tile), round(y0 - ty0 * tile),
                      round(x1 - tx0 * tile), round(y1 - ty0 * tile))).convert("RGB")
    return im, s, (x0, y0)


def _outline(im, scale, origin, bounds, labels, colour, width, fill_alpha=0.0):
    """Draw the outlines of the given cells on a stitched window, optionally filled.

    fill_alpha is how opaque the fill is, 0 for outline only.
    """
    drawn = 0
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    fill = colour + (int(255 * fill_alpha),) if fill_alpha else None
    for lab in labels:
        if lab not in bounds.index:
            continue
        r = bounds.loc[lab]
        pts = [(x * scale - origin[0], y * scale - origin[1]) for x, y in zip(r.x_list, r.y_list)]
        if fill:
            d.polygon(pts, fill=fill)
        d.line(pts + [pts[0]], fill=colour + (255,), width=width, joint="curve")
        drawn += 1
    im.paste(Image.alpha_composite(im.convert("RGBA"), over).convert("RGB"), (0, 0))
    return drawn


def cell_map(obj, label):
    """Two panels: the whole section with the cell marked, and the cell with its neighbours.

    The background is the DAPI of the run, read straight out of the viewer's mbtiles, and
    the outlines come from the per plane boundary files. Returns the numbers the caption
    quotes, or None when the tiles are not on this machine.
    """
    if not MBTILES.exists():
        print(f"no mbtiles at {MBTILES}, skipping the tissue map")
        return None

    cfg = obj.config
    lm = cfg["label_map"]
    row = lm[label] if lm else label
    cx, cy, cz = obj.cells.centroid.iloc[row][["x", "y", "z"]]
    # z is stored scaled to the xy pixel size, so undo that to get the plane
    vx, vy, vz = cfg["voxel_size"]
    plane = int(round(cz / (vz / vx)))
    inv = {v: k for k, v in lm.items()} if lm else None
    nbrs = [inv[r] if inv else r for r in obj.cells.nbrs["indices"][row]]

    con = sqlite3.connect(f"file:{MBTILES}?mode=ro", uri=True)
    meta = dict(con.execute("select name, value from metadata"))
    img_w, img_h = int(meta["width"]), int(meta["height"])
    maxzoom = int(meta["maxzoom"])
    bounds = pd.read_feather(BOUNDARIES / f"boundaries_plane_{plane:02d}.feather").set_index("label")

    pw, ph = MAP["panel_w"], MAP["panel_h"]
    # a: the whole section, cropped to the same shape as panel b so nothing is stretched
    a, sa, oa = _tile_window(con, plane, maxzoom - 4, img_w / 2, img_h / 2,
                             img_w, img_w * ph / pw, img_w, img_h)
    a = ImageOps.autocontrast(a, cutoff=(0.2, 0.05))  # the overview is dark, lift it a little
    ax, ay = cx * sa - oa[0], cy * sa - oa[1]
    r = 0.022 * max(a.size)
    ImageDraw.Draw(a).ellipse([ax - r, ay - r, ax + r, ay + r], outline=MAP["accent"], width=22)
    a = a.resize((pw, ph), Image.LANCZOS)

    # b: the cell and the neighbours the mrf term listens to
    b, sb, ob = _tile_window(con, plane, maxzoom, cx, cy, MAP["zoom_w"], MAP["zoom_h"], img_w, img_h)
    on_plane = _outline(b, sb, ob, bounds, nbrs, MAP["faint"], 4, fill_alpha=0.15)
    _outline(b, sb, ob, bounds, [label], MAP["accent"], 6, fill_alpha=0.3)
    b = b.resize((pw, ph), Image.LANCZOS)
    con.close()

    out = Image.new("RGB", (pw * 2 + MAP["gutter"], ph), (255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(b, (pw + MAP["gutter"], 0))
    d = ImageDraw.Draw(out)
    big = ImageFont.truetype(MAP["font"], 34)
    small = ImageFont.truetype(MAP["font"], 28)
    for x, txt in ((0, "a"), (pw + MAP["gutter"], "b")):
        d.text((x + 18, 12), txt, font=big, fill=(255, 255, 255))
        d.text((x + 18, ph - 44), f"cell {label}", font=small, fill=MAP["accent"])
    out.save(FIG_DIR / f"{MAP['name']}.png")
    print(f"wrote {MAP['name']}.png")
    return {"cell": label, "plane": plane, "centroid": [float(cx), float(cy), float(cz)],
            "neighbours": [int(n) for n in nbrs], "neighbours_on_plane": on_plane,
            "zoom_window_px": [MAP["zoom_w"], MAP["zoom_h"]]}


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
            if run == "mrf":
                numbers["cell_map"] = cell_map(obj, label)
                # the page walks through the prediction of the strongest gene
                top_gene = contr["diff"].idxmax()
                numbers["top_gene_breakdown"] = gene_breakdown(
                    obj, label, top_gene, [DG, L6CT, "017 CA3 Glut", CA1])
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
