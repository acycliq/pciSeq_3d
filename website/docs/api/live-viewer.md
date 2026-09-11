---
title: Live viewer
description: Watch the cell typing in a browser while pciSeq.fit() runs, and what each control on the page does.
---

# Live viewer

The live viewer shows the cells in a browser while [`fit`](./reference#fit) is still
running. Every cell is drawn at its position and coloured by the class it is currently
assigned to, and the picture updates after each iteration. It is useful for seeing where
the model is unsure, which cells keep changing class, and how quickly a run settles.

It only reads the class probabilities after each iteration, it does not change them, so a
run gives the same result with the viewer on or off.

## Turning it on

```python
opts = {
    "realtime_viewer": True,
    "realtime_viewer_port": 5001,   # the default
}
cellData, geneData = pciSeq.fit(spots=spots, coo=coo, scRNAseq=scRNAseq, opts=opts)
```

A browser tab opens by itself at `http://127.0.0.1:5001`. If it does not, open that
address yourself. The page says *Waiting for algorithm to start...* until the first
iteration is done, which can take a while on a large dataset since the data is prepared
first.

Two more settings help on large sections:

- `realtime_viewer_max_cells`: draw only this many cells, the ones the model is most sure
  about. `None` (the default) draws all of them.
- `realtime_viewer_fixed_radius`: draw every cell at this radius. `None` draws each cell
  at its own size, worked out from its area.

When `fit` returns the viewer shuts down and the page shows it is no longer connected. It
keeps the last picture and you can still hover and hide classes, but anything that needs
the running model, such as the cell diagnostics below, stops working.

::: tip Running on a remote machine
The viewer only listens on `127.0.0.1`, so it cannot be opened from another computer
directly. Forward the port over ssh and open the address on your own machine:

```bash
ssh -L 5001:127.0.0.1:5001 you@server
```
:::

## The page

**Header.** Whether the page is connected to the run, the current iteration, the
convergence value (see below) and the number of cells. The small box on the picture shows
the connection and the cell count too.

**Hover** over a cell for its label, its class, its position and how sure the model is
about that class.

**Pan and zoom** with the mouse.

**Cells that just changed class** get a cyan ring that fades out over two seconds, so you
can see where the assignments are still moving.

## Cell classes

The legend lists every class with its colour and how many cells are assigned to it.

- Click a class to hide it, click again to bring it back. **Show All** and **Hide All** do
  every class at once.
- Press `/` to jump to the filter box and type part of a class name to shorten the list.
  `Esc` clears it.
- Drag the handle under the list to make it taller or shorter.

## Updates

**All Cells** draws every cell. **Updated Only** draws just the cells whose probability for
their top class moved by at least the threshold since the previous iteration. The slider
sets the threshold, 1% by default. Two counts show how many cells were updated and how many
changed class. On the first iteration every cell counts as updated.

**Plane** is for 3D data. It adds a slider at the bottom of the picture and draws only the
cells whose centroid falls in the chosen z plane. It starts on the middle plane. The button
is greyed out for 2D data.

## Convergence

The chart plots the convergence value for every iteration: the biggest change in any
spot's cell assignment probability since the previous iteration. The dashed line is
`CellCallTolerance`, and the run stops once the value drops below it (or at `max_iter`).
Hover over a point to read its value.

## Cell diagnostics

`Ctrl`+click a cell (`Cmd`+click on a Mac) to open the diagnostics drawer. It explains, gene
by gene, why the model prefers the cell's class over another one. Pick that other class in
**Compare against**, it starts on Zero.

- The two charts show the 10 genes that favour the assigned class the most and the 10
  that favour the compared class the most, measured as the difference in log-likelihood.
- The table has the counts of those genes in this cell next to the mean counts in the
  cells typed as each of the two classes.

Comparing a class against itself is not allowed and gives an error in the drawer. Drag
the top edge of the drawer to resize it.

## Custom colours

By default the classes get evenly spread colours. To use your own, click **Import Color
Scheme** and choose a JSON file that maps class names to colours:

```json
{
  "037 DG Glut": "#5C79CC",
  "038 DG-PIR Ex IMN": "#86FF4D",
  "Zero": "#000000"
}
```

- The names must be spelled exactly like the columns of your scRNAseq data.
- A colour can be a hex code or a CSS colour name.
- Classes missing from the file keep their default colour. Names in the file that are not
  in the data are ignored and listed in the browser console.
- You can load the file before the first iteration arrives. It is then applied as soon as
  the class names are known.

::: tip Chrome and Opera on Linux
In the file dialog, select the file and press **Open**. Double-clicking the file does not
always pass it to the page in Chromium based browsers. Firefox is fine either way.
:::

## Opening it late

The viewer remembers the cell positions and the latest iteration, so a tab opened or
reloaded in the middle of a run catches up straight away. You can close the tab and
come back to it without affecting the run.
