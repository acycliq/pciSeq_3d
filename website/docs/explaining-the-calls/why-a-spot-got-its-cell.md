
# How a spot's call was made

A spot is assigned to one of its `nNeighbors` nearest cells or to the background, as a
probability. Each candidate cell gets a score, the background gets one too, and a softmax
over them gives the probabilities:

$$
\begin{aligned}
\text{score}_c = \;& \text{Gaussian fit} && \text{position} \\
+\;& \text{class expression} + \text{cell scale} + \text{cell-gene scale} + \text{gene efficiency} && \text{expression} \\
+\;& \text{inside-cell bonus} && \text{segmentation}
\end{aligned}
$$

The background scores the [misread density](../how-it-works/misread-density.md) of the
spot's gene, the same at every position. The terms are described in
[assigning spots to cells](../how-it-works/spots-to-cells.md).

[`check_spot`](../api/reference.md#check-spot) shows these terms for every candidate of a
spot. It reads the values the model used in its last spot update, so what it shows agrees
with the probabilities in `geneData`.

The examples use the same two fits as the [cell page](why-a-cell-got-its-type.md), with
`mrf_beta = 1.5` and `mrf_beta = 0`, and one spot of cell 18223.

## Calling `check_spot`

Spot 1642419 is a Synpr spot. In the fit with the spatial term it goes to cell 18223:

```python
import pandas as pd

obj = pd.read_pickle('pciSeq.pickle')
obj.check_spot(1642419)
```

The spot is given by its index in the spots table passed to `fit`.

![check_spot score decomposition for spot 1642419](/explaining-the-calls/spot-1642419-mrf-scores.png)

The bars are the score of each candidate, term by term, with the candidates ordered by
distance and the background last. The black tick is the total. A taller total means a
better explanation of the spot.

![check_spot assignment probabilities for spot 1642419](/explaining-the-calls/spot-1642419-mrf-probs.png)

The second chart is the softmax of those totals, which is the assignment the model uses:
0.74 on cell 18223, 0.13 on cell 21574, 0.10 on cell 17371 and 0.01 on the background.

Each expression term is a log, so its sign says whether the quantity inside the log is
above or below 1.

Take the class expression, the orange segment of every bar in the score chart above. It sits
above zero, so the quantity inside its log is above 1. The term is
`log(mean expression x Inefficiency + SpotReg)`, averaged over the cell's class
probabilities. Every candidate here is `037 DG Glut` with probability near 1, so the
average reduces to the value for DG. Synpr has a mean of 24.68 in `037 DG Glut`, and with
`Inefficiency` 0.1 and `SpotReg` 0.1 that gives `log(24.68 x 0.1 + 0.1) = log(2.57) = 0.943`, the value every DG
candidate shows here.

It is not positive by construction: the sign follows the class mean. For a cell
confidently of one class, as here, the term is zero when
`mean expression x Inefficiency + SpotReg` equals 1, which in this run is a class mean of
9 counts, and negative below that. For a cell spread over several classes it is the
average of those logs.

The cell scale and the cell-gene scale are positive when the cell holds more than its class
predicts. Here the cell-gene scale is positive for cell 18223, 0.18, and cell 21574, 0.07, the
cells with more Synpr than their class predicts; the cell scale is positive for cells 22339,
0.28, and 21574, 0.10, which hold more reads in total than their class predicts.

## The returned table

`check_spot` returns the same numbers as a table, one row per candidate and a final row
for the background:

<!--@include: ./_tables/spot-1642419-mrf.md-->

- **Gaussian fit**. The log density of the spot's position under the cell's Gaussian
  footprint. When several cells compete for the same spot and everything else is equal, it
  goes to the nearest one. It does not depend on the gene.
- **class expression, the alignment**. How well the cell's likely class matches the gene. When several cells
  compete for the same spot and everything else is equal, it goes to the cell whose likely
  class expresses that gene most. Two cells with the same class probabilities get the same
  value, since the gene is the same for every candidate of a spot, which is why every
  candidate here shows 0.943.
- **cell scale, the gravity**. The factor behind it is theta: how far a cell's total
  counts sit above or below what its likely class predicts. When several cells compete for
  the same spot and everything else is equal, it goes to the cell already holding more
  reads in total. Only the totals count here, so the term is the same whatever the gene of
  the spot.
- **cell-gene scale, the enrichment**. The factor behind it is gamma: how much of a gene
  is observed in a cell relative to the amount expected for its likely class. When several
  cells compete for the same spot and all other factors are equal, the spot is assigned to
  the cell with the higher observed-to-expected ratio for that gene.
- **gene efficiency**. The factor behind it is eta: how well the gene is detected across
  the whole experiment. It is the same for every candidate cell, so it cannot choose
  between them. It does weigh the cells against the background, which carries no such
  term: a poorly detected gene lowers every cell's score and makes the background more
  likely.
- **misread**. The background's whole score: how likely a spot of this gene is to be a
  misread. It is the same at every position, so a cell has to beat it or the spot goes to
  the background.
- **sum, prob**. The total and its softmax over the rows.

<p class="table-note">Gene efficiency and misread density are not the same thing. The
first scales what a cell is expected to hold of the gene, so it sits in every cell's
score. The second is the background's own rate for that gene, learned from the spots no
cell explains. A gene can be well detected and still produce many misreads.</p>

<p class="table-note">The inside-cell bonus is not shown: <code>InsideCellBonus</code> is 0
in these runs, so the column was empty. When it is set, a spot whose pixel falls inside the
cell's segmentation gets that bonus added to its score.</p>

Cell 18223 is not the only candidate close to the spot: its Gaussian fit is 1.7 above cell
17371 and 2.4 above cell 21574. It wins on the expression terms, where it has the largest
cell-gene scale, 0.18 against 0.07 for cell 21574 and negative values for the rest.

## The same spot without the spatial term

In the fit with `mrf_beta = 0`, cell 18223 is `030 L6 CT CTX Glut` rather than
`037 DG Glut`. The spot has not moved, so its position terms are unchanged:

```python
obj_nomrf = pd.read_pickle('pciSeq_nomrf.pickle')
obj_nomrf.check_spot(1642419)
```

<!--@include: ./_tables/spot-1642419-nomrf.md-->

| term for cell 18223 | without MRF | with MRF |
| --- | --- | --- |
| Gaussian fit | -10.405 | -10.405 |
| class expression | -2.203 | +0.943 |
| sum | -14.149 | -10.590 |
| prob | 0.06 | 0.74 |

`Gaussian fit` is identical, as it must be. What changes is the class expression: `037 DG Glut`
expresses Synpr and `030 L6 CT CTX Glut` does not, and the class expression term falls from +0.943
(with mrf) to -2.203 (without mrf). The spot is now assigned to cell 18223 with probability 6%
and its most likely parent cell is 21574 with prob 49.2%. It is worth noticing that cell 21574 is the
third closest to the spot. It has the type `037 DG Glut`. Also cell 17371 is the second closest, also `037 DG Glut`
. The spot however is not assigned to cell 17371 despite being closer because it holds
fewer reads in total, 60.0 against 98.8, and expresses fewer Synpr, 0.5 against 3.0, than
cell 21574.

<figure class="diagram">
<div class="two-panel">
  <div>
    <img src="/explaining-the-calls/spot-1642419-mrf-map.png" alt="the three candidate cells with the spatial term">
    <p>with the spatial term</p>
  </div>
  <div>
    <img src="/explaining-the-calls/spot-1642419-nomrf-map.png" alt="the same cells without the spatial term">
    <p>without it</p>
  </div>
</div>
<figcaption>The same field at plane 57, with Synpr the only gene shown. Cells carry the
colour of their class, and the line joins the spot to the cell it was assigned to. Cell
18223 is <code>037 DG Glut</code> on the left and <code>030 L6 CT CTX Glut</code>, in
green, on the right, while cells 17371 and 21574 stay <code>037 DG Glut</code> in both.
</figcaption>
</figure>
