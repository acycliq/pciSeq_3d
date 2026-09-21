
# Why a spot got its cell

A spot is assigned to one of its `nNeighbors` nearest cells or to the background, as a
probability. Each candidate cell gets a score, the background gets one too, and a softmax
over them gives the probabilities:

$$
\begin{aligned}
\text{score}_c = \;& \text{Gaussian LogLik} && \text{position} \\
+\;& \text{alignment} + \text{gravity} + \text{enrichment} + \text{gene efficiency} && \text{expression} \\
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
`mrf_beta = 1.5` and `mrf_beta = 0`, and two spots of cell 18223.

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

## The returned table

`check_spot` returns the same numbers as a table, one row per candidate and a final row
for the background:

<!--@include: ./_tables/spot-1642419-mrf.md-->

- **position**. The log density of the spot's position under the cell's Gaussian
  footprint. When several cells compete for the same spot and everything else is equal, it
  goes to the nearest one. It does not depend on the gene.
- **alignment**. How well the cell's likely class matches the gene. When several cells
  compete for the same spot and everything else is equal, it goes to the cell whose likely
  class expresses that gene most. Two cells with the same class probabilities get the same
  value, since the gene is the same for every candidate of a spot, which is why every
  candidate here shows 0.943.
- **gravity**. The cell's total counts compared with what its likely class predicts. When
  several cells compete for the same spot and everything else is equal, it goes to the
  cell already holding more reads in total than its class predicts. Only the totals count
  here, so the term is the same whatever the gene of the spot.
- **enrichment**. How much of a gene is observed in a cell relative to the amount expected
  for its likely class. When several cells compete for the same spot and all other factors
  are equal, the spot is assigned to the cell with the higher observed-to-expected ratio
  for that gene.
- **gene inefficiency**. How well the gene is detected across the whole experiment. It is
  the same for every candidate cell, so it cannot choose between them. It does weigh the
  cells against the background, which carries no such term: a poorly detected gene lowers
  every cell's score and makes the background more likely.
- **misread**. The background's whole score: how likely a spot of this gene is to be a
  misread. It is the same at every position, so a cell has to beat it or the spot goes to
  the background.
- **sum, prob**. The total and its softmax over the rows.

<p class="table-note">Gene inefficiency and misread density are not the same thing. The
first scales what a cell is expected to hold of the gene, so it sits in every cell's
score. The second is the background's own rate for that gene, learned from the spots no
cell explains. A gene can be well detected and still produce many misreads.</p>

<p class="table-note">The inside-cell bonus is not shown: <code>InsideCellBonus</code> is 0
in these runs, so the column was empty. When it is set, a spot whose pixel falls inside the
cell's segmentation gets that bonus added to its score.</p>

Cell 18223 is not the only candidate close to the spot: cell 17371 is 1.7 further in
position and cell 21574 is 2.4 further. Its advantage comes from the expression terms,
where it is the only candidate with a positive enrichment and the largest total.

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
| position | -10.405 | -10.405 |
| alignment | -2.203 | +0.943 |
| sum | -14.149 | -10.590 |
| prob | 0.06 | 0.74 |

`position` is identical, as it must be. What changes is the alignment: `037 DG Glut`
expresses Synpr and `030 L6 CT CTX Glut` does not, so the same spot is worth 3.1 more to
the cell once the cell is DG. The spot goes to cell 21574 with probability 0.49 instead,
a DG cell 2.4 further away.

## A spot that moves the other way

Spot 1533144 is a Neurod6 spot, a gene of `030 L6 CT CTX Glut`. Cell 18223 holds it in the
fit without the spatial term and loses it in the fit with it:

| term for cell 18223 | without MRF | with MRF |
| --- | --- | --- |
| position | -10.884 | -10.884 |
| alignment | -0.673 | -2.207 |
| sum | -10.258 | -11.812 |
| prob | 0.75 | 0.35 |

<!--@include: ./_tables/spot-1533144-mrf.md-->

In the fit with the spatial term the spot goes to cell 17371 with probability 0.42, and
0.16 goes to the background. Cell 17371 is 0.05 further from the spot and has the same
alignment, since both cells are DG; it wins on the other expression terms.

Taken together the two spots are the loop the cell page describes: the class of a cell
sets the attention it gives to a gene, the attention decides which spots it holds, and the
spots it holds decide its class in the next iteration.
