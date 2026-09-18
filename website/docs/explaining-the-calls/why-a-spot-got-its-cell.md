
# Why a spot got its cell

A spot is assigned to one of its `nNeighbors` nearest cells or to the background, as a
probability. Each candidate cell gets a score, the background gets one too, and a softmax
over them gives the probabilities:

$$
\text{score}_c = \underbrace{\text{mvn\_loglik}}_{\text{position}}
+ \underbrace{\text{attention} + \text{expr\_fluct} + \text{cell\_inefficiency} + \text{gene\_inefficiency}}_{\text{expression}}
+ \underbrace{\text{bonus}}_{\text{segmentation}}
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

- **mvn_loglik.** The log density of the spot's position under the cell's Gaussian
  footprint. It does not depend on the gene.
- **attention, expr_fluct, cell_inefficiency, gene_inefficiency.** The expression terms:
  the expected log count of the gene under the cell's class probabilities, and the log of
  gamma, theta and eta.
- **bonus.** The inside-cell bonus, `InsideCellBonus`, 0 here.
- **misread.** The score of the background, the log misread density of the gene.
- **sum, prob.** The total and its softmax over the rows.

Cell 18223 is not the only candidate close to the spot: cell 17371 is 1.7 further in
mvn_loglik and cell 21574 is 2.4 further. Its advantage comes from the expression terms,
where it is the only candidate with a positive expr_fluct and the largest total.

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
| mvn_loglik | -10.405 | -10.405 |
| attention | -2.203 | +0.943 |
| sum | -14.149 | -10.590 |
| prob | 0.06 | 0.74 |

`mvn_loglik` is identical, as it must be. What changes is attention: `037 DG Glut`
expresses Synpr and `030 L6 CT CTX Glut` does not, so the same spot is worth 3.1 more to
the cell once the cell is DG. The spot goes to cell 21574 with probability 0.49 instead,
a DG cell 2.4 further away.

## A spot that moves the other way

Spot 1533144 is a Neurod6 spot, a gene of `030 L6 CT CTX Glut`. Cell 18223 holds it in the
fit without the spatial term and loses it in the fit with it:

| term for cell 18223 | without MRF | with MRF |
| --- | --- | --- |
| mvn_loglik | -10.884 | -10.884 |
| attention | -0.673 | -2.207 |
| sum | -10.258 | -11.812 |
| prob | 0.75 | 0.35 |

<!--@include: ./_tables/spot-1533144-mrf.md-->

In the fit with the spatial term the spot goes to cell 17371 with probability 0.42, and
0.16 goes to the background. Cell 17371 is 0.05 further from the spot and has the same
attention, since both cells are DG; it wins on the other expression terms.

Taken together the two spots are the loop the cell page describes: the class of a cell
sets the attention it gives to a gene, the attention decides which spots it holds, and the
spots it holds decide its class in the next iteration.
