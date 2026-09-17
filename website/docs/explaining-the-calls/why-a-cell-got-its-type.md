
# Why a cell got its type

A cell is assigned the type with the highest posterior probability. For every type the
model adds three terms, and a softmax over all types turns the totals into
probabilities:

$$
\text{score}_k = \underbrace{\text{gene log-likelihood}_k}_{\text{the cell's counts}}
+ \underbrace{\log \pi_k}_{\text{the prior}}
+ \underbrace{\text{MRF}_k}_{\text{the neighbours}}
$$

[`check_cell`](../api/reference.md#check-cell) compares the assigned type of a cell with
a second type of your choice and shows how each of these terms contributes. It reads the
values the model used in its last cell type update, so the figure agrees with the
probabilities in `cellData`.

The examples on this page use the 3D dataset described in the [overview](overview.md),
fitted with `mrf_beta = 1.5`. The second half of the page compares the same cell in a
fit without the spatial term, `mrf_beta = 0`.

## The figure

Cell 7768 has 45 spots and is assigned `037 DG Glut` with probability 1.00. The following
compares it with `030 L6 CT CTX Glut`, a cortical type:

```python
import pandas as pd

# the fitted model, saved by pciSeq.fit at <output_path>/pciSeq/data/debug/pciSeq.pickle
obj = pd.read_pickle('pciSeq.pickle')
obj.check_cell(7768, '030 L6 CT CTX Glut')
```

`fit` saves the fitted model when `save_data` is `True`, the default, see
[the fitted model](overview.md#the-fitted-model).
The cell is given by its label in the segmentation passed to `fit`.

![check_cell output for cell 7768](/explaining-the-calls/cell-7768-mrf.png)

The figure has four panels.

1. **Top left.** For each gene, the log-likelihood of the cell's count under the assigned
   type minus its log-likelihood under the second type. A positive difference means the
   count is more likely under the assigned type, so the gene favours it. The panel shows
   up to `top_n` genes with the largest positive differences, and the title gives their
   sum.
2. **Top right.** The same difference for the genes where it is negative: the count is
   more likely under the second type, so the gene favours the second type. The panel
   shows up to `top_n` genes with the most negative differences.
3. **Bottom left.** The three terms of the score for both types: the gene log-likelihood
   summed over all genes, the log prior and the MRF term. Higher is better: the type with
   the larger total wins.
4. **Bottom right.** The posterior over all types, for the `top_classes` most likely
   types and the second type. The note gives the probability of the types not shown.

The top panels explain the gene log-likelihood in panel 3. Panel 3 shows whether the
genes, the prior or the neighbours decide the call. Panel 4 shows how confident the call
is, and which other types remain plausible.

For cell 7768:

- **Top panels.** Sema5a, Gad1 and Cdh9 favour `037 DG Glut`, by 3.08, 1.78 and 1.59;
  the ten genes on that side sum to 10.65. Rprm and Glul favour `030 L6 CT CTX Glut`, by
  2.26 and 1.25; the ten genes on that side sum to -5.90.
- **Bottom left.** The gene log-likelihood is -110.4 for `037 DG Glut` against -116.5 for
  `030 L6 CT CTX Glut`, so the genes alone favour DG by 6.1. The priors are equal. The MRF
  term is 7.2 for `037 DG Glut` and 0 for `030 L6 CT CTX Glut`: four of the cell's nine
  neighbours are DG cells and none is L6 CT.
- **Bottom right.** The posterior is 100% on `037 DG Glut`.

Genes and neighbours agree, and the call is certain.

## The table

`check_cell` also returns a table with one row for each gene in the two gene charts of the
figure (top left and top right):

<!--@include: ./_tables/cell-7768-mrf.md-->

The table has the same columns as the DataFrame `check_cell` returns:

- **Cell 7768, observed.** The count of the gene in this cell, the sum of the assignment
  probabilities of its spots.
- **Model prediction for cell 7768.** The number of spots of the gene the model expects in
  this cell if the cell were of that type: the cell type definition rescaled by the
  [scaling factors](../how-it-works/warping-the-reference.md), see the
  [example below](#the-prediction-is-not-the-observed-mean). A gene favours the type whose
  prediction is closer to the observed count.
- **Cells typed as a type, observed mean.** The average count of the gene in the cells
  currently assigned to that type, weighted by their probability of the type. It is
  measured, not predicted, and it does not enter the score.

Sema5a is the strongest gene. The cell has 11.9 Sema5a spots. `037 DG Glut` predicts 4.4
and `030 L6 CT CTX Glut` predicts 1.9. Both predict too few, but `037 DG Glut` predicts
more, and the evidence from a gene grows with the number of its spots. With 11.9 spots
Sema5a contributes 3.08, the largest bar in the top-left panel. The cells assigned to
`037 DG Glut` have 6.9 Sema5a spots on average, those assigned to `030 L6 CT CTX Glut`
2.7.

::: details How the 3.08 is computed
The gene log-likelihood is a negative binomial with mean $\mu$, the expected count, and
dispersion $r$, the `rSpot` setting, here 2. For a count $x$:

$$
\log \text{NB}(x;\, r, \mu) = x \log\frac{\mu}{r+\mu} + r \log\frac{r}{r+\mu} + \log\frac{\Gamma(x+r)}{x!\,\Gamma(r)}
$$

The last term does not depend on $\mu$ and cancels in the difference between two types
$A$ and $B$:

$$
\Delta = x \left[\log\frac{\mu_A}{r+\mu_A} - \log\frac{\mu_B}{r+\mu_B}\right] + r \log\frac{r+\mu_B}{r+\mu_A}
$$

The first term grows with the count, the second does not. For Sema5a, with the expected
counts from the table:

| type | $\mu$ | $\mu/(r+\mu)$ | $\log$ |
| --- | --- | --- | --- |
| `037 DG Glut` | 4.438 | 0.6893 | -0.3720 |
| `030 L6 CT CTX Glut` | 1.913 | 0.4889 | -0.7156 |

Each spot adds $-0.3720 - (-0.7156) = 0.3436$ in favour of `037 DG Glut`. The second term
is $2 \log(3.913/6.438) = -0.996$ and favours `030 L6 CT CTX Glut`, which predicts fewer
spots overall. With $x = 11.851$:

$$
\Delta = 11.851 \times 0.3436 - 0.996 = 3.08
$$

With a single Sema5a spot, $\Delta = -0.65$: one spot of a gene that both types predict
would favour the type that predicts fewer.
:::

### The prediction is not the observed mean

For Sema5a the model predicts 4.44 spots in cell 7768 as `037 DG Glut`, while the cells
typed as `037 DG Glut` have 6.94 on average. The two numbers are different quantities.

The prediction is built from the single-cell reference:

| factor | value |
| --- | --- |
| reference mean of Sema5a in `037 DG Glut` | 18.49 |
| × `Inefficiency` | 0.1 |
| × eta of Sema5a | 4.04 |
| × theta of cell 7768 as `037 DG Glut` | 0.58 |
| + `SpotReg` | 0.1 |
| **prediction** | **4.44** |

gamma, the factor for one gene in one cell, is not part of the product. It expresses the
discrepancy between the observed count in the cell and the prediction, regularised by
`rSpot`.

The prediction is lower than the observed mean of the cells typed as `037 DG Glut`, 6.94,
and this is expected. eta is one number per gene, shared by all cell types, so it cannot
correct the reference for each type separately, and the reference does not match the in
situ data exactly. What decides the call is the comparison within the cell: the observed
count of 11.85 is closer to the prediction under `037 DG Glut`, 4.44, than under
`030 L6 CT CTX Glut`, 1.91.

## The same cell without the spatial term

Fitted with `mrf_beta = 0`, and otherwise identical settings, cell 7768 is assigned
`030 L6 CT CTX Glut`:

```python
# the fitted model of the run with mrf_beta = 0
obj_nomrf = pd.read_pickle('pciSeq_nomrf.pickle')
obj_nomrf.check_cell(7768, '037 DG Glut')
```

![check_cell output for cell 7768 without the MRF](/explaining-the-calls/cell-7768-nomrf.png)

- **Top panels.** The order of the genes is reversed. Rprm and Glul now favour
  `030 L6 CT CTX Glut` by 3.57 and 2.78, Sema5a, Gad1 and Cdh9 favour `037 DG Glut` by
  2.43, 1.40 and 1.04.
- **Bottom left.** The gene log-likelihood is -111.0 for `030 L6 CT CTX Glut` against
  -113.4 for `037 DG Glut`: the genes favour L6 CT by 2.4. There is no MRF term.
- **Bottom right.** The posterior is 90.9% `030 L6 CT CTX Glut` and 8.2% `037 DG Glut`.

In both fits the cell's own genes support its call. What differs is not how the evidence
is weighed but the evidence itself: the two fits give the cell different spots.

| gene | favours | spots, without MRF | spots, with MRF |
| --- | --- | --- | --- |
| Glul | L6 CT | 3.66 | 1.82 |
| Rprm | L6 CT | 3.27 | 2.35 |
| Sema5a | DG | 10.01 | 11.85 |
| Tafa1 | DG | 0.04 | 0.43 |

The total number of spots is almost the same, 44 against 45. Scoring the spots the cell
holds in the fit with the MRF under the parameters of the fit without it turns the
preference of the genes from 2.4 for L6 CT to 5.5 for DG, so the difference in the call
comes from which spots the cell holds.
