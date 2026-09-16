
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

The examples below use the CA1 example data, see the [overview](overview.md) for how to
run it and load the fitted model.

## The figure

```python
obj.check_cell(1023, 'Sst.Erbb4.Rgs10')
```

![check_cell output for cell 1023](/explaining-the-calls/cell-1023.png)

The figure has four panels.

1. **Top left.** The genes whose counts favour the assigned type, as the difference in
   log-likelihood between the two types, up to `top_n` genes. The title gives their sum.
2. **Top right.** The same for the genes that favour the second type. The differences
   are negative.
3. **Bottom left.** The three terms of the score for both types: the gene log-likelihood
   summed over all genes, the log prior and the MRF term. The type with the larger total
   wins.
4. **Bottom right.** The posterior over all types, for the `top_classes` most likely
   types and the second type. The note gives the probability of the types not shown.

The top panels explain the gene log-likelihood in panel 3. Panel 3 shows whether the
genes, the prior or the neighbours decide the call. Panel 4 shows how confident the call
is, and which other types remain plausible.

## Example 1: the genes decide

Cell 1023 has 83 spots and is assigned `Pvalb.Tac1.Sst` with probability 1.00. It is
compared with `Sst.Erbb4.Rgs10`.

- **Top panels.** Tac1 alone contributes a difference of 11.8. The ten genes that favour
  `Pvalb.Tac1.Sst` sum to 17.7, and the ten that favour `Sst.Erbb4.Rgs10` to -6.0.
- **Bottom left.** The gene log-likelihood is -113.2 against -124.8. The prior is the
  same for both types and the MRF term is 0 for both, so the gene counts alone decide.
- **Bottom right.** The posterior is 100% on `Pvalb.Tac1.Sst`.

`check_cell` also returns a table with one row per gene shown in the top panels:

<!--@include: ./_tables/cell-1023.md-->

Each row compares the cell's count of a gene with what the two types predict for it.
*observed* is the count in this cell. *expected* is the count the model predicts for this
cell under the type, after the reference has been
[warped](../how-it-works/warping-the-reference.md) to the scale of the experiment. The
gene favours the type whose expected count agrees better with the observed count. *mean*
is the average count of the gene in the cells currently assigned to the type, and shows
whether this cell is typical of them.

Tac1 decides this cell. The cell has 26.9 Tac1 spots. `Pvalb.Tac1.Sst` predicts 1.6 and
`Sst.Erbb4.Rgs10` predicts 0.8. Both predict too few, but `Pvalb.Tac1.Sst` predicts
more, and the evidence from a gene grows with the number of its spots. With 26.9 spots
Tac1 contributes 11.8, the largest bar in the top-left panel. The cells assigned to
`Pvalb.Tac1.Sst` have 9.6 Tac1 spots on average, those assigned to `Sst.Erbb4.Rgs10`
0.6.

::: details How the 11.8 is computed
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

The first term grows with the count, the second does not. For Tac1, with the expected
counts from the table:

| type | $\mu$ | $\mu/(r+\mu)$ | $\log$ |
| --- | --- | --- | --- |
| `Pvalb.Tac1.Sst` | 1.585 | 0.4421 | -0.8162 |
| `Sst.Erbb4.Rgs10` | 0.777 | 0.2798 | -1.2737 |

Each spot adds $-0.8162 - (-1.2737) = 0.4575$ in favour of `Pvalb.Tac1.Sst`. The second
term is $2 \log(2.777/3.585) = -0.511$ and favours `Sst.Erbb4.Rgs10`, which predicts
fewer spots overall. With $x = 26.895$:

$$
\Delta = 26.895 \times 0.4575 - 0.511 = 11.79
$$

With a single Tac1 spot, $\Delta = -0.05$, and the gene would carry almost no evidence.
:::

## Example 2: the neighbours decide

```python
obj.check_cell(430, 'PC.Other2')
```

![check_cell output for cell 430](/explaining-the-calls/cell-430.png)

Cell 430 has 39 spots and is assigned `PC.Other1` with probability 0.999. It is compared
with `PC.Other2`.

- **Top panels.** The genes are split. Pcp4 favours `PC.Other1`, Crym and Neurod6 favour
  `PC.Other2`. Over all genes, those favouring `PC.Other1` sum to 3.8 and those favouring
  `PC.Other2` to -5.8. The top panels show only the ten largest on each side.
- **Bottom left.** The gene log-likelihood favours `PC.Other2`, -65.5 against -67.4. The
  priors are equal. The MRF term is 9.0 for `PC.Other1` and 0 for `PC.Other2`. With
  `mrf_beta = 1` and `nNeighbors = 9`, 9.0 is the largest value the term can take: all
  neighbours are `PC.Other1`. The totals are -63.4 against -70.4.
- **Bottom right.** The posterior is 99.9% on `PC.Other1`.

The neighbours override a small preference of the genes for the other type. This is the
purpose of the spatial term, but for closely related types that occur in the same
region the neighbours carry little information. Such types can be grouped with
[`mrf_pooled_classes`](../the-model/cell-class.md#pooling-sister-classes), so that the
spatial term supports them equally and the genes decide between them.

## Example 3: a near-empty cell

```python
obj.check_cell(2979, 'Zero')
```

![check_cell output for cell 2979](/explaining-the-calls/cell-2979.png)

Cell 2979 has 0.6 spots and is assigned `Oligo.4` with probability 0.60. It is compared
with Zero, the type that expects no expression.

- **Top panels.** One gene, Cplx2, favours `Oligo.4`, by 0.03. Plp1 favours Zero by 2.8:
  `Oligo.4` expects 6.6 Plp1 spots in this cell and there are none. The genes that
  favour Zero sum to -4.1.
- **Bottom left.** The gene log-likelihood favours Zero, -10.8 against -14.9. So does the
  prior, -0.7 against -5.0, because `cell_type_weights` gives Zero a weight of 0.5 and
  the remaining 0.5 is shared by the 71 real types. The MRF term is 8.9 for `Oligo.4`
  and 0.04 for Zero. The totals are -11.0 against -11.4.
- **Bottom right.** The posterior is 60.2% `Oligo.4` and 37.3% Zero.

The cell has almost no counts, so the gene evidence is weak and the neighbours move it
onto the type around it. A call like this rests on the neighbours alone, and the
posterior shows it is uncertain. The [Zero boost](../the-model/cell-class.md#the-zero-boost)
(`zero_boost`, off by default) gives Zero a spatial bonus that decreases with the
number of spots in the cell, so that empty cells remain Zero.

## Notes

- `my_label` is the cell label in the input segmentation. If pciSeq renumbered the labels
  internally, `check_cell` maps them back.
- The second type must differ from the assigned type.
- The figure is drawn when `show_plot` is `True`. The function also returns the table,
  the per-gene log-likelihoods of both types and the figure, see the
  [API reference](../api/reference.md#check-cell).
- `check_cell` reads `cells.nb_contr` and `cells.mrf` from the fitted model. A model
  saved by an earlier version of pciSeq may not contain them.
