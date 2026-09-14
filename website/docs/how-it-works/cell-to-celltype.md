
# 3. Assigning cells to cell types

Every cell is scored against every cell type by how well its gene counts agree with
the type's warped expected expression, and the scores are converted to probabilities.
The inputs are the gene counts per cell from the current spot assignment and the warped
definitions from the previous step.

## The expression score

The score of a cell for a type is the log probability of the cell's counts under that
type's expected expression, summed over genes. A type that predicts genes the cell lacks,
or omits genes the cell has, scores low. A few strong marker genes can dominate the sum,
which is what separates closely related types.

## The softmax

A softmax over the scores gives each cell a distribution over the types, for example
0.80 on one type, 0.15 on a close relative and 0.05 elsewhere. A confident assignment
concentrates the mass on one type, an ambiguous one spreads it.

## The class prior and the spatial term

Two further terms are added to the score.

- **The class prior.** With `cell_type_prior` set to `uniform`, the default, every real
  type has the same prior weight. With `weighted` the weights are estimated from the
  data with a Dirichlet update, so common types are favoured over rare ones.

- **The spatial term (the MRF).** Cells of the same type cluster in space. Each type
  receives a bonus proportional to the weighted number of the cell's neighbours that
  carry it, with closer neighbours weighted more, on a scale set per cell from its own
  neighbour distances. The term is added to the score, so strong gene evidence can
  outweigh it, and it is bounded. Its strength is `mrf_beta`; `0` switches it off. The
  weights are derived in
  [the model](../the-model/cell-class.md#weighting-the-neighbours-by-distance).
  Sister types can be pooled so the neighbours support all of them equally and the
  expression alone decides between them, see
  [pooling sister classes](../the-model/cell-class.md#pooling-sister-classes).

## The Zero class

One class, Zero, expects no expression. It takes the cells that are effectively empty:
debris, segmentation fragments, and cells whose markers are not in the gene panel.
Without it such cells would be forced onto a real type. A near-empty cell has no counts
to weigh against its neighbours, so the spatial term alone can pull it onto the type
around it; the [Zero boost](../the-model/cell-class.md#the-zero-boost) (`zero_boost`,
off by default) counters that.

The output, a distribution over types for every cell, is the input to
[spot assignment](spots-to-cells.md).
