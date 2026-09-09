
# 3. Assigning cells to cell types

At this point in the loop two inputs are available: the gene counts within each cell, and
the warped definitions, now on the correct scale for the experiment. The model can then
evaluate the type of each cell.

Score every cell against every known cell type according to how well its gene counts
agree with the type's (warped) expected expression, and convert the scores into
probabilities.

## The expression score

For a given cell and a candidate type, the model evaluates how probable the observed
counts are under that type. A type whose expected expression agrees with the cell's
counts receives a high score; a type that predicts genes the cell lacks, or omits genes
the cell expresses, receives a low one. Each gene contributes a term, and the terms
combine across all genes. A small number of strong marker genes can dominate the result,
which is appropriate: the presence or absence of a few characteristic transcripts is
often what distinguishes closely related types.

## The softmax

The scores are passed through a **softmax**, which converts them into probabilities that
sum to one. For each cell the result is a distribution over the types, for example 0.80
on one type, 0.15 on a close relative, and 0.05 elsewhere. This distribution is itself
informative: a confident assignment concentrates almost all of the probability on a
single type, whereas an ambiguous one spreads it across several.

## The class prior and the spatial term

Beyond the gene-expression match, two further terms enter the score:

- **A prior.** Cell types differ in overall abundance. The prior shifts the scores so
  that, other things being equal, a cell is more readily assigned to a common type than
  to a rare one.

- **A spatial term (the MRF).** Cells of the same type tend to be spatially clustered.
  The model adds a bonus when a cell's neighbours share its type, which discourages
  isolated, biologically implausible assignments. Nearby neighbours count for more than
  distant ones, on a scale set per cell from its own neighbour distances, so "close" means
  the same thing in dense and sparse tissue. The term is a soft preference rather than a
  constraint: it is added to the score, so strong gene evidence can outweigh it. Its size
  is bounded, and how it is built is set out in
  [the model derivation](../the-model/cell-class.md#weighting-the-neighbours-by-distance).

## The Zero class

One class, labelled **Zero**, expects no expression. It absorbs cells that are
effectively empty: debris, poorly segmented fragments, or cells whose markers are absent
from the gene panel. Providing this class prevents such cells from being forced onto a
genuine type to which they do not belong.

It reads the gene counts per cell, the warped definitions, the class prior and the
neighbourhood structure, and produces a probability distribution over cell types for every
cell. [Spot assignment](spots-to-cells.md) uses that distribution to assess which spots a cell is
likely to emit.
