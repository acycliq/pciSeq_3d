# pciSeq

pciSeq (probabilistic cell typing by in situ sequencing) assigns each RNA spot of an in
situ sequencing experiment to a cell, and each cell to a cell type. The two assignments
are estimated jointly: the type of a cell depends on the spots inside it, and the cell of
a spot depends on the types of the cells around it. Both are returned as probabilities.

## Inputs

- **Spots.** A table of detected RNA reads with gene identity and position (`x`, `y`, and
  `z_plane` for 3D data).
- **Segmentation.** A label image giving the cell each pixel belongs to, typically from a
  DAPI nuclear stain.
- **Cell type definitions.** Mean expression per gene for each cell type, from a separate
  scRNA-seq experiment. These are the reference profiles the cells are scored against.

## Outputs

- For every cell, a probability distribution over the cell types.
- For every spot, a probability distribution over its candidate parent cells and the
  background.

## Probabilistic output

Every assignment is a probability, not a label. Segmentation boundaries are imprecise,
detection is imperfect and a fraction of the reads are noise, so a single answer would
discard the uncertainty. A spot inside one cell whose gene the cell's type expresses
gets a probability near one; a spot on a boundary is split between the cells; a spot
matching no cell goes to the background. Cell types are treated the same way.

## Usage

[Install](installation.md), then see [Running pciSeq](running-pciseq.md). The
algorithm is described in [How it works](how-it-works/overview.md).
