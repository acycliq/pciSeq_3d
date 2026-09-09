
# 1. Estimating the misread density

Not every detected spot corresponds to a genuine transcript. Some reads arise from
technical artefacts, optical crosstalk, or decoding errors. The misread density
quantifies this background, so that each spot can be weighed against the possibility
that it is noise rather than signal.

For each gene, estimate the rate at which its spots are produced by background noise
spread uniformly across the tissue, and use that rate as the baseline a genuine spot
must exceed.

## The background process

The background is modelled as a spatially uniform process: noise spots of a given gene
occur at a constant rate everywhere in the section, independent of location. A gene with
a high background rate therefore produces spurious spots throughout the tissue, including
regions far from any cell, whereas a gene with a low rate does so rarely.

When a spot is assigned later, in [spot assignment](spots-to-cells.md), this rate sets the score
of the background option. The background competes with the candidate cells as a constant,
location-independent alternative: a spot is assigned to a cell only if that cell explains
it better than the background would.

## Per gene, not one global rate

The original pciSeq used a single background rate for all genes. This version estimates a
separate rate for each gene, since genes differ in how noisy they are. With a per-gene
rate, spots of a noisy gene must exceed a higher background level, while spots of a clean
gene need only exceed a lower one.

The rate is updated on every iteration from the spots currently attributed to the
background:

$$
\text{background rate of gene } g \;\approx\;
\frac{\text{background spots of gene } g}{\text{extent of the ROI}}
$$

This ratio of an observed count to the extent over which it is spread is the first
instance of a form that recurs throughout pciSeq: an estimate expressed as an observed
quantity divided by an expected one. The same structure underlies the scaling factors in
the next step.

Formally, with a conjugate Gamma prior the background rate has a Gamma posterior whose
mean is exactly this ratio (regularised by the prior):

$$
q(\rho_g) = \mathrm{Gamma}\Big(\rho_g;\; r_\rho + \bar{N}_{0,g},\;\; \tfrac{r_\rho}{\rho_0} + A_{\text{total}}\Big),
$$

where $\bar{N}_{0,g}$ is the expected number of background spots of gene $g$,
$A_{\text{total}}$ is the extent of the region of interest, $\rho_0$ is the prior mean
misread density, and $r_\rho$ sets how strongly that prior is held.

In 2D that extent is an area. In 3D it is a **volume**, and it has to be measured in the
same units as the cell shapes it is compared against, which means correcting for
anisotropic voxels. Getting this wrong makes the background too strong and pushes spots
that belong to cells into the background instead. The
[full derivation](../the-model/misread-density.md) covers both the correction and the
role of $r_\rho$.

It reads the spots attributed to the background on the previous iteration and
produces a per-gene background rate. [Spot assignment](spots-to-cells.md) uses that rate as the
background option each spot is compared against.
