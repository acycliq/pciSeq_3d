# How it works

pciSeq treats every unknown as a latent variable of one Bayesian model: the cell of
origin of each spot, the class of each cell, the misread density and the detection
efficiency of each gene, the per-cell and per-cell-gene scale factors, and the position and
shape of each cell. With the optional Unclassified class, the mean expression of that
class is one more. Each has a prior. The target of inference is
their joint posterior given the spots and the cell type definitions.

The posterior has no closed form. It is approximated by variational inference: the
approximating distribution is restricted to a mean-field family, one factor per group of
latent variables, and the member of the family closest to the posterior in
Kullback-Leibler divergence is selected. This is equivalent to maximising a lower bound
on the model evidence.

The optimal factors depend on one another, so they are fitted by coordinate ascent. Each
factor is updated to its optimum given the current values of the others, and the sweep
repeats until the estimates stop changing. The steps below are these updates.

## The variational loop

<figure class="diagram">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 500" role="img" aria-label="The pciSeq variational loop: misread density, warping the definitions, cell typing, the class weights when cell_type_prior is weighted, the Unclassified class when unclassified_class is on, spot assignment and cell geometry, in that order">
  <defs>
    <path id="vlTxt0" d="M 413.3,60.5 A 190,190 0 0,1 537.5,118.9" />
    <path id="vlTxt1" d="M 589.9,312.8 A 200,200 0 0,0 568.4,142.1" />
    <path id="vlTxt2" d="M 503.4,421.2 A 200,200 0 0,0 574.4,347.9" />
    <path id="vlTxt3" d="M 349.1,443.4 A 200,200 0 0,0 468.9,437.8" />
    <path id="vlTxt4" d="M 207.7,305.1 A 200,200 0 0,0 313.1,430.1" />
    <path id="vlTxt5" d="M 236.7,134.5 A 200,200 0 0,0 200.8,267.4" />
    <path id="vlTxt6" d="M 268.7,112.7 A 190,190 0 0,1 376.8,61.4" />
  </defs>
  <text x="400" y="255" text-anchor="middle" class="vl-center-title">THE VARIATIONAL LOOP</text>
  <circle cx="400" cy="250" r="126" fill="none" stroke="currentColor" stroke-width="0.5" stroke-dasharray="2 6" opacity="0.2" />
  <g>
    <path class="vl-arc-band" d="M 411.5,30.3 A 220,220 0 0,1 551.0,90.0 L 560.7,79.9 L 556.5,133.6 L 507.1,136.6 L 516.7,126.4 A 170,170 0 0,0 408.9,80.2 Z" />
    <text class="vl-text-label"><textPath href="#vlTxt0" startOffset="50%" text-anchor="middle">Misread density</textPath></text>
    <path class="vl-arc-band" d="M 583.2,128.1 A 220,220 0 0,1 612.2,308.0 L 625.7,311.7 L 576.3,333.3 L 550.5,291.2 L 564.0,294.8 A 170,170 0 0,0 541.5,155.8 Z" />
    <text class="vl-text-label"><textPath href="#vlTxt1" startOffset="50%" text-anchor="middle">Warping definitions</textPath></text>
    <path class="vl-arc-band" d="M 593.7,354.3 A 220,220 0 0,1 523.5,432.1 L 531.3,443.7 L 479.8,427.9 L 487.6,379.1 L 495.4,390.7 A 170,170 0 0,0 549.7,330.6 Z" />
    <text class="vl-text-label"><textPath href="#vlTxt2" startOffset="50%" text-anchor="middle">Cell typing</textPath></text>
    <path class="vl-arc-band vl-arc-optional" d="M 479.4,455.2 A 220,220 0 0,1 355.2,465.4 L 352.3,479.1 L 327.7,431.1 L 368.2,402.7 L 365.4,416.4 A 170,170 0 0,0 461.3,408.6 Z" />
    <text class="vl-text-label vl-text-optional"><textPath href="#vlTxt3" startOffset="50%" text-anchor="middle">Class weights</textPath></text>
    <path class="vl-arc-band vl-arc-optional" d="M 307.9,449.8 A 220,220 0 0,1 192.0,321.5 L 178.7,326.1 L 207.4,280.4 L 252.5,300.7 L 239.2,305.3 A 170,170 0 0,0 328.8,404.4 Z" />
    <text class="vl-text-label vl-text-optional"><textPath href="#vlTxt4" startOffset="50%" text-anchor="middle">Unclassified class</textPath></text>
    <path class="vl-arc-band" d="M 181.2,272.9 A 220,220 0 0,1 214.0,132.5 L 202.1,125.1 L 255.7,118.8 L 268.1,166.7 L 256.3,159.2 A 170,170 0 0,0 230.9,267.7 Z" />
    <text class="vl-text-label"><textPath href="#vlTxt5" startOffset="50%" text-anchor="middle">Spot assignment</textPath></text>
    <path class="vl-arc-band" d="M 245.2,93.7 A 220,220 0 0,1 361.8,33.3 L 359.4,19.6 L 400.0,55.0 L 372.9,96.4 L 370.5,82.6 A 170,170 0 0,0 280.4,129.2 Z" />
    <text class="vl-text-label"><textPath href="#vlTxt6" startOffset="50%" text-anchor="middle">Cell geometry</textPath></text>
  </g>
</svg>
<figcaption>The updates, in the order they run. The two dashed steps are optional: the class weights are updated only with <code>cell_type_prior='weighted'</code>, and the Unclassified class only with <code>unclassified_class=True</code>. Each sweep starts from the output of the previous one.</figcaption>
</figure>

1. **[Misread density.](misread-density.md)** The rate of background spots per gene,
   estimated from the spots currently assigned to the background.

2. **[Warping the cell type definitions.](warping-the-reference.md)** Per-gene detection
   efficiencies and per-cell and per-cell-gene scale factors that rescale the cell type
   definitions to this experiment. These are latent, with no observed counterpart, and
   are identified only through the fit between cells and classes.

3. **[Cell to class.](cell-to-celltype.md)** Every cell is scored against every class
   with the warped definitions, and the scores are normalised to probabilities.

4. **[Class weights.](../the-model/class-prior.md)** Optional. With
   `cell_type_prior='weighted'` the class weights, the prior share of each class, are
   updated from the class probabilities of the cells. With the default, `'uniform'`, they
   stay fixed at their initial values, set by `cell_type_weights`, and this step is
   skipped.

5. **[Unclassified class.](../the-model/unclassified-class.md)** Optional. With
   `unclassified_class=True` there is one more class, with no entry in the cell type
   definitions, and its mean expression is re-estimated from the cells currently assigned
   to it. With the default, `False`, there is no such class and this step is skipped.

6. **[Spots to cells.](spots-to-cells.md)** Every spot is assigned to one of its
   neighbouring cells or to the background, as a probability, given the cell class
   probabilities.

7. **[Cell geometry.](../the-model/gaussian.md)** The centre and the covariance of each
   cell's Gaussian are updated from the spots now assigned to it. This comes after the
   spot assignment so that it uses the fitted spot probabilities rather than the
   uniform initial ones.

The new spot assignments change the background counts and the gene counts per cell, which
enter steps 1 to 5 of the next sweep, and the new geometry enters the next spot assignment. A single pass would leave each step conditioned on the initial values of the
others; iterating propagates each update to the rest.

## Convergence

After each sweep the change in the spot-to-cell probabilities is measured as the maximum
absolute change over all spots, their candidate cells and the background. The loop stops when this falls below
`CellCallTolerance`, or at `max_iter`. The maximum is a strict criterion: one spot whose
probabilities still change keeps the loop running after every other spot has stopped changing,
where a mean would have stopped earlier.
