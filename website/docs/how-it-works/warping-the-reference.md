
# 2. Warping the cell type definitions

This is the most subtle of the four, and the hardest to verify.

The cell type definitions give the average expression of each gene in each known cell
type, measured by scRNA-seq. That measurement comes from a different technology to the
one that produced the spots. Detection efficiency varies between genes, transcript
capture varies between cells, and the two experiments differ in overall scale, so
observed counts and raw definitions are not on comparable footing. pciSeq therefore
**warps** the definitions before comparing: it rescales the expected expression to the
scale of the current experiment.

The other three steps produce output that can be inspected: spot-to-cell assignments
overlay on the image and can be judged for spatial plausibility, and an assigned cell
type can be checked against established marker genes. The correction factors admit no
such check. They are **nuisance parameters**, estimated only because the cell types and
spot assignments depend on them, and absent from the output. Nothing observed corresponds to
them, so they are identified indirectly, through the agreement they produce between
cells and their assigned types.

## The scaling factors

The warp is not one number but a **family of correction factors**, each rescaling the
expected expression at a different level of detail. pciSeq calls them *inefficiencies*,
since they mostly describe signal lost relative to the single-cell data.

From the broadest to the most specific:

- **Inefficiency** - a single constant applied to the whole reference, encoding the fact
  that in situ sequencing detects only a fraction (for example, around a fifth) of what
  scRNA-seq reports. It rescales **every gene in every cell** by this one factor, and is
  fully systemic: it does not distinguish between individual genes or cells.

- **eta** ($\eta_g$) - one factor **per gene**, shared across all cells. Some genes are
  detected more efficiently than others; eta captures that. It is the same for every
  cell, but different for every gene.

- **theta** ($\theta_{c\mid k}$) - one factor **per cell, for each candidate cell type**.
  Some cells simply yield more transcripts than the definitions predict, others fewer;
  theta is a single whole-cell **scalar** that stretches or shrinks that cell's expected
  counts across all its genes. It is worked out separately for every type the cell might be,
  because what counts as "expected" depends on the type being tested.

- **gamma** ($\gamma_{g,c\mid k}$) - one factor **per gene, per cell, per candidate type**.
  This is the most fine-grained and idiosyncratic correction: it adjusts a single gene
  in a single cell, and again it is computed separately for each type that cell might be.
  It accounts for the residual mismatch that none of the broader factors can explain.

The four divide into two groups. The two broad factors, **Inefficiency** and **eta**, are
the same no matter what type a cell turns out to be. The two fine ones, **theta** and
**gamma**, are
**conditional on the class**: they are recomputed for each candidate type, because the
expectation they correct against is itself class-specific. This is why
[cell typing](cell-to-celltype.md) can use them while it scores a cell against every type at
once.

## Granularity of the factors

These factors can be arranged as a stack, ordered by how much of the experiment each one
covers. The broad, systemic factor sits at the base, affecting everything at once.
Higher up, the corrections get narrower and more specific, up to gamma at the apex, which
applies to just one gene, in one cell, under one candidate type.

<figure class="diagram">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 460" role="img" aria-label="A pyramid of the four scaling factors">
  <defs></defs>
  <line x1="70" y1="396" x2="70" y2="66" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.5" />
  <path d="M 70,54 L 76,68 L 70,64 L 64,68 Z" fill="currentColor" fill-opacity="0.5" />
  <text class="ip-axis" transform="rotate(-90 90,230)" x="90" y="230" text-anchor="middle">GRANULARITY INCREASES</text>
  <path class="ip-tier" fill="#34d399" d="M 300,60.0 L 342.9,141.0 L 257.1,141.0 Z" />
  <path class="ip-tier" fill="#10b981" d="M 252.9,149.0 L 347.1,149.0 L 387.9,226.0 L 212.1,226.0 Z" />
  <path class="ip-tier" fill="#059669" d="M 207.9,234.0 L 392.1,234.0 L 432.9,311.0 L 167.1,311.0 Z" />
  <path class="ip-tier" fill="#047857" d="M 162.9,319.0 L 437.1,319.0 L 477.9,396.0 L 122.1,396.0 Z" />
  <line class="ip-leader" x1="343" y1="100.5" x2="540" y2="100.5" />
  <line class="ip-leader" x1="368" y1="187.5" x2="540" y2="187.5" />
  <line class="ip-leader" x1="413" y1="272.5" x2="540" y2="272.5" />
  <line class="ip-leader" x1="458" y1="357.5" x2="540" y2="357.5" />
  <g>
    <text x="550" y="96" class="ip-glyph">&#947; <tspan class="ip-name">(gamma)</tspan></text>
    <text x="550" y="116" class="ip-desc">scales gene g's count in cell c, per class k</text>
    <text x="550" y="183" class="ip-glyph">&#952; <tspan class="ip-name">(theta)</tspan></text>
    <text x="550" y="203" class="ip-desc">scales cell c's total count, per class k</text>
    <text x="550" y="268" class="ip-glyph">&#951; <tspan class="ip-name">(eta)</tspan></text>
    <text x="550" y="288" class="ip-desc">scales gene g's count across all cells</text>
    <text x="550" y="353" class="ip-glyph" font-size="20">Inefficiency</text>
    <text x="550" y="373" class="ip-desc">scales the whole experiment at once</text>
  </g>
</svg>
<figcaption>The same idea, four levels of granularity. Wide and systemic at the bottom, narrow and idiosyncratic at the top.</figcaption>
</figure>

Why have all four instead of one? Because mismatch occurs at all of these levels at once.
There is a global scale difference between the two technologies (handled at the base),
on top of that a per-gene detection pattern (eta), on top of that per-cell variation
(theta), and on top of all that, irreducible gene-by-cell noise (gamma). Each factor
absorbs the mismatch at its own scale, and the rest is left to the others.

The factors are described separately but applied together. For each (cell, gene, class)
triplet, Inefficiency, eta, theta and gamma multiply into a single number, and that number
rescales the reference expression for exactly that triplet. The four levels of granularity are
just how that one combined adjustment is built up.

The [demo](scale-factors-demo.md) puts three of them on sliders, over a field of five
cells generated from known class definitions, so the effect of each factor on the call
can be seen directly.

## Inefficiencies

Although each acts at a different level of detail, **every inefficiency is the same
statistic**: a ratio of observed over expected.

$$
\text{factor} \;=\;
\frac{\text{what was actually observed}}{\text{what the model expected}}
$$

- **gamma** compares the observed counts of *one gene in one cell* against what the
  definitions predict for that same gene and cell, *assuming a given type*.
- **theta** compares the observed total counts of *one cell* against the total the
  definitions predict for it, *assuming a given type*.
- **eta** compares the observed counts of *one gene across all cells* against the total
  predicted for that gene.

In each case the factor is the observed quantity divided by the expected one, so it is
**greater than 1 when more was observed than predicted** and **less than 1 when less was
observed**. The only thing that differs between the factors is the level of aggregation
before the ratio is formed: a single gene-cell pair under one type, a whole cell under
one type, or a whole gene across all cells.

It therefore reduces to a single principle applied at different scales: **observed
over expected.**

## The priors

Each factor has a prior centred on 1, meaning no rescaling, and is regulated by a hyperparameter
that determines how firmly it is held there: `rGene` for eta, `rTheta` for theta, `rSpot` for
gamma. Whether a factor actually moves depends on that hyperparameter against the number of
counts available to estimate it, and those counts differ by orders of magnitude between the three
levels.

- **eta** is estimated from one gene's reads across the whole section, typically thousands. At
  the default `rGene` of 20 the prior is negligible and the data determine the final (that is,
  posterior) eta.
- **theta** is estimated from one cell's reads, typically tens. At the default `rTheta` of 25 the
  prior is comparable to the data. Lower it to around 2 and the data determine the posterior
  theta. It has to stay above 1: theta is
  $(\text{reads} + r_\theta - 1) / (r_\theta + \text{expected})$, so at 1 or below a cell with no
  reads gives a theta of zero or less.
- **gamma** is estimated from one gene in one cell, usually a fraction of a read. The default
  `rSpot` of 2 therefore dominates, which is the intent: there is too little data at that level
  to estimate anything on its own. `rSpot` is also the dispersion of the negative binomial, since
  integrating gamma out is what produces it.

Raise any of them and that factor stays near 1, so that level stops rescaling the definitions.
Lower it and the data determine the posterior. As a rule of thumb for `rTheta`, start at roughly
the typical number of counts a cell has.

## The spatial factor (the MRF)

Cells of the same type are often strongly localised, arranged in layers or clear regions.
The model uses this when it scores a cell against the candidate types: it adds a bonus to
any type the cell's neighbours already belong to, so a cell surrounded by one type is more
likely to be called that type itself.

This is the spatial factor, or MRF (short for Markov random field, a model where each cell's
label depends on its neighbours' labels). The `mrf_beta` hyperparameter sets how strong it
is: the bonus for a type adds up the neighbours that favour it, with closer neighbours
counting more. A large `mrf_beta` weighs the neighbours heavily; at `mrf_beta = 0` the factor
is off and only the gene counts and the prior decide the type.

A cell with almost no reads of its own has nothing to weigh against its neighbours, so the
spatial factor alone can decide what it is. The
[Zero boost](../the-model/cell-class.md#the-zero-boost), off by default, protects such cells
from being taken over by the type around them.

It reads the current gene counts per cell, the current cell-type estimates and the
raw cell type definitions, and produces a warped expected expression rescaled at every
level. [Cell typing](cell-to-celltype.md) scores cells against types using it.
