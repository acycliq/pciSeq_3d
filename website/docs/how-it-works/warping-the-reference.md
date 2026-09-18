
# 2. Warping the cell type definitions

The cell type definitions give the mean expression of each gene in each type. The
observed counts are on a different scale: detection efficiency differs between genes,
transcript yield differs between cells, and the two datasets differ in overall scale.
pciSeq rescales the expected expression to the scale of the current experiment before
comparing. The rescaling is the warp.

The step reads the current gene counts per cell, the current cell type estimates and the
raw cell type definitions, and produces a warped expected expression rescaled at every
level. [Cell typing](cell-to-celltype.md) scores cells against types using it.

The correction factors are nuisance parameters. They are estimated because the cell
types and spot assignments depend on them, they are not in the output, and nothing
observed corresponds to them directly. They are identified through the agreement they
produce between cells and their assigned types.

## The scaling factors

The warp is a family of four factors, each rescaling the expected expression at a
different level of detail. pciSeq calls them inefficiencies. From the broadest to the
most specific:

- **Inefficiency.** One constant for the whole reference, the `Inefficiency` setting.
  It rescales every gene in every cell type definition to the overall scale of the
  observed counts. It can be below or above 1.

- **eta** ($\eta_g$). One factor per gene, shared by all cells: the gene's detection
  efficiency relative to the constant above.

- **theta** ($\theta_{c\mid k}$). One factor per cell and candidate type: the cell's
  total yield relative to what the type predicts, applied to all its genes.

- **gamma** ($\gamma_{g,c\mid k}$). One factor per gene, cell and candidate type: the
  residual mismatch of a given gene in a given cell that the broader factors leave.

Inefficiency and eta do not depend on the cell's type. theta and gamma do, since the
expectation they correct is class-specific, and they are computed for every candidate
type. That is what lets [cell typing](cell-to-celltype.md) use them while scoring a cell
against every type.

## Granularity of the factors

Ordered by how much of the experiment each covers: Inefficiency applies to everything,
gamma to a given gene in a given cell under a given class.

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
<figcaption>The four factors by granularity.</figcaption>
</figure>

Four factors are needed because the mismatch occurs at all four levels at once: a
global scale difference between the two datasets, a per-gene detection pattern, per-cell
variation in yield, and gene-by-cell noise. Each factor absorbs the mismatch at its own
scale.

The factors are applied together. For each (cell, gene, class) triplet the expected
expression is the reference value times Inefficiency, eta, theta and gamma.

The [demo](scale-factors-demo.md) puts three of them on sliders over five cells generated
from known class definitions.

## Inefficiencies

Every inefficiency is the same statistic, a ratio of observed over expected:

$$
\text{factor} \;=\;
\frac{\text{observed}}{\text{expected}}
$$

- **gamma** compares the observed count of a *given gene in a given cell* with its
  expected count, under a candidate type.
- **theta** compares the observed total count of a *given cell* with its expected
  total, under a candidate type.
- **eta** compares the observed count of a *given gene across all cells* with its
  expected total.

A factor is greater than 1 when the observed count exceeds the expected count and less
than 1 otherwise. The factors differ only in the level of aggregation before the ratio
is formed.

## The priors

Each factor has a prior with mean 1, no rescaling, whose strength is a hyperparameter:
`rGene` for eta, `rTheta` for theta, `rSpot` for gamma. How far a factor moves from 1
depends on that hyperparameter against the number of counts available to estimate it,
and those counts differ by orders of magnitude between the three levels.

- **eta** is estimated from one gene's spots across the whole section, typically thousands. At
  the default `rGene` of 20 the prior is negligible and the data determine the final (that is,
  posterior) eta.
- **theta** is estimated from one cell's spots, typically tens. At the default `rTheta` of 25 the
  prior is comparable to the data. Lower it to around 2 and the data determine the posterior
  theta. It has to stay above 1: theta is
  $(\text{spots} + r_\theta - 1) / (r_\theta + \text{expected})$, so at 1 or below a cell with no
  spots gives a theta of zero or less.
- **gamma** is estimated from a given gene in a given cell, usually a fraction of a spot. The default
  `rSpot` of 2 therefore dominates, which is the intent: there is too little data at that level
  to estimate anything on its own. `rSpot` is also the dispersion of the negative binomial, since
  integrating gamma out is what produces it.

Raising a hyperparameter holds its factor near 1; lowering it lets the data set it. A
starting value for `rTheta` is the typical number of counts in a cell.

## The spatial factor (the MRF)

Cells of the same type cluster in space, in layers or regions. When a cell is scored
against the types, each type receives a bonus proportional to the weighted number of
the cell's neighbours that carry it, closer neighbours weighted more. This is the
spatial term, a Markov random field (MRF), where a cell's label depends on its
neighbours' labels. `mrf_beta` sets its strength; at `0` only the gene counts and the
prior decide the type. It is described with the cell typing step in
[cell to cell type](cell-to-celltype.md#the-class-prior-and-the-spatial-term).

A cell with almost no spots has nothing to weigh against its neighbours, so the spatial
term alone can decide its type. The
[Zero boost](../the-model/cell-class.md#the-zero-boost), off by default, counters that.

<details>
<summary>Earlier working, retained for the record and due for removal</summary>

## The mrf cap

::: warning Removed from the model
The cap is not implemented in the current code. It did not converge, and two subsequent
modifications behaved the same way. The section is retained because it describes the problem
the Zero class presents, which remains. See
[Why the cap does not converge](../the-model/cell-class.md#why-the-cap-does-not-converge).
What is in the code, off by default, is the
[Zero boost](../the-model/cell-class.md#the-zero-boost).
:::

The mrf cap was a limit on the spatial bonus, set separately for each cell and each candidate
type. It stops the neighbours from adding so much that they overturn what a cell's own gene
counts and prior already say.

We added it because of empty cells. A near-empty cell has almost no gene counts of its own,
so the spatial term dominates and the cell simply takes on the type of its neighbourhood.
Worse, once it does, it becomes a neighbour that pushes the same type onto the next empty
cell, so a single type can spread across a whole patch of background like a relay. The cap
is there to stop that.

### How it protects the Zero class

For every cell and every candidate type the cap works out how strong the neighbour bonus
would need to be to overtake Zero, and never lets the bonus grow that large, so long as the
cell's own counts and the prior already point to Zero. Under the cap the Zero class gets no
neighbour bonus of its own, so the whole job of the cap is to hold the other types back rather
than to prop Zero up. The [Zero boost](../the-model/cell-class.md#the-zero-boost) that replaced
it works the other way round, giving Zero a bonus and leaving the real classes alone.

The result is a soft preference that cannot overrule the evidence. If a cell's own counts say
it is empty, no amount of like-typed neighbours can flip it off Zero. If instead the counts
genuinely favour a real type, there is nothing to protect, the cap does not act, and the
neighbour bonus works at full strength. The formula is derived on the
[cell-class assignment](../the-model/cell-class.md#the-mrf-cap) page.

### Example: an empty cell

Cell 16150 has zero gene counts. With `rTheta = 2` and a uniform class
prior, the posterior of its cell-to-class assignment is almost uniform: Zero at 3.5%, all
other classes with prob around 3.1% or 3.2%.

<figure class="diagram">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 356" role="img" aria-label="Cell-to-class posterior for empty cell 16150, top 12 classes">
  <text x="242" y="32.0" text-anchor="end" class="barc-name">Zero</text>
  <rect x="262" y="19.0" width="430.0" height="17" rx="2.5" fill="#000000" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="699.0" y="32.0" class="barc-pct">3.6%</text>
  <text x="242" y="59.0" text-anchor="end" class="barc-name">036 HPF CR Glut</text>
  <rect x="262" y="46.0" width="389.1" height="17" rx="2.5" fill="#AA3377" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="658.1" y="59.0" class="barc-pct">3.2%</text>
  <text x="242" y="86.0" text-anchor="end" class="barc-name">331 Peri NN</text>
  <rect x="262" y="73.0" width="389.1" height="17" rx="2.5" fill="#AA3377" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="658.1" y="86.0" class="barc-pct">3.2%</text>
  <text x="242" y="113.0" text-anchor="end" class="barc-name">045 OB-STR-CTX Inh IMN</text>
  <rect x="262" y="100.0" width="389.1" height="17" rx="2.5" fill="#4477AA" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="658.1" y="113.0" class="barc-pct">3.2%</text>
  <text x="242" y="140.0" text-anchor="end" class="barc-name">338 Lymphoid NN</text>
  <rect x="262" y="127.0" width="389.1" height="17" rx="2.5" fill="#DD44AA" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="658.1" y="140.0" class="barc-pct">3.2%</text>
  <text x="242" y="167.0" text-anchor="end" class="barc-name">337 DC NN</text>
  <rect x="262" y="154.0" width="389.1" height="17" rx="2.5" fill="#FF9955" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="658.1" y="167.0" class="barc-pct">3.2%</text>
  <text x="242" y="194.0" text-anchor="end" class="barc-name">336 Monocytes NN</text>
  <rect x="262" y="181.0" width="389.1" height="17" rx="2.5" fill="#99DD55" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="658.1" y="194.0" class="barc-pct">3.2%</text>
  <text x="242" y="221.0" text-anchor="end" class="barc-name">038 DG-PIR Ex IMN</text>
  <rect x="262" y="208.0" width="385.9" height="17" rx="2.5" fill="#CC3333" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="654.9" y="221.0" class="barc-pct">3.2%</text>
  <text x="242" y="248.0" text-anchor="end" class="barc-name">326 OPC NN</text>
  <rect x="262" y="235.0" width="374.8" height="17" rx="2.5" fill="#CC5500" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="643.8" y="248.0" class="barc-pct">3.1%</text>
  <text x="242" y="275.0" text-anchor="end" class="barc-name">332 SMC NN</text>
  <rect x="262" y="262.0" width="373.5" height="17" rx="2.5" fill="#BBBBBB" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="642.5" y="275.0" class="barc-pct">3.1%</text>
  <text x="242" y="302.0" text-anchor="end" class="barc-name">319 Astro-TE NN</text>
  <rect x="262" y="289.0" width="351.4" height="17" rx="2.5" fill="#BF94E4" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="620.4" y="302.0" class="barc-pct">2.9%</text>
  <text x="242" y="329.0" text-anchor="end" class="barc-name">333 Endo NN</text>
  <rect x="262" y="316.0" width="342.4" height="17" rx="2.5" fill="#A07038" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="611.4" y="329.0" class="barc-pct">2.8%</text>
</svg>
<figcaption>Empty cell 16150 at <code>rTheta = 2</code>: the cell-to-class posterior is almost uniform (top 12 of 39 classes; bar length is relative to the top class).</figcaption>
</figure>

The near-uniform **cell-to-class** posterior (the distribution over which type the cell is,
not the posterior of any single scale factor) comes from theta. The cell has no counts, so
theta collapses for every class, scaling each class's predicted expression down to the floor.
The gene likelihood is then the same across classes, and the uniform prior adds no separation,
so the cell-to-class posterior is almost uniform. Zero ends up just ahead because of the cap:
it holds every real class a fixed distance below Zero, so the classes pressing hardest against
it, the ones with enough like-typed neighbours, land right underneath at 3.2% against Zero's
3.5%. They cannot get any closer.

At `rTheta = 2` the cap does its job and Zero wins, but only just: the cell-to-class posterior
is almost uniform. How concentrated that posterior becomes on Zero is set by `rTheta`.

### The same cell with a stronger Zero prior

Raise `rTheta` from 2 to 20 and theta can no longer collapse as far. Theta is fit per cell
**and** per class, so even for this one empty cell it takes a different value for each class:
at `rTheta = 20` those values land in the 0.2 to 0.6 range, against the near-zero they reached
at `rTheta = 2`. Here is the same cell.

<figure class="diagram">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 356" role="img" aria-label="Cell-to-class posterior for empty cell 16150, top 12 classes">
  <text x="242" y="32.0" text-anchor="end" class="barc-name">Zero</text>
  <rect x="262" y="19.0" width="430.0" height="17" rx="2.5" fill="#000000" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="699.0" y="32.0" class="barc-pct">31.2%</text>
  <text x="242" y="59.0" text-anchor="end" class="barc-name">336 Monocytes NN</text>
  <rect x="262" y="46.0" width="389.1" height="17" rx="2.5" fill="#99DD55" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="658.1" y="59.0" class="barc-pct">28.2%</text>
  <text x="242" y="86.0" text-anchor="end" class="barc-name">338 Lymphoid NN</text>
  <rect x="262" y="73.0" width="389.1" height="17" rx="2.5" fill="#DD44AA" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="658.1" y="86.0" class="barc-pct">28.2%</text>
  <text x="242" y="113.0" text-anchor="end" class="barc-name">045 OB-STR-CTX Inh IMN</text>
  <rect x="262" y="100.0" width="66.9" height="17" rx="2.5" fill="#4477AA" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="335.9" y="113.0" class="barc-pct">4.8%</text>
  <text x="242" y="140.0" text-anchor="end" class="barc-name">337 DC NN</text>
  <rect x="262" y="127.0" width="31.7" height="17" rx="2.5" fill="#FF9955" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="300.7" y="140.0" class="barc-pct">2.3%</text>
  <text x="242" y="167.0" text-anchor="end" class="barc-name">331 Peri NN</text>
  <rect x="262" y="154.0" width="22.7" height="17" rx="2.5" fill="#AA3377" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="291.7" y="167.0" class="barc-pct">1.6%</text>
  <text x="242" y="194.0" text-anchor="end" class="barc-name">036 HPF CR Glut</text>
  <rect x="262" y="181.0" width="13.8" height="17" rx="2.5" fill="#AA3377" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="282.8" y="194.0" class="barc-pct">1.0%</text>
  <text x="242" y="221.0" text-anchor="end" class="barc-name">038 DG-PIR Ex IMN</text>
  <rect x="262" y="208.0" width="8.6" height="17" rx="2.5" fill="#CC3333" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="277.6" y="221.0" class="barc-pct">0.6%</text>
  <text x="242" y="248.0" text-anchor="end" class="barc-name">332 SMC NN</text>
  <rect x="262" y="235.0" width="8.2" height="17" rx="2.5" fill="#BBBBBB" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="277.2" y="248.0" class="barc-pct">0.6%</text>
  <text x="242" y="275.0" text-anchor="end" class="barc-name">326 OPC NN</text>
  <rect x="262" y="262.0" width="5.7" height="17" rx="2.5" fill="#CC5500" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="274.7" y="275.0" class="barc-pct">0.4%</text>
  <text x="242" y="302.0" text-anchor="end" class="barc-name">333 Endo NN</text>
  <rect x="262" y="289.0" width="3.4" height="17" rx="2.5" fill="#A07038" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="272.4" y="302.0" class="barc-pct">0.2%</text>
  <text x="242" y="329.0" text-anchor="end" class="barc-name">319 Astro-TE NN</text>
  <rect x="262" y="316.0" width="2.9" height="17" rx="2.5" fill="#BF94E4" stroke="currentColor" stroke-opacity="0.28" stroke-width="1" />
  <text x="271.9" y="329.0" class="barc-pct">0.2%</text>
</svg>
<figcaption>The same cell at <code>rTheta = 20</code>: the high-expression classes drop out, leaving Zero and the two lowest-expression classes (top 12 of 39).</figcaption>
</figure>

Zero climbs from 3.5% to 31%. Because theta can no longer shrink to near zero, a
high-expression class like Oligo (theta 0.37 here) still predicts far more counts than the
cell's zero, so it is penalised and its probability drops to 0. The lowest-expression classes
on this panel, Monocytes and Lymphoid, predict almost nothing whatever theta does, because
their reference expression is tiny; they stay beside Zero at 28%, and the stronger prior
cannot separate them.

</details>
