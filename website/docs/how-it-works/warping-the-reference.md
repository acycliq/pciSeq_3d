
# Block 2: Warping the cell type definitions

This block is the most subtle of the four, and the hardest to verify.

The cell type definitions are a lookup table: for every known cell type, they list the
average expression of every gene, measured in a **separate** scRNA-seq experiment. The
problem is that the in situ experiment in front of you does not behave exactly like that
separate experiment. Genes are detected at different efficiencies, some cells capture
more transcripts than others, and the overall scale is different. If you compared your
cells against the raw definitions, the match would be off for reasons that have nothing
to do with biology.

So before comparing, pciSeq **warps** the definitions to fit this experiment. It rescales
the expected expression numbers until they are on the same scale as what you actually
observe.

This calibration is also part of how the model separates signal from noise. Once the
technical losses are absorbed by these factors, genuine reads align with the adjusted
expected expression, while reads that match no calibrated cell type are left to be
accounted for as background. Putting the definitions on the right scale is therefore not
only about comparability; it is also what lets real expression be told apart from
technical noise.

## Sources of difficulty

The outputs of the other blocks can be inspected directly. Spot-to-cell assignments can
be overlaid on the image and assessed for spatial plausibility, and a cell's assigned
type can be compared against the expression of established marker genes. The adjustments
made in this block admit no comparable check. The inefficiency factors are **nuisance
parameters**: quantities the model must estimate in order to reach the results of
interest (the cell types and spot assignments), but which are not themselves reported.
They are latent, never observed, and there is no ground truth against which to validate
them. They are identified only indirectly, through the improvement they
produce in the agreement between cells and their assigned types. Their influence is
evident in the final result, but the adjustments themselves are not, which makes this
block intrinsically harder to validate than the assignment steps.

## The scaling factors

The warp is not one number. It is a **family of correction factors**, each one rescaling
the expected expression at a different level of detail. pciSeq calls them
*inefficiencies*, because they mostly describe how much signal is lost relative to the
single-cell data.

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
  because what counts as "expected" depends on which type you are testing it against.

- **gamma** ($\gamma_{g,c\mid k}$) - one factor **per gene, per cell, per candidate type**.
  This is the most fine-grained and idiosyncratic correction: it adjusts a single gene
  in a single cell, and again it is computed separately for each type that cell might be.
  It accounts for the residual mismatch that none of the broader factors can explain.

The four divide into two groups. The two broad factors, **Inefficiency** and **eta**, are
the same no matter what type a cell turns out to be. The two fine ones, **theta** and
**gamma**, are
**conditional on the class**: they are recomputed for each candidate type, because the
expectation they correct against is itself class-specific. This is why
[block 3](cell-to-celltype.md) can use them while it scores a cell against every type at
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

The block therefore reduces to a single principle applied at different scales: **observed
over expected.**

## Theta

Theta is one of the scaling factors: it rescales the single-cell reference for a whole cell
at once.

**Empty or near-empty cells.** Since theta is observed over expected, it can **potentially**
scale the reference data down by a large amount to match a cell's few counts. A real class
scaled down that far predicts almost nothing, so it can compete with the Zero class for the
cell.

For this to happen the hyperparameter `rTheta` has to be close to its lowest value. `rTheta`
is the strength of theta's prior: a value close to $1.0$ makes the prior very weak and lets
the data drive theta freely. With a weak prior, theta on a near-empty cell collapses for
every class alike, and the cell-to-class posterior comes out close to uniform.

## The spatial factor (the MRF)

You might be working with tissue where cells of the same type are strongly localised,
arranged in layers or clear regions. The model uses this when it scores a cell against the
candidate types: it adds a bonus to any type that the cell's neighbours already belong to,
so a cell surrounded by one type is a little more likely to be called that type itself.

This is the spatial factor, or MRF (short for Markov random field, a model where each cell's
label depends on its neighbours' labels). The `mrf_beta` hyperparameter sets how strong it
is: the bonus for a type adds up the neighbours that favour it, with closer neighbours
counting more. A large `mrf_beta` weighs the neighbours heavily; at `mrf_beta = 0` the factor
is off and only the gene counts and the prior decide the type.

## The mrf cap

::: warning Removed from the model
The cap is not implemented in the current code. It did not converge, and two subsequent
modifications behaved the same way. The section is retained because it describes the problem
the Zero class presents, which remains. See
[Why the cap does not converge](../the-model/cell-class.md#why-the-cap-does-not-converge).
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
cell's own counts and the prior already point to Zero. The Zero class gets no neighbour bonus
of its own, so the whole job of the cap is to hold the other types back rather than to prop
Zero up.

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

### The breadth of theta

Theta was not introduced to handle empty or near-empty cells alone; its scope is wider. It lets the model type cells whose total counts sit far from
what the reference predicts, rescaling the class expectations up or down until they match the
cell. An empty cell is just the extreme
case, zero counts, so every class has to shrink to fit it. So `rTheta`, the strength of
theta's prior, controls how freely theta may stretch to fit such cells: weak, and here the
cell-to-class posterior spreads across every class; strong, and only the classes that are
genuinely near empty in the reference still fit. Even at `rTheta = 20` Zero is not confident on
its own; separating it from those last two classes is a job for a prior weight on the Zero
class.

Do not read this as "`rTheta = 20` is the better setting." It is one cell, shown only to make
the mechanism visible: what theta does and how `rTheta` tunes it. Which value to actually use
is a whole-dataset question, decided by how cells are typed across the board, not by one cell,
and a higher `rTheta` can look more sensible here yet give worse calls overall.

As a rule of thumb, start `rTheta` at roughly the typical number of counts a cell has, then
lower it to let the data drive theta, or raise it and theta tends toward 1, so it stops
rescaling the cell type definitions (the other three factors still act on them).

<small>*The cell in this example comes from Christina's Fiona dataset.*</small>

The block reads the current gene counts per cell, the current cell-type estimates and the
raw cell type definitions, and produces a warped expected expression rescaled at every
level. [Block 3](cell-to-celltype.md) scores cells against types using it.
