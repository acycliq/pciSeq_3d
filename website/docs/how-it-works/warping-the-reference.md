
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

## Why this block is harder to judge

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

## The warp is a stack of scaling factors

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

- **theta** ($\theta_{c,k}$) - one factor **per cell, for each candidate cell type**.
  Some cells simply yield more transcripts than the definitions predict, others fewer;
  theta is a single whole-cell **scalar** that stretches or shrinks that cell's expected
  counts across all its genes. It is worked out separately for every type the cell might be,
  because what counts as "expected" depends on which type you are testing it against.

- **gamma** ($\gamma_{g,c,k}$) - one factor **per gene, per cell, per candidate type**.
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

## A pyramid of granularity

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

## Inefficiencies: the common statistic

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

## An important point about theta

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

The mrf cap is a limit on the spatial bonus, set separately for each cell and each candidate
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

### A worked example: an empty cell

Cell 16150 in one of our runs has zero gene counts. With `rTheta = 2` and a uniform class
prior, the posterior of its cell-to-class assignment is almost uniform: Zero at 3.5%, all
other classes with prob around 3.1% or 3.2%.

<figure class="diagram">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 410" role="img" aria-label="Class-probability donut for the empty cell 16150: Zero at 3.5 percent, a cluster of classes at 3.2 percent, the rest lower">
  <path d="M 189.67 48.79 A 150 150 0 0 1 220.33 48.79 L 214.50 105.49 A 93 93 0 0 0 195.50 105.49 Z" fill="#cf9f68" fill-opacity="1.0" />
  <path d="M 223.32 56.12 A 150 150 0 0 1 250.15 61.96 L 233.00 116.31 A 93 93 0 0 0 216.36 112.70 Z" fill="#5e9ea4" fill-opacity="0.92" />
  <path d="M 253.02 62.89 A 150 150 0 0 1 278.12 74.03 L 250.33 123.80 A 93 93 0 0 0 234.77 116.89 Z" fill="#61a0a5" fill-opacity="0.92" />
  <path d="M 280.73 75.52 A 150 150 0 0 1 303.07 91.50 L 265.80 134.63 A 93 93 0 0 0 251.95 124.72 Z" fill="#64a1a7" fill-opacity="0.92" />
  <path d="M 305.33 93.49 A 150 150 0 0 1 323.97 113.65 L 278.76 148.36 A 93 93 0 0 0 267.20 135.86 Z" fill="#67a3a8" fill-opacity="0.92" />
  <path d="M 325.78 116.06 A 150 150 0 0 1 339.98 139.57 L 288.68 164.43 A 93 93 0 0 0 279.89 149.85 Z" fill="#6aa4a9" fill-opacity="0.92" />
  <path d="M 341.26 142.29 A 150 150 0 0 1 350.41 168.18 L 295.15 182.17 A 93 93 0 0 0 289.48 166.12 Z" fill="#6da6aa" fill-opacity="0.92" />
  <path d="M 351.12 171.11 A 150 150 0 0 1 354.84 198.06 L 297.90 200.70 A 93 93 0 0 0 295.59 183.99 Z" fill="#70a7ac" fill-opacity="0.92" />
  <path d="M 354.95 201.07 A 150 150 0 0 1 353.32 227.37 L 296.96 218.87 A 93 93 0 0 0 297.97 202.57 Z" fill="#73a9ad" fill-opacity="0.92" />
  <path d="M 352.84 230.34 A 150 150 0 0 1 346.17 255.72 L 292.52 236.44 A 93 93 0 0 0 296.66 220.71 Z" fill="#76abae" fill-opacity="0.92" />
  <path d="M 345.12 258.54 A 150 150 0 0 1 334.52 280.65 L 285.31 251.90 A 93 93 0 0 0 291.87 238.19 Z" fill="#79acb0" fill-opacity="0.92" />
  <path d="M 332.98 283.24 A 150 150 0 0 1 318.98 302.51 L 275.67 265.45 A 93 93 0 0 0 284.35 253.51 Z" fill="#7baeb1" fill-opacity="0.92" />
  <path d="M 317.00 304.77 A 150 150 0 0 1 299.96 321.11 L 263.88 276.99 A 93 93 0 0 0 274.44 266.86 Z" fill="#7eafb2" fill-opacity="0.92" />
  <path d="M 297.61 323.00 A 150 150 0 0 1 279.42 335.24 L 251.14 285.75 A 93 93 0 0 0 262.42 278.16 Z" fill="#81b1b4" fill-opacity="0.92" />
  <path d="M 276.79 336.70 A 150 150 0 0 1 257.05 345.68 L 237.27 292.22 A 93 93 0 0 0 249.51 286.66 Z" fill="#84b2b5" fill-opacity="0.92" />
  <path d="M 254.22 346.70 A 150 150 0 0 1 234.04 352.16 L 223.01 296.24 A 93 93 0 0 0 235.51 292.85 Z" fill="#87b4b6" fill-opacity="0.92" />
  <path d="M 231.08 352.71 A 150 150 0 0 1 210.99 354.88 L 208.72 297.93 A 93 93 0 0 0 221.17 296.58 Z" fill="#8ab6b7" fill-opacity="0.92" />
  <path d="M 207.98 354.97 A 150 150 0 0 1 187.86 354.02 L 194.37 297.39 A 93 93 0 0 0 206.85 297.98 Z" fill="#8db7b9" fill-opacity="0.92" />
  <path d="M 184.87 353.64 A 150 150 0 0 1 165.20 349.62 L 180.32 294.67 A 93 93 0 0 0 192.52 297.16 Z" fill="#90b9ba" fill-opacity="0.92" />
  <path d="M 162.30 348.79 A 150 150 0 0 1 143.71 341.91 L 167.00 289.88 A 93 93 0 0 0 178.53 294.15 Z" fill="#93babb" fill-opacity="0.92" />
  <path d="M 140.98 340.65 A 150 150 0 0 1 123.83 331.14 L 154.67 283.21 A 93 93 0 0 0 165.31 289.10 Z" fill="#96bcbd" fill-opacity="0.92" />
  <path d="M 121.31 329.49 A 150 150 0 0 1 105.92 317.62 L 143.57 274.83 A 93 93 0 0 0 153.11 282.18 Z" fill="#99bdbe" fill-opacity="0.92" />
  <path d="M 103.68 315.61 A 150 150 0 0 1 90.41 301.80 L 133.96 265.02 A 93 93 0 0 0 142.18 273.58 Z" fill="#9cbfbf" fill-opacity="0.92" />
  <path d="M 88.49 299.48 A 150 150 0 0 1 77.45 283.93 L 125.92 253.94 A 93 93 0 0 0 132.77 263.58 Z" fill="#9fc0c1" fill-opacity="0.92" />
  <path d="M 75.89 281.36 A 150 150 0 0 1 67.28 264.43 L 119.61 241.85 A 93 93 0 0 0 124.95 252.34 Z" fill="#a2c2c2" fill-opacity="0.92" />
  <path d="M 66.11 261.65 A 150 150 0 0 1 60.09 243.75 L 115.16 229.03 A 93 93 0 0 0 118.89 240.13 Z" fill="#a5c4c3" fill-opacity="0.92" />
  <path d="M 59.34 240.84 A 150 150 0 0 1 56.07 222.85 L 112.66 216.07 A 93 93 0 0 0 114.69 227.22 Z" fill="#a8c5c4" fill-opacity="0.92" />
  <path d="M 55.74 219.86 A 150 150 0 0 1 55.04 201.67 L 112.02 202.93 A 93 93 0 0 0 112.46 214.21 Z" fill="#abc7c6" fill-opacity="0.92" />
  <path d="M 55.13 198.66 A 150 150 0 0 1 57.00 180.61 L 113.24 189.88 A 93 93 0 0 0 112.08 201.07 Z" fill="#aec8c7" fill-opacity="0.92" />
  <path d="M 57.51 177.65 A 150 150 0 0 1 61.89 160.07 L 116.27 177.14 A 93 93 0 0 0 113.56 188.04 Z" fill="#b0cac8" fill-opacity="0.92" />
  <path d="M 62.82 157.20 A 150 150 0 0 1 69.33 141.03 L 120.88 165.34 A 93 93 0 0 0 116.85 175.37 Z" fill="#b3cbca" fill-opacity="0.92" />
  <path d="M 70.64 138.32 A 150 150 0 0 1 79.07 123.50 L 126.93 154.47 A 93 93 0 0 0 121.70 163.66 Z" fill="#b6cdcb" fill-opacity="0.92" />
  <path d="M 80.73 120.99 A 150 150 0 0 1 90.86 107.68 L 134.23 144.66 A 93 93 0 0 0 127.96 152.91 Z" fill="#b9cfcc" fill-opacity="0.92" />
  <path d="M 92.83 105.41 A 150 150 0 0 1 104.27 93.86 L 142.55 136.09 A 93 93 0 0 0 135.46 143.25 Z" fill="#bcd0ce" fill-opacity="0.92" />
  <path d="M 106.52 91.86 A 150 150 0 0 1 119.32 81.88 L 151.88 128.66 A 93 93 0 0 0 143.94 134.85 Z" fill="#bfd2cf" fill-opacity="0.92" />
  <path d="M 121.81 80.18 A 150 150 0 0 1 135.68 71.98 L 162.02 122.53 A 93 93 0 0 0 153.42 127.61 Z" fill="#c2d3d0" fill-opacity="0.92" />
  <path d="M 138.36 70.62 A 150 150 0 0 1 152.32 64.55 L 172.34 117.92 A 93 93 0 0 0 163.68 121.68 Z" fill="#c5d5d1" fill-opacity="0.92" />
  <path d="M 155.15 63.52 A 150 150 0 0 1 169.39 59.29 L 182.92 114.66 A 93 93 0 0 0 174.09 117.29 Z" fill="#c8d6d3" fill-opacity="0.92" />
  <path d="M 172.32 58.60 A 150 150 0 0 1 186.68 56.12 L 193.64 112.70 A 93 93 0 0 0 184.74 114.23 Z" fill="#cbd8d4" fill-opacity="0.92" />
  <text x="205" y="200" text-anchor="middle" class="donut-mid">cell 16150</text>
  <text x="205" y="221" text-anchor="middle" class="donut-sub">0 gene counts</text>
  <g class="donut-legend">
    <rect x="452" y="128" width="15" height="15" rx="3" fill="#cf9f68" />
    <text x="478" y="140">Zero</text>
    <text x="744" y="140" text-anchor="end" class="donut-pct">3.5%</text>
    <rect x="452" y="176" width="15" height="15" rx="3" fill="#5e9ea4" />
    <text x="478" y="188">Peri, IMN, Lymphoid,</text>
    <text x="478" y="206">DC, Monocytes, HPF CR</text>
    <text x="744" y="188" text-anchor="end" class="donut-pct">3.2%</text>
    <rect x="452" y="242" width="15" height="15" rx="3" fill="#9abebe" />
    <text x="478" y="254">every other class</text>
    <text x="744" y="254" text-anchor="end" class="donut-pct">3.1% and below</text>
  </g>
</svg>
<figcaption>Cell 16150 has no gene counts, so the classes come out almost evenly matched. Zero leads at 3.5%; the six classes just behind it sit at 3.2%, exactly one <code>tol</code> margin back.</figcaption>
</figure>

The near-uniform posterior comes from theta. The cell has no counts, so theta collapses for
every class, scaling each class's predicted expression down to the floor. The gene likelihood
is then the same across classes, and the uniform prior adds no separation, so the posterior is
almost uniform. Zero ends up just ahead because of the cap: it holds every real class a fixed
distance below Zero, so the classes pressing hardest against it, the ones with enough
like-typed neighbours, land right underneath at 3.2% against Zero's 3.5%. They cannot get any
closer.

So an empty cell gets an almost uniform posterior with Zero just ahead, not a confident Zero.
The cap only guarantees Zero wins the argmax; raising its probability (say above 50%) needs a
prior weight on the Zero class, not the cap.

## What feeds in and what comes out

- **Feeds in:** the current gene counts per cell, the current cell-type estimates, and
  the raw cell type definitions.
- **Comes out:** a warped version of the expected expression, rescaled at every level,
  ready for [block 3](cell-to-celltype.md) to score cells against types.