
# Derivation: the cell-class assignment $q(\zeta)$

This is the posterior over the type of each cell. It is also where the **spatial prior**
enters: the Markov Random Field term that rewards neighbouring cells for sharing a class.

## From a Poisson-Gamma mixture to a Negative Binomial

By the [CAVI update](overview.md#the-variational-approximation), the structured factor for the
class and its cell-gene scale is the expected
[log-joint](overview.md#the-generative-model) over the remaining latents:

$$
\log q(\zeta,\gamma) = \mathbb{E}_{z,\eta,\theta}\big[\log p(x, g, z, \zeta, \gamma, \eta, \theta)\big] + \text{const}.
$$

Because $\theta_c$ is a [point estimate](scale-factors.md#theta), its expectation is exact and
$\theta_c$ passes through as a constant. Taking the expectation and keeping the terms in
$\zeta$ and $\gamma$ leaves a Poisson-Gamma mixture in $\gamma$:

$$
\begin{aligned}
\log q(\zeta, \gamma)
= \sum_{c,k} \zeta_{c,k} \Big[ \sum_g \big[
& \underbrace{- \mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}\, \gamma_{g,c}
   + \bar N_{c,g}\log(\mu_{g,k}\, \bar\theta_{c\mid k}\, \gamma_{g,c})}_{\text{Poisson}} \\
& + \underbrace{(r_\gamma - 1)\log\gamma_{g,c} - r_\gamma\,\gamma_{g,c}}_{\text{prior: }\gamma \sim \mathrm{Gamma}(r_\gamma, r_\gamma)}
  \big] \\
& + \underbrace{\log\pi_k}_{\text{baseline log-prior}}
  + \underbrace{\beta \sum_{c'\in\mathcal{N}_c} \bar\zeta_{c',k}}_{\text{spatial log-prior}}
  + \text{const} \Big] .
\end{aligned}
$$

Integrating out the per-gene-per-cell factor $\gamma_{g,c}$ against its Gamma prior
collapses each gene's Poisson term into a Negative Binomial. The cell-class posterior is
therefore

$$
\boxed{\;
q\big(k(c)=k\big)
\propto
\Big(\prod_g \mathrm{NB}\big(\bar N_{c,g};\, r_\gamma,\, \mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}\big)\Big)
\cdot
\pi_k \exp\Big(\beta \sum_{c'\in\mathcal{N}_c} \bar\zeta_{c',k}\Big)
\;}
$$

The two factors are:

- a **Negative Binomial likelihood** over genes, with effective mean
  $\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}$ - how well the cell's gene counts match the
  warped expectation for class $k$;
- the **baseline prior** $\pi_k$ scaled by the **spatial term**
  $\beta \sum_{c'\in\mathcal{N}_c} \bar\zeta_{c',k}$.

## The spatial prior (MRF)

The spatial term comes from a Markov Random Field prior on the class assignments. In the
extended log-joint it appears as

$$
\beta \sum_{c,k} \sum_{c'\in\mathcal{N}_c} \mathbf{1}(\zeta_{c,k} = \zeta_{c',k} = 1)
= \beta \sum_{c,k} \zeta_{c,k} \sum_{c'\in\mathcal{N}_c} \zeta_{c',k},
$$

which induces the prior

$$
p(\zeta) \propto \exp\Big(\beta \sum_{c,k} \sum_{c'\in\mathcal{N}_c} \zeta_{c,k}\,\zeta_{c',k}\Big) .
$$

Here $\mathcal{N}_c$ is the set of nearest neighbours of cell $c$ (there are
`nNeighbors` of them, 9 by default) and $\beta \geq 0$ controls the strength of the
coupling. The result is a soft preference: a cell is more readily assigned to a class when
its neighbours already share that class, which discourages isolated, biologically
implausible assignments. Because it enters the score additively in log space it competes
with the expression evidence rather than replacing it.

### Weighting the neighbours by distance

The sum above treats all nine neighbours alike, but a cell touching the one being typed
should count for more than one sitting several diameters away. Each neighbour therefore
carries a weight, and the support that multiplies $\beta$ is

$$
S_{c,k} = \sum_{c' \in \mathcal{N}_c} w_{c,c'}\, \bar\zeta_{c',k} .
$$

The weight is a gaussian on the distance, with a width set per cell:

$$
w_{c,c'} \;\propto\; \exp\!\Big(-\big(d_{c,c'} / \sigma_c\big)^2\Big),
\qquad
\sigma_c = \operatorname{median}_{c' \in \mathcal{N}_c} d_{c,c'} ,
$$

then rescaled so that $\sum_{c'} w_{c,c'} = |\mathcal{N}_c|$. That normalisation keeps the
support on the same scale as the class probabilities it multiplies, so $\beta$ means the
same thing everywhere, and it puts a ceiling on the whole term: since $\bar\zeta \le 1$,

$$
S_{c,k} \;\le\; |\mathcal{N}_c| \qquad\text{and so}\qquad
\beta\, S_{c,k} \;\le\; \beta \cdot \texttt{nNeighbors} .
$$

$\sigma_c$ is a scale, not a location: the distance is divided by it, and it is not a
measure of how far a neighbour sits from some reference point. Taking it from the cell's own
neighbour distances means "close" is judged relative to how crowded that piece of tissue is,
so a single $\beta$ applies to dense and sparse regions alike. This is the
`scaled_gaussian` weighting used by BANKSY.

**Example.** Cell $A$ has two neighbours, $B$ at distance 30 and $C$ at 60. The
implementation uses `nNeighbors` of them; two are shown here:

$$
\sigma_A = \operatorname{median}(30, 60) = 45, \qquad
w_B = e^{-(30/45)^2} = 0.641, \qquad
w_C = e^{-(60/45)^2} = 0.169 ,
$$

After rescaling to sum to 2, $w_B = 1.583$ and $w_C = 0.417$, so $B$ contributes to $A$'s
score with 3.8 times the weight of $C$.

### Pooling sister classes

Some classes in the reference are close relatives. `037 DG Glut` and `038 DG-PIR Ex IMN` are
an example: they share most of their markers, `037` is common in the dentate gyrus and `038`
is rare. A `038` cell usually sits inside a patch of `037` cells, and the MRF works against it.

**Example.** Take a cell whose 9 neighbours are 8 confident `037` cells and 1 confident `038`
cell. To keep the numbers simple, say all 9 are equally close (weight 1 each) and
$\beta = 1$. The neighbour bonus is then

- `037`: 8 (eight neighbours back it)
- `038`: 1 (one neighbour backs it)

Now say the cell's own reads fit `038` better: its log-likelihood is $-100$ under `038` and
$-103$ under `037`. Adding the bonus:

| | reads | bonus | total |
|---|---|---|---|
| `037` | $-103$ | $8$ | $-95$ |
| `038` | $-100$ | $1$ | $-99$ |

`037` wins by 4, and after the softmax the cell is `037` with probability 0.98. The reads said
`038`, but the neighbours outvoted them. The neighbours are not really evidence here, though.
Both classes live in the same place, so being surrounded by `037` cells says very little about
whether this cell is `037` or `038`.

The `mrf_pooled_classes` setting fixes this. Put the two classes in a group,

```python
opts = {
    "mrf_pooled_classes": [["037 DG Glut", "038 DG-PIR Ex IMN"]],
}
```

and a neighbour of either class backs both. The bonus becomes $8 + 1 = 9$ for each:

| | reads | bonus | total |
|---|---|---|---|
| `037` | $-103$ | $9$ | $-94$ |
| `038` | $-100$ | $9$ | $-91$ |

The bonus is the same for both so it cancels out, and the reads decide: `038` with
probability 0.95. The neighbours still count against every class outside the group, so a
patch of DG cells still pulls the cell away from, say, an interneuron class.

A group can hold more than two classes, but keep groups small. Pooling does more than stop the
neighbours choosing inside the group: the votes spread across the group add up, and that
total goes up against every class outside it. Take a cell whose 9 neighbours are 4 cells of a
pyramidal class `P` and 5 interneurons, each from a different interneuron class:

| | bonus for `P` | bonus for each interneuron class |
|---|---|---|
| no pooling | $4$ | $1$ |
| all interneurons in one group | $4$ | $5$ |

Without pooling the neighbours favour `P`. With the interneurons pooled they favour
"interneuron", even though no single interneuron class has more than one neighbour. With a
pair like `037` and `038`, which sit in the same place anyway, this hardly matters. With a big
group it can pull in cells the group would not have won before.

**In the maths.** The MRF prior rewards two neighbours for having the same class. Pooling
changes that to "the same class, or two classes in the same group". The support that
multiplies $\beta$ becomes

$$
\tilde S_{c,j} = \begin{cases}
\sum_{k \in G} S_{c,k} & \text{if class } j \text{ is in a group } G, \\
S_{c,j} & \text{otherwise,}
\end{cases}
$$

which in the example is $\tilde S_{037} = \tilde S_{038} = 8 + 1$. The ceiling from the
previous section still holds: a neighbour's class probabilities add up to at most 1, over a
group as well as over one class, so the term still never exceeds $\beta \cdot \texttt{nNeighbors}$.

**A class can only be in one group.**

## Protecting the Zero class

The MRF term brings neighbourhood information into the cell typing, which helps wherever cells
of the same class sit together. However, it could have a side effect on cells with few reads.
Such a cell has little expression of its own, so the neighbour term could become the deciding
factor in which class the cell is assigned, and a cell that should be typed Zero is given
whatever class dominates around it.

The effect propagates. A relabelled cell becomes evidence for its own neighbours, so across a
patch of near-empty cells one real class spreads from cell to cell.

Preventing this means protecting Zero where the cell's own expression supports it, without
weakening the neighbour term for cells that do have reads, since that is what the term is for.
The [Zero boost](#the-zero-boost) does this.

<details>
<summary>Earlier working, retained for the record and due for removal</summary>

## The MRF cap

::: warning Removed from the model
The cap is not implemented in the current code and there is no `apply_mrf_cap` setting. It
was run on four datasets and did not converge: the convergence measure settles into a limit
cycle between 0.42 and 0.54 rather than approaching a fixed point. Two subsequent
modifications, one of them provably optimal per cell, behave the same way. The derivation is
retained here; the measurements are in
[Why the cap does not converge](#why-the-cap-does-not-converge). What is in the code, off
by default, is the [Zero boost](#the-zero-boost).
:::

The spatial term can outweigh the gene evidence when $\beta$ is large and a cell's
neighbours strongly favour one class. That is a problem for the **Zero** class, the
background type with $\mu_{g,\text{Zero}} = 0$ for every gene. A near-empty cell has little
gene evidence of its own, so the spatial term alone can pull it onto whatever class
surrounds it when it should have stayed Zero. The idea of the cap is to compare each real
class $k$ against Zero, find the largest coupling $\beta_{c,k}$ that still leaves Zero the
winner, and never exceed it.

### The Negative Binomial kernel

Drop the combinatorial $\Gamma$ prefactor of the
[Negative Binomial](#from-a-poisson-gamma-mixture-to-a-negative-binomial): it depends only
on the count and the dispersion, not on the class, so it cancels in any comparison between
classes. What is left is the kernel

$$
\log \mathrm{NB}(N; r_\gamma, \mu) \doteq r_\gamma\log\frac{r_\gamma}{r_\gamma+\mu} + N\log\frac{\mu}{r_\gamma+\mu},
$$

where $\doteq$ means "up to that class-independent constant". Every effective mean carries
an additive floor $\varepsilon$ (the `SpotReg` setting). For the Zero class, which expresses
nothing, the mean is just the floor, $\mu = \varepsilon$:

$$
r_\gamma\log\frac{r_\gamma}{r_\gamma+\varepsilon} + \bar N_{c,g}\log\frac{\varepsilon}{r_\gamma+\varepsilon} .
$$

For a real class $k$ the mean is the warped expectation plus the floor,
$\mu = \mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k} + \varepsilon$:

$$
\begin{aligned}
& r_\gamma\log\frac{r_\gamma}{r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}+\varepsilon} \\
& \quad + \bar N_{c,g}\log\frac{\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}+\varepsilon}{r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}+\varepsilon} .
\end{aligned}
$$

### The log-odds against Zero

Subtract the Zero kernel from the class-$k$ kernel, sum over genes, and add the prior and
spatial differences. We also set $\beta_{c,\text{Zero}} = 0$: Zero carries no spatial support
of its own, so neighbours can pull a cell into a real class but never push one into Zero. The
log-odds are

$$
\begin{aligned}
\Delta_{c,k} &:= \log q(k \mid c) - \log q(\text{Zero} \mid c) \\
&= \sum_g\Big[\, r_\gamma\log\frac{r_\gamma+\varepsilon}{r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}+\varepsilon} + \bar N_{c,g}\log\frac{(\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}+\varepsilon)(r_\gamma+\varepsilon)}{\varepsilon\,(r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}+\varepsilon)} \,\Big] \\
&\quad + \log\frac{\pi_k}{\pi_{\text{Zero}}} + \beta_{c,k}\sum_{c'\in\mathcal{N}_c}\bar\zeta_{c',k} .
\end{aligned}
$$

Only the last term carries $\beta_{c,k}$, and it does so linearly. Collect the two
$\beta$-free blocks (the gene-likelihood difference and the prior difference) into the
data-and-prior score $D_{c,k}$,

$$
\begin{aligned}
D_{c,k} = {} & \sum_g\Big[\, r_\gamma\log\frac{r_\gamma+\varepsilon}{r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}+\varepsilon} + \bar N_{c,g}\log\frac{(\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}+\varepsilon)(r_\gamma+\varepsilon)}{\varepsilon\,(r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_{c\mid k}+\varepsilon)} \,\Big] \\
& + \log\frac{\pi_k}{\pi_{\text{Zero}}} .
\end{aligned}
$$

so that the log-odds are affine in the coupling,

$$
\Delta_{c,k} = D_{c,k} + \beta_{c,k} \sum_{c'\in\mathcal{N}_c}\bar\zeta_{c',k}  .
$$

$D_{c,k}$ is the whole comparison with the spatial term switched off: positive when the
counts and prior favour class $k$, negative when they favour Zero.

### Solving for the cap

We want Zero to keep winning whenever the evidence and prior favour it, so we ask the
log-odds to stay a small margin below zero,

$$
\Delta_{c,k} = D_{c,k} + \beta_{c,k}\, \sum_{c'\in\mathcal{N}_c}\bar\zeta_{c',k} \le -\,\text{tol} .
$$

The margin `tol` is set to `SpotReg`. Without it the target would be an exact tie
($\Delta_{c,k} = 0$), and because Zero is the last class the label could still land on the
real class rather than Zero. Assuming $\sum_{c'\in\mathcal{N}_c}\bar\zeta_{c',k} > 0$ (with no class-$k$ neighbours the spatial
term is zero and cannot promote $k$ anyway), move $D_{c,k}$ across and divide:

$$
\beta_{c,k}\, \sum_{c'\in\mathcal{N}_c}\bar\zeta_{c',k} \le -(D_{c,k} + \text{tol})
\qquad\Longrightarrow\qquad
\beta^\star_{c,k} = -\,\frac{D_{c,k} + \text{tol}}{\sum_{c'\in\mathcal{N}_c}\bar\zeta_{c',k}} .
$$

Since $\sum_{c'\in\mathcal{N}_c}\bar\zeta_{c',k} > 0$, $\Delta_{c,k}$ increases with $\beta_{c,k}$, so $\beta^\star_{c,k}$ is
the largest coupling that still keeps Zero ahead of class $k$ by the margin. When $D_{c,k} < 0$
(the counts and prior favour Zero over $k$) it is positive, the most spatial strength Zero can
absorb before $k$ overtakes it; when $D_{c,k} \ge 0$ ($k$ already beats Zero on the evidence)
it is non-positive and there is nothing for the cap to hold back.

### Only cap the cells whose own data points to Zero

$\beta^\star_{c,k}$ answers a pairwise question: how much coupling keeps Zero ahead of *this
one* class $k$. But the cap should only step in when Zero is where the cell actually belongs on
its own evidence, and that is a statement about the whole cell, not a single class: Zero must
beat *every* real class, not just $k$. That holds exactly when

$$
D_{c,j} < 0 \quad\text{for every real class } j,
$$

that is, when Zero is the class the counts and prior would pick with the spatial term switched
off. On those cells, and only those, we cap each real class $k$ at $\beta^\star_{c,k}$ so the
neighbours cannot overturn Zero. On any other cell some real class already beats Zero on the
evidence, so there is no Zero label to protect: we keep the full $\beta$ and let the neighbours
clean the cell from one real class to another, which is what the spatial prior is for. So the
applied strength is the smaller of the user's $\beta$ (the `mrf_beta` setting) and the
threshold, applied only when Zero wins the whole cell:

$$
\beta^{\text{cap}}_{c,k} =
\begin{cases}
\min\!\big(\beta,\ \max(0,\ \beta^\star_{c,k})\big), & D_{c,j} < 0 \ \text{ for every real class } j, \\[4pt]
\beta, & \text{otherwise.}
\end{cases}
$$

(Inside such a cell every $D_{c,j} < 0$, so the top branch runs for all its real classes at
once.) The cap can only lower $\beta$, never raise it, and it never acts on a cell the data
already assigns to a real class. When the cap was in the code it was controlled by an
`apply_mrf_cap` setting; the flat $\beta$ is what the model uses now.

## Why the cap only fires when Zero wins: a cell that would otherwise oscillate

The whole-row condition is load-bearing. The alternative is to apply $\beta^\star_{c,k}$
wherever $D_{c,k} < 0$, one class at a time. On a
genuinely empty cell that is the same thing, because there every real class loses to Zero. But
on a cell whose winner is a real class the per-class version caps the wrong channel, and the
cell never settles. This is why the whole-row condition is required; the per-class form is
what the cap used originally.

Read the per-class version as a function of $D_{c,k}$ with the neighbours held fixed. For
$D_{c,k}\ge 0$ the class gets the full coupling $\beta$. For $D_{c,k}$ just below zero,
$\beta^\star_{c,k}=-(D_{c,k}+\text{tol})/\sum_{c'\in\mathcal{N}_c}\bar\zeta_{c',k}$ goes negative
and the $\max(0,\cdot)$ clamps it to $0$. So the applied coupling does not ease down as
$D_{c,k}$ crosses zero, it jumps: full $\beta$ on one side, $0$ on the other. A cell sitting
right on $D_{c,k}=0$ for some class it does not even belong to can get stuck bouncing across
it, and the VB loop then never settles.

### A real cell that never settles

Take cell 22786 from an Espio 3D run. It has very few reads, about 10. Its nine spatial
neighbours are frozen across iterations: roughly five are stably the real class
`030 L6 CT CTX Glut` and the rest are Zero. They never flip, so the raw neighbour vote
$\sum_{c'\in\mathcal{N}_c}\bar\zeta_{c',030}$ is essentially constant, worth about $6$ in
$\beta\cdot\text{support}$ units.

On its own gene data the cell prefers `016 CA1-ProS Glut`, and it also fits `030` a few
log-odds units better than Zero (the run shows `030` ahead of Zero by about $+2$ to $+5$ every
iteration). So on the reads alone the order is `016`, then `030`, then Zero. This near tie
between `030` and Zero is not in the gene data. It is put there by the Zero prior: the
`{Zero:0.3}` head start for Zero is about the same size as `030`'s lead over Zero on the genes,
so once the prior is folded in the two nearly cancel and $D_{c,030}$, which carries that prior,
lands right on $0$:

$$
D_{c,016} > 0 \quad(\text{winner}), \qquad D_{c,030} \approx 0 \quad(\text{level with Zero}).
$$

So with the spatial term switched off `016` is the clear winner, and `030` and Zero sit in a
near tie just below it. Because the cell has so few reads, gaining or losing a single owned
spot swings the gene term enough to push $D_{c,030}$ from just above $0$ to just below, and
that back-and-forth is what the cap turns into a flip. Now watch one full lap of the loop under
the per-class cap:

- $D_{c,030}\ge 0$: cap off on `030`, the full neighbour vote ($\approx 6$) is applied. It
  beats the cell's own mild preference for `016`, so the cell is called `030`.
- Being `030` changes which handful of nearby spots the cell holds (the
  [spot-to-cell step](../how-it-works/spots-to-cells.md)), which nudges its per-gene counts
  $\bar N_{c,g}$. The nudged counts make it look slightly more empty, so $D_{c,030}$ dips
  just below $0$.
- $D_{c,030} < 0$: the per-class cap fires on `030` and the neighbour term collapses to
  $\approx 0$. With the neighbours silenced the cell falls back to its own preferred class
  `016`.
- Being `016` shifts its spots again, $\bar N_{c,g}$ moves back, $D_{c,030}$ climbs back
  above $0$, the cap releases, and the neighbours pull it back to `030`. Repeat forever.

The neighbours do not change and the raw neighbour vote does not change. What toggles is the
cap on the `030` channel, through $D_{c,030}$ crossing $0$. A $\approx 6$-unit term is switched
fully on and fully off, so the label flips even though the data term barely moves.

The reason this is a defect in the cap rather than an inherent limitation: **Zero is never
this cell's winner.** On the evidence the cell is `016`, and Zero never reaches the top.
The per-class cap fired on the `030` channel to keep Zero ahead of `030`, but the cell was never
going to be Zero, so it was defending a label it would never take, and each time it did so it
knocked out a legitimate neighbour clean to `030`.

### Why one cell can hold up the whole run

Convergence is judged by the max (L-infinity) change in the spot-to-cell probabilities
against the `CellCallTolerance` (0.02). Each flip of one boundary cell swings its few spots'
assignment probabilities by a lot, from about 0.1 up to 0.5. So even one or two of these
cells keep the max change well above tolerance forever, even though 99.9%+ of all spots
settled long ago (the mean change is around $10^{-5}$). A couple of genuinely ambiguous
cells hold the entire run hostage.

### Restricting the cap to cells whose own winner is Zero

Only capping when Zero wins, as in the boxed formula, removes this at the source. Cell 22786's
counts-and-prior winner is `016`, a real class, so Zero does not win its row and the cap is not
applied to it at all. The `030` channel keeps its full $\beta$ every iteration, the constant
neighbour vote holds the cell at `030`, and it stops flipping. A genuinely empty cell (every
$D_{c,j} < 0$) still has Zero as its winner and is still capped, so the protection the cap was
built for is untouched. On the
Espio run this converges on the existing L-infinity criterion at iteration 83 at the production
settings ($\beta = 1$, `rTheta` $= 5$), and it holds across `rTheta` from 2 to 10.

Restricting the cap this way does not smooth it. The coupling is still a hard switch at
$D_{c,k}=0$; it just never engages on a cell whose winner is a real class, which is where 22786
and cells like it lived. A cell genuinely balanced between Zero and a single real class could
in principle still oscillate where its $D_{c,k}$ crosses $0$, but on the Espio data none do and
the run converges.

Raising `rTheta` also converged the old run, back when the cap fired on any class below Zero,
because it shifts $\theta$ for the Zero class and so shifts $D_{c,k}$, sliding a stuck cell off
the $D_{c,k}=0$ boundary onto its neighbour class. But that only moves where the boundary sits,
it does not remove the misfire, so it is dataset specific: on a harder dataset another cell can
land on the boundary.

## Why the cap does not converge

The section above closes with a caveat:

> A cell genuinely balanced between Zero and a single real class could in principle still
> oscillate where its $D_{c,k}$ crosses $0$, but on the Espio data none do and the run
> converges.

On other datasets such cells are present and the run does not converge. Restricting the cap
to cells whose own winner is Zero removes the misfire described above, but not the
underlying problem.

**The cap places every protected cell on the decision boundary.** This follows from the
definition of $\beta^\star$, which is the largest coupling leaving Zero ahead by exactly
`tol`. A protected cell is therefore within `tol` of changing class, and the spot-to-cell
step perturbs the gene counts by more than that on each iteration.

Two subsequent modifications were measured.

**Adding the $\theta$ prior to $D$.** One variant tested whether $D$ should carry
$\log \text{Ga}(\bar\theta_{c\mid k}; r_\theta, r_\theta)$ alongside the terms derived above.
It should not. The prior is placed on $\theta_c$, a single per-cell scale factor carrying no
class index. Only the estimate $\hat\theta_{c\mid k}$ acquires one, and it does so because
$\mu_{g,k}$ sits in the denominator of the estimator, not because the variable is
class-specific ([scale factors](scale-factors.md#theta)). A term constant across $k$ cancels
in the softmax and in any difference of scores, so it cannot enter $D$. The class dependence
of $\theta$ reaches the update through the likelihood, where $\bar\theta_{c\mid k}$ multiplies
$\mu_{g,k}$, and $D$ already carries it there.

Evaluated at the class-conditional estimate the term is nevertheless not constant across $k$:
on silver 180 its class-to-class spread averages 1.86, against a `tol` of 0.1. Adding it moves
every protected cell's margin by more than the margin itself, which is why it was worth
measuring. The run still cycles, between 0.40 and 0.51 rather than 0.42 and 0.54. Changing
what $D$ contains shifts the boundary without removing it.

**Exact constrained solution.** The protection can be imposed as a constraint rather than a
cap: maximise the ELBO subject to $u_{c,\text{Zero}} \ge u_{c,k}$ on the protected cells.
Restricted to one cell the objective is linear plus entropy, so the constrained maximiser has
a closed form obtained by sorting the gaps $w_k - w_{\text{Zero}}$, equivalent to projection
onto a simplex. Each per-cell solution was verified against a brute-force optimiser. The run
plateaus at 0.370. The exact solution assigns the constrained classes and Zero identical
scores, that is zero slack rather than a finite margin, and constraints active at zero slack
are a documented cause of cycling in active-set methods.

The three variants behave as follows:

| Variant | Convergence measure |
| --- | --- |
| cap, $D$ as derived | cycles between 0.42 and 0.54 |
| cap, $D$ including the $\theta$ prior | cycles between 0.40 and 0.51 |
| exact constrained maximiser | plateaus at 0.370 |
| no Zero protection | converges, $5\times10^{-5}$ |

Measured on silver 180 with `mrf_beta` $=1.5$ and `rTheta` $=2$; the first three do not reach
a fixed point within 300 iterations.

Each variant makes Zero win by a boundary, and the accuracy with which that boundary is
computed does not determine the outcome.

</details>

## The Zero boost

The boost gives Zero a margin that varies continuously with how much evidence the cell has of
its own. No cell is left balanced on a decision boundary, and nothing is conditioned on which
class the data currently favours. Implemented as `zero_boost` and off by default, it sets the
Zero column of the MRF term outright:

$$
\mathrm{mrf}_{c,\text{Zero}} = \beta\,|\mathcal{N}_c|\;e^{-N_c/r_0},
$$

where $N_c$ is the cell's total read count and $r_0$ is the `zero_boost_r0` setting, a decay
length measured in reads. The Zero column is substituted rather than incremented, and the real
classes are untouched, so the MRF still moves a cell between two real classes as before.

### The ceiling

$\beta|\mathcal{N}_c|$ is the largest MRF value any class can attain. The distance weights are
normalised to sum to $|\mathcal{N}_c|$ ([weighting the neighbours by
distance](#weighting-the-neighbours-by-distance)) and no $\zeta_{c',k}$ exceeds 1, so

$$
\mathrm{mrf}_{c,k} = \beta \sum_{c'\in\mathcal{N}_c} w_{c,c'}\,\zeta_{c',k} \;\le\; \beta\,|\mathcal{N}_c|,
$$

with equality only when every neighbour is class $k$ at probability 1. At $N_c = 0$ the boost
equals that ceiling, so on neighbour evidence a real class can at best draw level with Zero and
never exceed it.

Dividing by $\beta$ expresses the boost as a number of neighbour votes,
$|\mathcal{N}_c|\,e^{-N_c/r_0}$.

*One vote here means one neighbour at full weight, $w_{c,c'} = 1$, assigned to that class with
probability 1.* It is a unit of support, not a count of cells: the weights are not equal, so a
close neighbour can carry around 3 votes and a distant one a fraction of that. Three votes might
therefore be one adjacent cell or four remote ones. At $|\mathcal{N}_c| = 9$ and $r_0 = 2$:

| reads in the cell | min neighbour votes to overturn Zero |
| --- | --- |
| 0 | 9.0, the ceiling, so unattainable |
| 2 | 3.3 |
| 4 | 1.2 |

The margin therefore depends on each cell's read count rather than on a single global setting:
complete for an empty cell, negligible for one with reads.

An empty cell cannot be moved off Zero, since the boost equals the ceiling. A miscalled cell in
the wrong region is still corrected by its neighbours, since only the Zero column is modified and
the real classes retain the full $\beta$.

The protection is partial for cells with a few reads: they receive a reduced margin, and enough
agreeing neighbours will still move them. A guarantee would require a threshold, which would
leave those cells on a decision boundary.

### Settings

`zero_boost` is `False` by default, so none of the above applies unless it is switched on.
`zero_boost_r0` sets the decay length; raising it extends the protection to cells with more
reads. Both are described in the [configuration reference](../api/configuration.md).
