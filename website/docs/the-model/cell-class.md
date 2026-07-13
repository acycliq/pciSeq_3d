
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
& \underbrace{- \mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c\, \gamma_{g,c}
   + \bar N_{c,g}\log(\mu_{g,k}\, \bar\theta_c\, \gamma_{g,c})}_{\text{Poisson}} \\
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
\Big(\prod_g \mathrm{NB}\big(\bar N_{c,g};\, r_\gamma,\, \mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c\big)\Big)
\cdot
\pi_k \exp\Big(\beta \sum_{c'\in\mathcal{N}_c} \bar\zeta_{c',k}\Big)
\;}
$$

The two factors are:

- a **Negative Binomial likelihood** over genes, with effective mean
  $\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c$ - how well the cell's gene counts match the
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

Here $\mathcal{N}_c$ is the set of nearest neighbours of cell $c$ and $\beta \geq 0$
controls the strength of the coupling. The result is a soft preference: a cell is more
readily assigned to a class when its neighbours already share that class, which discourages
isolated, biologically implausible assignments. Because it enters the score additively in
log space and is [capped](#the-mrf-cap), it cannot override the expression evidence. The
coupling can be made class-specific by replacing the scalar $\beta$ with a vector
$(\beta_1, \dots, \beta_K)$, which is what the cap below does.

## The MRF cap

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
$\mu = \mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c + \varepsilon$:

$$
\begin{aligned}
& r_\gamma\log\frac{r_\gamma}{r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c+\varepsilon} \\
& \quad + \bar N_{c,g}\log\frac{\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c+\varepsilon}{r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c+\varepsilon} .
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
&= \sum_g\Big[\, r_\gamma\log\frac{r_\gamma+\varepsilon}{r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c+\varepsilon} + \bar N_{c,g}\log\frac{(\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c+\varepsilon)(r_\gamma+\varepsilon)}{\varepsilon\,(r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c+\varepsilon)} \,\Big] \\
&\quad + \log\frac{\pi_k}{\pi_{\text{Zero}}} + \beta_{c,k}\sum_{c'\in\mathcal{N}_c}\bar\zeta_{c',k} .
\end{aligned}
$$

Only the last term carries $\beta_{c,k}$, and it does so linearly. Collect the two
$\beta$-free blocks (the gene-likelihood difference and the prior difference) into the
data-and-prior score $D_{c,k}$,

$$
\begin{aligned}
D_{c,k} = {} & \sum_g\Big[\, r_\gamma\log\frac{r_\gamma+\varepsilon}{r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c+\varepsilon} + \bar N_{c,g}\log\frac{(\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c+\varepsilon)(r_\gamma+\varepsilon)}{\varepsilon\,(r_\gamma+\mu_{g,k}\, A_c\, \bar\eta_g\, \bar\theta_c+\varepsilon)} \,\Big] \\
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
already assigns to a real class. It is switched on by the `apply_mrf_cap` setting (on by
default); with it off, the flat $\beta$ is used everywhere.

## Why the cap only fires when Zero wins: a cell that would otherwise oscillate

Only capping the cells where Zero wins the whole row is not decoration. The tempting shortcut
is to drop that and apply $\beta^\star_{c,k}$ wherever $D_{c,k} < 0$, one class at a time. On a
genuinely empty cell that is the same thing, because there every real class loses to Zero. But
on a cell whose winner is a real class the per-class version caps the wrong channel, and the
cell never settles. This is exactly why the cap now only fires when Zero wins the whole cell,
and it is worth walking through because it is what the cap did before.

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

The point to hold onto, and the reason this is a cap bug rather than a fact of life: **Zero is
never this cell's winner.** On the evidence the cell is `016`, and Zero never reaches the top.
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

### The fix: only cap when the cell's own winner is Zero

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
land on the boundary. Only capping when Zero wins removes the misfire itself, which is why it
is the fix.
