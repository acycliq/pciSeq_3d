
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
