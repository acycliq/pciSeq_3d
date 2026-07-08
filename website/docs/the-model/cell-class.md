
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
the largest coupling that still keeps Zero ahead by the margin. Its sign follows $D_{c,k}$:

- **$D_{c,k} < 0$** (counts and prior favour Zero): $\beta^\star_{c,k} > 0$. This is the most
  spatial strength Zero can absorb before flipping, so we cap the applied coupling at it.
- **$D_{c,k} \ge 0$** (counts and prior already favour class $k$): $\beta^\star_{c,k} \le 0$.
  There is nothing to protect, the cell is $k$ on evidence alone, so no cap is applied.

The applied strength is the smaller of the user's $\beta$ (the `mrf_beta` setting) and the
threshold:

$$
\beta^{\text{cap}}_{c,k} =
\begin{cases}
\min\!\big(\beta,\ \max(0,\ \beta^\star_{c,k})\big), & D_{c,k} < 0, \\[4pt]
\beta, & D_{c,k} \ge 0 .
\end{cases}
$$

The cap can only lower $\beta$, never raise it. It is switched on by the `apply_mrf_cap`
setting (on by default); with it off, the flat $\beta$ is used everywhere.
