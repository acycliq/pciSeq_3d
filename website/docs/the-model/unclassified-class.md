# Derivation: the Unclassified class

::: warning Experimental
The Unclassified class is off by default (`unclassified_class: False`) and its behaviour
is still being assessed on real data.
:::

Every cell is given one of the classes in the cell type definitions, or
[Zero](overview.md#the-zero-class) when it holds few spots, even when none of them fits.
That happens in two cases. The cell is of a type that the definitions do not include. Or
the cell holds a mix of genes that no single class expresses together, which occurs even
when the definitions are complete. The Unclassified class is a "none of the above" class
for these cells: one more class, with no entry in the cell type definitions. Every other
class reads its mean expression $\mu_{g,k}$ from the
cell type definitions and holds it fixed. For the Unclassified class the mean expression is
unknown: it has a prior, and its posterior is updated on every iteration from the cells
currently assigned to the class.

Throughout this page $u$ is the index of the Unclassified class and

$$
m_g \;\equiv\; \eta_0\,\mu_{g,u}
$$

is its mean expression for gene $g$, on the scale of the reference after the `Inefficiency`
scaling $\eta_0$ (see the [efficiency derivation](scale-factors.md#eta)).

## Starting values

The prior needs a mean, $M_g$, and the class needs a column to start from. Both are built
from the cell type definitions.

Briefly, for each gene, the defined classes are ranked according to their mean gene counts,
and the $k$ classes with the highest values are retained. The starting value for that gene
is then calculated as the average of its mean counts across the retained classes. The
selection of classes is performed independently for each gene, meaning that the retained
classes may differ between genes. However, the number of classes retained, $k$, is fixed
across all genes and is not user-defined.

Repeating this process for every gene produces a candidate for the class. Each candidate is
represented as a vector containing one value per gene, corresponding to the starting value
calculated for that gene, denoted $\mu^\ast_{g,k}$ below. Similarly, each defined class is represented by a vector of the
same length, containing the mean count of each gene within that class. There is one
candidate for each value of $k$, from $1$ to the number of defined classes.

For each candidate, the cosine similarity is calculated between its vector and the vector
of every defined class. The highest similarity identifies the defined class most similar to
that candidate. Finally, the candidate selected is the one with the lowest maximum cosine
similarity, i.e. the candidate that is furthest from its closest defined class.

Formally, for a given $k$, let
$T_{g,k}$ be the $k$ defined classes (Zero excluded) with the highest values of gene $g$.
The candidate mean expression of gene $g$, $\mu^\ast_{g,k}$, is the mean over those classes:

$$
\mu^\ast_{g,k} = \frac{1}{k} \sum_{j \in T_{g,k}} \mu_{g,j} .
$$

$k = 1$ is the maximum over the classes and $k = K$, the number of defined classes, is their
plain mean. A candidate that resembles one of the defined classes would compete with that
class for its cells. Every $k$ from $1$ to $K$ is therefore tried, and the one kept is the
candidate whose most similar class is least similar to it:

$$
k^\ast = \arg\min_k \; \max_{j} \;
\frac{\sum_g \mu^\ast_{g,k}\, \mu_{g,j}}{\sqrt{\sum_g (\mu^\ast_{g,k})^2}\; \sqrt{\sum_g \mu_{g,j}^2}} ,
$$

the ratio being the cosine between candidate $k$ and class $j$, with the sums over the genes
of the panel. The chosen $k^\ast$ is written to the log. The prior mean is the chosen
candidate after the `Inefficiency` scaling, with the floor `SpotReg` ($\varepsilon$) added:

$$
M_g = \eta_0\, \mu^\ast_{g,k^\ast} + \varepsilon .
$$

The floor keeps $M_g$ above zero for a gene that the chosen candidate has at zero.

## Prior

Each $m_g$ has a conjugate Gamma prior, independent across genes:

$$
p(m_g) = \mathrm{Gamma}\!\Big(m_g;\; r,\; \frac{r}{M_g}\Big)
= \frac{(r/M_g)^{\,r}}{\Gamma(r)}\; m_g^{\,r - 1}\; e^{-(r/M_g)\, m_g},
$$

with shape $r$ and rate $r / M_g$, where $r$ is the **`rUnclassified`** setting. Its
mean and variance are

$$
\mathbb{E}[m_g] = \frac{r}{r/M_g} = M_g,
\qquad
\mathrm{Var}[m_g] = \frac{r}{(r/M_g)^2} = \frac{M_g^2}{r} .
$$

The mean is $M_g$ for any $r$. Equivalently $m_g = M_g\, w_g$ with
$w_g \sim \mathrm{Gamma}(r, r)$, a variate of mean $1$: the same construction as the
priors on $\theta$, $\gamma$ and $\eta$, scaled by $M_g$, and the same parameterisation as
the prior on $\rho_g$ ([misread density](misread-density.md#prior)). The standard deviation
relative to the mean is $1/\sqrt{r}$ for every gene, and $r$ acts as a pseudo-count:
the prior is worth $r$ reads of each gene.

## Likelihood

Given that cell $c$ is of class $u$, the count of gene $g$ in the cell is Poisson, with the
same mean the [scale factor derivations](scale-factors.md) use for a defined class, where
$m_g$ stands in the place of $\eta_0\mu_{g,k}$:

$$
N_{c,g} \mid \zeta_{c,u}=1 \;\sim\; \mathrm{Poisson}\big(m_g\, A_c\, \eta_g\, \theta_c\, \gamma_{g,c}\big).
$$

The terms of the log-joint that contain $m_g$ are, summing over cells,

$$
\sum_c \zeta_{c,u} \Big[ N_{c,g} \log m_g \;-\; m_g\, A_c\, \eta_g\, \theta_c\, \gamma_{g,c} \Big]
\;+\; \log p(m_g) .
$$

## Variational update

Under the mean-field approximation, the update for $q(m_g)$ is the expectation of those
terms over all other latent variables:

$$
\begin{aligned}
\log q(m_g)
={}& \sum_c \bar\zeta_{c,u} \Big[ \bar N_{c,g} \log m_g
  - m_g\, A_c\, \bar\eta_g\, \bar\theta_{c\mid u}\, \bar\gamma_{g,c\mid u} \Big] \\
& + (r - 1) \log m_g - \frac{r}{M_g}\, m_g + \text{const},
\end{aligned}
$$

where $\bar\zeta_{c,u}$ is the probability that cell $c$ is Unclassified, $\bar N_{c,g}$ the
expected count of gene $g$ in cell $c$, and $\bar\theta_{c\mid u}$,
$\bar\gamma_{g,c\mid u}$ the scale factors of the cell conditional on class $u$. Grouping
the $\log m_g$ and $m_g$ terms:

$$
\begin{aligned}
\log q(m_g)
={}& \Big( r + \sum_c \bar\zeta_{c,u} \bar N_{c,g} - 1 \Big) \log m_g \\
& - \Big( \frac{r}{M_g} + \bar\eta_g \sum_c \bar\zeta_{c,u}\, A_c\, \bar\theta_{c\mid u}\, \bar\gamma_{g,c\mid u} \Big)\, m_g
+ \text{const}.
\end{aligned}
$$

This is the log-density of a Gamma distribution. Therefore

$$
\boxed{\;
q(m_g) = \mathrm{Gamma}(m_g;\; \hat a_g,\; \hat b_g)
\;}
$$

with

$$
\hat a_g = r + \sum_c \bar\zeta_{c,u}\, \bar N_{c,g},
\qquad
\hat b_g = \frac{r}{M_g} + \bar\eta_g \sum_c \bar\zeta_{c,u}\, A_c\, \bar\theta_{c\mid u}\, \bar\gamma_{g,c\mid u} .
$$

The sums run over the cells; the background ($c = 0$) is not a cell and is excluded. The
shape is the prior pseudo-count plus the reads of gene $g$ held by the cells of the class.
The rate is the prior rate plus the expected count of those cells at a mean expression of
$1$, the count they would show of gene $g$ if $m_g$ were $1$. The update is the counterpart
of the [efficiency update](scale-factors.md#eta): $\eta_g$ is estimated for one gene across
all classes given the means, $m_g$ for one class given $\eta_g$.

The step runs once per iteration, after the cell-class assignment and before the spot
assignment.

## Reading the result

Two moments of $q(m_g)$ enter the other steps.

**The mean.**

$$
\bar m_g = \mathbb{E}[m_g] = \frac{\hat a_g}{\hat b_g}
= \frac{M_g \sum_c \bar\zeta_{c,u}\, \bar N_{c,g} + r\, M_g}
       {M_g\, \bar\eta_g \sum_c \bar\zeta_{c,u}\, A_c\, \bar\theta_{c\mid u}\, \bar\gamma_{g,c\mid u} + r} .
$$

The table of the cell type definitions holds $\max(\bar m_g - \varepsilon,\, 0)$ in the
column of the class, so that the steps adding the floor $\varepsilon$ to the table recover
$\bar m_g$. The updates of $\eta$ and $\theta$ do so and read $\bar m_g$, in the place of
$\eta_0\mu_{g,k} + \varepsilon$. The update of $\gamma$ and the
[cell-class assignment](cell-class.md) read the table as it is, as they do for every class.
With no cells in the class it is $M_g$. With many, the prior terms are negligible and it is
the reads of the gene in the cells of the class divided by their expected count at a mean
expression of $1$: observed over expected, as for $\eta_g$.

**The expected log.**

$$
\overline{\log m}_g = \mathbb{E}[\log m_g] = \psi(\hat a_g) - \log \hat b_g ,
$$

with $\psi$ the digamma function. It is read by the
[spot assignment](spot-assignment.md), as the class expression term of class $u$. Since
$\psi(a) < \log a$, it is below $\log \bar m_g$, and the difference,
$\log \hat a_g - \psi(\hat a_g)$, decreases as the shape grows. The shape is never below
$r$, so the difference is at most $\log r - \psi(r)$, which is $0.58$ at $r = 1$,
and it is approximately $1/(2\hat a_g)$ for a large shape. With no cells in the class the
expected log is $\psi(r) - \log r + \log M_g$.

### The role of $r$ (the prior strength)

$r$ must be above $0$. The two limits are read directly from the mean:

- **$r \to \infty$: the class is frozen at its starting values.** The terms in $r$
  dominate, so $\bar m_g \to r M_g / r = M_g$. The data have no effect, and the class acts
  as one more fixed class whose column is the starting values. The expected log tends to
  $\log M_g$.
- **$r \to 0$: the mean is fully data driven.** The terms in $r$ vanish and $M_g$ cancels,
  so

  $$
  \bar m_g \to \frac{\sum_c \bar\zeta_{c,u}\, \bar N_{c,g}}
  {\bar\eta_g \sum_c \bar\zeta_{c,u}\, A_c\, \bar\theta_{c\mid u}\, \bar\gamma_{g,c\mid u}} ,
  $$

  the observed count of the gene in the cells of the class divided by their expected count
  at a mean expression of $1$. The starting values no longer enter. This limit cannot be
  used in practice, because the expected log does not follow the mean: the shape
  $\hat a_g$ tends to the reads of the gene in the class, and $\psi(\hat a_g) \to -\infty$
  as these tend to $0$. Before the first update the class holds no cells, so this applies
  to every gene, and the class cannot receive spots.

Between the two, $\bar m_g$ is a weighted combination of the estimate from the data and
$M_g$, and $r$ sets the weights.

The prior is weighed gene by gene. For gene $g$ the data dominate once the cells of the
class hold many more than $r$ reads of the gene and their expected count at a mean
expression of $1$ is well above $r / M_g$. A gene with a small prior mean $M_g$ has a
large prior rate and needs the most cells, so the estimate of a gene that the starting
values have near the floor rises more slowly than that of one they have high. The default
is $r = 1$.

## Evidence lower bound

The unknown $m_g$ adds one term to the evidence lower bound, the divergence of its posterior
from its prior, summed over the genes and entering with a negative sign:

$$
-\sum_g \mathrm{KL}\big(q(m_g)\,\|\,p(m_g)\big),
$$

$$
\begin{aligned}
\mathrm{KL}\big(q(m_g)\,\|\,p(m_g)\big)
={}& (\hat a_g - r)\,\psi(\hat a_g) - \log\Gamma(\hat a_g) + \log\Gamma(r) \\
& + r \Big( \log \hat b_g - \log \frac{r}{M_g} \Big)
+ \hat a_g\, \frac{r/M_g - \hat b_g}{\hat b_g} .
\end{aligned}
$$

It is zero while the class is at its prior. In the likelihood terms of the bound the class
enters through the two moments above: $\bar m_g$ in the expected total count and
$\overline{\log m}_g$ in the term of the observed spots. The update on this page maximises
the bound over $\hat a_g$ and $\hat b_g$, and the step can be named in `elbo_per_step` as
`unclassified_upd`.

## Limitations

- The steps that read the mean use it as a number. They do not average over $q(m_g)$.
- The floor $\varepsilon$ is not placed the same way in every step, for this class as for
  every other. The updates of $\eta$ and $\theta$ and the update on this page use
  $\bar m_g$. The update of $\gamma$ uses $\bar m_g - \varepsilon$ with no floor. The
  cell-class assignment uses $\bar m_g - \varepsilon$ and adds $\varepsilon$ after the
  scaling by $A_c$, $\bar\eta_g$ and $\bar\theta_{c\mid u}$.
- A mean below $\varepsilon$ is written to the table as $0$, so the steps that add the
  floor read $\varepsilon$ and not $\bar m_g$. The expected log is not floored.
- The class takes part in the class prior and in the spatial (MRF) coupling like any
  defined class.
- The name `Unclassified` is reserved: a run stops if the cell type definitions already
  contain a class of that name.
