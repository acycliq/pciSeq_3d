# Derivation: the Unclassified class

::: warning Experimental
The Unclassified class is off by default (`unclassified_class: False`) and its behaviour
is still being assessed on real data.
:::

The cell type definitions seldom cover every cell type present in a section. A cell of a
type that is missing from them still receives a class: the nearest one in the definitions,
or [Zero](overview.md#the-zero-class) when it holds few spots. The Unclassified class is
one more class, with no entry in the cell type definitions, intended for the cells that no
defined class explains. Every other class reads its mean expression $\mu_{g,k}$ from the
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
from the cell type definitions. For a given $k$ the candidate value of gene $g$ is the mean
of its $k$ highest values over the defined classes (Zero excluded):

$$
c^{(k)}_g = \frac{1}{k} \sum_{j=1}^{k} \mu_{g,(j)},
\qquad
\mu_{g,(1)} \ge \mu_{g,(2)} \ge \dots \ge \mu_{g,(K)} .
$$

$k = 1$ is the maximum over the classes and $k = K$, the number of defined classes, is their
plain mean. A candidate that resembles one of the defined classes would compete with that
class for its cells. Every $k$ from $1$ to $K$ is therefore tried, and the one kept is the
candidate whose most similar class is least similar to it:

$$
k^\ast = \arg\min_k \; \max_{j} \;
\frac{c^{(k)} \cdot \mu_{\cdot,j}}{\lVert c^{(k)} \rVert \, \lVert \mu_{\cdot,j} \rVert} ,
$$

the cosine being taken over the genes of the panel. The chosen $k^\ast$ is written to the
log. The prior mean is the chosen candidate after the `Inefficiency` scaling, with the floor
`SpotReg` ($\varepsilon$) added:

$$
M_g = \eta_0\, c^{(k^\ast)}_g + \varepsilon .
$$

The floor keeps $M_g$ above zero for a gene that the chosen candidate has at zero.

## Prior

Each $m_g$ has a conjugate Gamma prior, independent across genes:

$$
p(m_g) = \mathrm{Gamma}\!\Big(m_g;\; r_u,\; \frac{r_u}{M_g}\Big)
= \frac{(r_u/M_g)^{\,r_u}}{\Gamma(r_u)}\; m_g^{\,r_u - 1}\; e^{-(r_u/M_g)\, m_g},
$$

with shape $r_u$ and rate $r_u / M_g$, where $r_u$ is the **`rUnclassified`** setting. Its
mean and variance are

$$
\mathbb{E}[m_g] = \frac{r_u}{r_u/M_g} = M_g,
\qquad
\mathrm{Var}[m_g] = \frac{r_u}{(r_u/M_g)^2} = \frac{M_g^2}{r_u} .
$$

The mean is $M_g$ for any $r_u$. Equivalently $m_g = M_g\, w_g$ with
$w_g \sim \mathrm{Gamma}(r_u, r_u)$, a variate of mean $1$: the same construction as the
priors on $\theta$, $\gamma$ and $\eta$, scaled by $M_g$, and the same parameterisation as
the prior on $\rho_g$ ([misread density](misread-density.md#prior)). The standard deviation
relative to the mean is $1/\sqrt{r_u}$ for every gene, and $r_u$ acts as a pseudo-count:
the prior is worth $r_u$ reads of each gene.

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
\log q(m_g)
= \sum_c \bar\zeta_{c,u} \Big[ \bar N_{c,g} \log m_g
  - m_g\, A_c\, \bar\eta_g\, \bar\theta_{c\mid u}\, \bar\gamma_{g,c\mid u} \Big]
  + (r_u - 1) \log m_g - \frac{r_u}{M_g}\, m_g + \text{const},
$$

where $\bar\zeta_{c,u}$ is the probability that cell $c$ is Unclassified, $\bar N_{c,g}$ the
expected count of gene $g$ in cell $c$, and $\bar\theta_{c\mid u}$,
$\bar\gamma_{g,c\mid u}$ the scale factors of the cell conditional on class $u$. Grouping
the $\log m_g$ and $m_g$ terms:

$$
\log q(m_g)
= \Big( r_u + \sum_c \bar\zeta_{c,u} \bar N_{c,g} - 1 \Big) \log m_g
- \Big( \frac{r_u}{M_g} + \bar\eta_g \sum_c \bar\zeta_{c,u}\, A_c\, \bar\theta_{c\mid u}\, \bar\gamma_{g,c\mid u} \Big)\, m_g
+ \text{const}.
$$

This is the log-density of a Gamma distribution. Therefore

$$
\boxed{\;
q(m_g) = \mathrm{Gamma}(m_g;\; \hat a_g,\; \hat b_g)
\;}
$$

with

$$
\hat a_g = r_u + \sum_c \bar\zeta_{c,u}\, \bar N_{c,g},
\qquad
\hat b_g = \frac{r_u}{M_g} + \bar\eta_g \sum_c \bar\zeta_{c,u}\, A_c\, \bar\theta_{c\mid u}\, \bar\gamma_{g,c\mid u} .
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
= \frac{r_u + \sum_c \bar\zeta_{c,u}\, \bar N_{c,g}}
       {\dfrac{r_u}{M_g} + \bar\eta_g \sum_c \bar\zeta_{c,u}\, A_c\, \bar\theta_{c\mid u}\, \bar\gamma_{g,c\mid u}} .
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
$r_u$, so the difference is at most $\log r_u - \psi(r_u)$, which is $0.58$ at $r_u = 1$,
and it is approximately $1/(2\hat a_g)$ for a large shape. With no cells in the class the
expected log is $\psi(r_u) - \log r_u + \log M_g$.

### The role of $r_u$ (the prior strength)

- **Large $r_u$.** $\hat a_g \approx r_u$ and $\hat b_g \approx r_u / M_g$, so
  $\bar m_g \approx M_g$: the class stays at its starting values and acts as a fixed
  class.
- **Small $r_u$.** The prior contributes little to the shape and to the rate, and
  $\bar m_g$ is close to the observed count of the gene in the cells of the class divided
  by their expected count at a mean expression of $1$.

The prior is weighed gene by gene. For gene $g$ the data dominate once the cells of the
class hold many more than $r_u$ reads of the gene and their expected count at a mean
expression of $1$ is well above $r_u / M_g$. A gene with a small prior mean $M_g$ has a
large prior rate and needs the most cells, so the estimate of a gene that the starting
values have near the floor rises more slowly than that of one they have high. The default
is $r_u = 1$.

## Limitations

- The steps that read the mean use it as a number. They do not average over $q(m_g)$.
- The floor $\varepsilon$ is not placed the same way in every step, for this class as for
  every other. The updates of $\eta$ and $\theta$ and the update on this page use
  $\bar m_g$. The update of $\gamma$ uses $\bar m_g - \varepsilon$ with no floor. The
  cell-class assignment uses $\bar m_g - \varepsilon$ and adds $\varepsilon$ after the
  scaling by $A_c$, $\bar\eta_g$ and $\bar\theta_{c\mid u}$.
- A mean below $\varepsilon$ is written to the table as $0$, so the steps that add the
  floor read $\varepsilon$ and not $\bar m_g$. The expected log is not floored.
- The evidence lower bound reported with `elbo_per_step` has no terms for the prior and the
  entropy of $q(m_g)$. With the class on it is not the bound of this model.
- The class takes part in the class prior and in the spatial (MRF) coupling like any
  defined class.
- The name `Unclassified` is reserved: a run stops if the cell type definitions already
  contain a class of that name.
