# Notes: where the code departs from the model pages

This page records the places where the implementation departs from the
derivations on the model pages, with the reason and a worked example for each.
It is built but kept out of the sidebar, like the errata, so it reaches the
agent and anyone who has the address without taking the reader of the model
pages through it. The model pages keep the simpler form on purpose.

## 1. The Zero class in the spot assignment

*Since 21 October 2026.*

### What the page says

The [spot assignment page](spot-assignment.md) models the spots as a Poisson
process whose intensity, the rate at which cell $c$ produces spots of gene $g$
at position $x$, is one product for every class,

$$
\lambda_{g,c}(x) = \theta_c\, \mu_{g,k(c)}\, e^{-D_c(x)}\, \gamma_{g,c}\, \eta_g ,
$$

and every sum in the update runs over all classes. Applied to the Zero class
that product is zero, since $\mu$ is the Zero column of the cell type
definitions and that column is all zeros. The page therefore does not say what
the intensity of a Zero cell is.

### What the code does

The Zero class expects no expression, so none of the factors that scale an
expectation apply to it: no cell scale $\theta$, no cell-gene scale $\gamma$,
no efficiency $\eta$. The intensity is, with $\mu$ the cell type definitions
after the Inefficiency factor and $\varepsilon$ the SpotReg floor,

$$
\lambda_{g,c}(x) =
\begin{cases}
(\mu_{g,k} + \varepsilon)\, \theta_{c,k}\, \gamma_{g,c,k}\, \eta_g\, e^{-D_c(x)}, & k \text{ a real class},\\[4pt]
\varepsilon\, e^{-D_c(x)}, & k = \text{Zero}.
\end{cases}
$$

A Zero cell produces spots of every gene at the same small flat rate,
$\varepsilon$, falling off with the distance from the cell the same way as
for any other cell. The only thing that varies from spot to spot on the Zero
line is the position term.

Taking the expectation over the class probabilities $\bar\zeta_{c,k}$, with
$k = 0$ standing for Zero, the score of cell $c$ for spot $s$ is

$$
\begin{aligned}
\log q(z_{s,c}=1) =\;
& \sum_{k} \bar\zeta_{c,k} \log(\mu_{g_s,k} + \varepsilon)
  && \text{all classes; Zero's line is } \log\varepsilon \\
&+ \sum_{k \neq 0} \bar\zeta_{c,k}\, \overline{\log\gamma}_{g_s,c,k}
  && \text{real classes only} \\
&+ \sum_{k \neq 0} \bar\zeta_{c,k} \log\theta_{c,k}
  && \text{real classes only} \\
&+ (1 - \bar\zeta_{c,0})\, \overline{\log\eta}_{g_s}
  && \text{on every real line, not on Zero's} \\
&- D_c(x_s) + \text{const.}
  && \text{on every line, weight 1}
\end{aligned}
$$

The efficiency term carries the weight $1 - \bar\zeta_{c,0}$ because
$\overline{\log\eta}_{g_s}$ is pulled out of the real class terms only, and
their probabilities add up to that. The position term is on the Zero line as
well, a Zero cell still has a position, so it keeps weight 1.

The class step, the efficiency update and the class prior update have always
treated Zero this way. The spot step was the one place that scaled it, and
that is what changed.

### Why

Zero means the class that expects nothing. A factor saying that a cell
expresses 1.15 times what Zero expects has no meaning. It was not harmless
either. The cell-gene scale of a Zero cell is $(r + n) / r$, with $n$ the
reads the cell holds of the gene and $r$ the rSpot setting, because the rate
of its posterior has nothing in it but $r$. A Zero cell holding ten reads of a
gene at $r = 2$ gained $\psi(12) - \log 2 = 1.75$ in the log score on that
gene from the cell-gene scale alone, and used it to keep claiming spots of
that gene.

On a 25000 cell mouse hippocampus run the change moved fewer than one cell in
a hundred, all of them near-empty cells going to Zero, and the run converged
at the same speed. It is adopted because it is the model that makes sense,
not for the numbers.

### A worked example

One spot of gene $g$ lies between two cells. Cell A has been called a real
class, say Oligo, and expects 10 reads of $g$ from the cell type definitions.
Cell B has been called Zero and expects none, only the floor $\varepsilon =
0.1$. The gene is detected poorly, $\eta_g = 0.37$, so A's 10 becomes
$10 \times 0.37 = 3.7$ in practice. B's floor is not an expectation that
detection could shrink: 37 percent of nothing is nothing. A produces reads of
$g$ at $3.7 / 0.1 = 37$ times B's rate, so a read of $g$ seen between them is
37 times more likely to be A's; in the score that is a difference of logs,
$3.61$ in A's favour. The position term is then added: the spot is near B and
far from A.

The score, term by term, with B holding 5 stray reads of $g$ and $r = 2$:

| term | A | B, old code | B, new code |
|---|---:|---:|---:|
| $\log(\mu + \varepsilon)$, the definitions or the floor | $\log 10.1 = 2.31$ | $\log 0.1 = -2.30$ | $-2.30$ |
| $\overline{\log\gamma}$, cell-gene scale | $0$ | $+1.18$ | $0$ |
| $\log\theta$, cell scale | $0$ | $+0.10$ | $0$ |
| $\overline{\log\eta}$, detection | $-0.99$ | $-0.99$ | $0$ |
| $-D$, position | $-3.00$ | $-1.00$ | $-1.00$ |
| total | $-1.68$ | $-3.01$ | $-3.30$ |

The old code took B's floor and multiplied it by B's cell-gene scale, B's
cell scale and the efficiency, as if 0.1 were a reference expectation. Five
reads at $r = 2$ gave the cell-gene scale $+1.18$; the cell scale came out
above one; the efficiency shrank the floor and happened to pull the other
way. None of the three means anything for a class that expects nothing. Under
the new code B's line is the floor and the position, nothing else. A wins by
$1.62$ instead of $1.33$, and B can no longer improve its own case by holding
reads.

### A cell that is partly Zero

The class call is a probability, not a label. If cell B were 90 percent Zero
and 10 percent Oligo, its line is the average of the two: the Oligo line with
weight 0.1 and the Zero line with weight 0.9. The efficiency term is on the
Oligo line and not on the Zero line, so it enters with weight $1 - 0.9 =
0.1$. A cell that is all real class gets the full efficiency term, a cell that
is all Zero gets none, a cell in between gets its real-class part. This is why
the efficiency term, the same number for every candidate cell of a spot, no
longer cancels between candidates with different Zero probabilities.

### What this changes elsewhere

The diagnostics database does not store the efficiency term per spot. It
stores one $\overline{\log\eta}$ per gene and the readers rebuild the term,
so a run fitted with the unscaled Zero writes the key `zero_unscaled` into
its metadata, and the readers weight the term by each candidate's probability
of not being Zero when the key is present. A run without the key was fitted
by the older code and is shown as it was computed. The cell scale and
cell-gene scale columns of a Zero cell are still computed and exported; the
loop never reads them.
