"""Numba kernels for the variational-Bayes loop.

These are the fast compiled inner loops, kept out of main.py so the algorithm
code stays readable. The plain-numpy reference version of each kernel lives next
to the method in VarBayes that calls it.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def spots_to_cell_numba_kernel(parent, gene_id, classProb, log_gamma_bar, log_theta,
                               expected_counts, logeta_bar, nNb,
                               wSpotCell, attention, expr_fluct, cell_ineff, gene_ineff):
    """Computes the three spot-to-cell weight terms for every spot and its nearest cells.

    For each spot s and each of its nNb nearest cells, adds up the three terms below.
    It reads values straight from the arrays instead of first copying out the rows, so
    it never builds the large temporary [nS, nK] arrays the numpy version does, and it
    runs over spots on all cores (numba).
      term_1 = sum_k expected_counts[s,k] * classProb[cell,k]
      term_2 = sum_k classProb[cell,k]   * log_gamma_bar[cell, gene_s, k]
      term_3 = sum_k classProb[cell,k]   * log_theta[cell,k]
    mvn_loglik and the misread column are handled by the caller.
    """
    nS = parent.shape[0]
    nK = classProb.shape[1]
    for s in prange(nS):
        g = gene_id[s]
        le = logeta_bar[s]
        for n in range(nNb):
            cell = parent[s, n]
            t1 = 0.0
            t2 = 0.0
            t3 = 0.0
            for k in range(nK):
                cpk = classProb[cell, k]
                t1 += expected_counts[s, k] * cpk
                t2 += cpk * log_gamma_bar[cell, g, k]
                t3 += cpk * log_theta[cell, k]
            wSpotCell[s, n] = t1 + t2 + t3 + le
            attention[s, n] = t1
            expr_fluct[s, n] = t2
            cell_ineff[s, n] = t3
            gene_ineff[s, n] = le

@njit(parallel=True, cache=True)
def gamma_upd_kernel(area_factor, mu, eta_bar, theta_bar, rho, psi_rho, rSpot,
                     beta, log_gamma_bar, gamma_bar):
    """The gamma step for every cell, gene and class in one pass.

    Same maths as VarBayes.gamma_upd in numpy, which stays as the reference:
        beta[c,g,k]          = area_factor[c] * mu[g,k] * eta_bar[g] * theta_bar[c,k] + rSpot
        log_gamma_bar[c,g,k] = psi(rho[c,g]) - log(beta[c,g,k])
        gamma_bar[c,g,k]     = rho[c,g] / beta[c,g,k]
    psi(rho) is only nC by nG, so the caller does it in scipy and passes it in.
    The numpy version builds the nC by nG by nK scaled expression and then walks
    over it three more times, every pass on one core. Here every element is
    touched once and cells are spread over all cores. The Zero column is not
    special-cased here, the caller overwrites it after, same as before.
    """
    nC = rho.shape[0]
    nG = rho.shape[1]
    nK = mu.shape[1]
    for c in prange(nC):
        af = area_factor[c]
        for g in range(nG):
            r = rho[c, g]
            p = psi_rho[c, g]
            e = eta_bar[g]
            for k in range(nK):
                b = af * mu[g, k] * e * theta_bar[c, k] + rSpot
                beta[c, g, k] = b
                log_gamma_bar[c, g, k] = p - np.log(b)
                gamma_bar[c, g, k] = r / b


@njit(parallel=True, cache=True)
def nb_loglik_kernel(area_factor, mu, eta_bar, theta_bar, geneCount, SpotReg, rSpot,
                     contr, contr_sum):
    """The negative binomial loglik of every cell under every class, per gene.

    Same maths as likelihood.compute_gene_loglikelihood_matrix, which stays as
    the reference:
        S[c,g,k]     = area_factor[c] * mu[g,k] * eta_bar[g] * theta_bar[c,k] + SpotReg
        q[c,g,k]     = S / (rSpot + S)
        contr[c,g,k] = geneCount[c,g] * log(q) + rSpot * log(1 - q)
    contr is kept whole because check_cell and the diagnostics read it per gene.
    contr_sum[c,k] is the sum over genes, what cell_to_cellType actually wants,
    added up here so there is no second pass over the big array.
    """
    nC = geneCount.shape[0]
    nG = geneCount.shape[1]
    nK = mu.shape[1]
    for c in prange(nC):
        af = area_factor[c]
        for k in range(nK):
            th = theta_bar[c, k]
            acc = 0.0
            for g in range(nG):
                S = af * mu[g, k] * eta_bar[g] * th + SpotReg
                q = S / (rSpot + S)
                v = geneCount[c, g] * np.log(q) + rSpot * np.log(1.0 - q)
                contr[c, g, k] = v
                acc += v
            contr_sum[c, k] = acc


@njit(parallel=True, cache=True)
def theta_beta_kernel(area_factor, gamma_bar, eta_bar, mu, rTheta, beta):
    """The rate of the theta posterior per cell and class, (nC, nK).

    Same maths as the einsum in VarBayes.theta_upd, which stays as the reference:
        beta[c,k] = sum_g area_factor[c] * gamma_bar[c,g,k] * eta_bar[g] * mu[g,k] + rTheta
    where mu is the reference plus SpotReg. One pass over gamma_bar, cells on all cores.
    """
    nC = gamma_bar.shape[0]
    nG = gamma_bar.shape[1]
    nK = gamma_bar.shape[2]
    for c in prange(nC):
        af = area_factor[c]
        for k in range(nK):
            acc = 0.0
            for g in range(nG):
                acc += gamma_bar[c, g, k] * eta_bar[g] * mu[g, k]
            beta[c, k] = af * acc + rTheta


@njit(parallel=True, cache=True)
def eta_counts_kernel(classProb, mu, area_factor, gamma_bar, theta_bar, partial):
    """The expected reads of each gene in each cell, summed over the classes
    except Zero, (nC, nG). Summing the rows gives the class_total_counts of
    VarBayes.eta_upd, which stays as the reference:
        partial[c,g] = sum_{k<nK-1} classProb[c,k] * mu[g,k] * area_factor[c] * gamma_bar[c,g,k] * theta_bar[c,k]
    Written per cell so the parallel loop never adds into the same slot twice;
    the caller sums over cells, that is only nC by nG.
    """
    nC = gamma_bar.shape[0]
    nG = gamma_bar.shape[1]
    nK = gamma_bar.shape[2]
    for c in prange(nC):
        af = area_factor[c]
        for g in range(nG):
            acc = 0.0
            for k in range(nK - 1):
                acc += classProb[c, k] * mu[g, k] * gamma_bar[c, g, k] * theta_bar[c, k]
            partial[c, g] = af * acc


@njit(parallel=True, cache=True)
def mvn_loglik_kernel(x, parent, centroid, eig_vals, eig_vecs, nu, kappa, E_logdet, nNb, out):
    """The position term of the spot score for every spot and each of its nNb
    nearest cells, (nS, nNb). Same maths as Spots.mvn_loglik, which stays as
    the reference and explains the formula:
        -d/2 log 2pi + 1/2 E[log|Lambda|] - 1/2 (d / kappa + nu (x - m)^T V^-1 (x - m))
    with the quadratic done through the eigen pairs of V. E_logdet is per cell
    and the caller computes it once in scipy. The numpy version is called once
    per neighbour and each call gathers the cell arrays for all spots again;
    here one pass over spots does all nNb neighbours on all cores.
    """
    nS = parent.shape[0]
    d = x.shape[1]
    half_log2pi_d = 0.5 * d * np.log(2.0 * np.pi)
    for s in prange(nS):
        for n in range(nNb):
            c = parent[s, n]
            maha = 0.0
            for j in range(d):
                # (x - m) projected on eigenvector j of V
                dev_u = 0.0
                for i in range(d):
                    dev_u += (x[s, i] - centroid[c, i]) * eig_vecs[c, i, j]
                maha += dev_u * dev_u / eig_vals[c, j]
            maha *= nu[c]
            out[s, n] = -half_log2pi_d + 0.5 * E_logdet[c] - 0.5 * (d / kappa[c] + maha)


@njit(parallel=True, cache=True)
def soft_stats_kernel(ids, w, X, x_bar, nchunk, part1, part2):
    """The per cell sums behind gaussian_model.soft_stats, which stays as the
    reference. One row of ids, w and X per (spot, neighbour) pair.

    Two accumulations, both scatter-adds into cells. Pass 1 (x_bar not given,
    part2 untouched) sums the weight and the weighted coordinates:
        part1[., c, 0]   += w
        part1[., c, 1+j] += w * X[:, j]
    Pass 2 (x_bar given) sums the weighted scatter about the soft mean:
        part2[., c, j*d+k] += w * (X[:, j] - x_bar[c, j]) * (X[:, k] - x_bar[c, k])
    Two threads adding into the same cell would race, so the rows are cut into
    nchunk blocks, each block adds into its own slab of part1 / part2, and the
    caller sums the slabs. That costs nchunk copies of an nC by 13 array, small.
    pass2 is signalled by x_bar having rows; pass 1 gets an empty (0, d) array.
    """
    n = ids.shape[0]
    d = X.shape[1]
    pass2 = x_bar.shape[0] > 0
    block = (n + nchunk - 1) // nchunk
    for b in prange(nchunk):
        lo = b * block
        hi = min(n, lo + block)
        for r in range(lo, hi):
            c = ids[r]
            wr = w[r]
            if pass2:
                for j in range(d):
                    dj = X[r, j] - x_bar[c, j]
                    for k in range(d):
                        part2[b, c, j * d + k] += wr * dj * (X[r, k] - x_bar[c, k])
            else:
                part1[b, c, 0] += wr
                for j in range(d):
                    part1[b, c, 1 + j] += wr * X[r, j]


@njit(cache=True)
def by_cell_index_kernel(parent_flat, gene_id, nN, indptr, order, gene_sorted):
    """For every cell, lists the (spot, neighbour) entries that point at it.

    The spots hold the list the other way round, each spot knows its candidate
    cells. This turns it around.

    parent_flat is parent_cell_id flattened, so entry i belongs to spot i // nN.
    On return cell c owns the stretch indptr[c] to indptr[c + 1] of order and
    gene_sorted. order holds the flat position of each entry (so the same
    position in the flattened parent_cell_prob) and gene_sorted the gene of the
    spot behind it.

    It is a counting sort: count the entries of each cell, turn the counts into
    start positions, then drop every entry into the next free slot of its cell.
    Entries of the same cell keep the order they had in parent_flat. That matters,
    geneCount_upd adds them up in that order and floats care about the order.

    indptr must come in as zeros of length nC + 1, order and gene_sorted as empty
    arrays of length parent_flat.shape[0].
    """
    n = parent_flat.shape[0]
    nC = indptr.shape[0] - 1
    for i in range(n):
        indptr[parent_flat[i] + 1] += 1
    for c in range(nC):
        indptr[c + 1] += indptr[c]
    nxt = indptr[:-1].copy()
    for i in range(n):
        c = parent_flat[i]
        j = nxt[c]
        order[j] = i
        gene_sorted[j] = gene_id[i // nN]
        nxt[c] = j + 1


@njit(parallel=True, cache=True)
def gene_count_kernel(indptr, order, gene_sorted, prob_flat, out, background):
    """The sums behind VarBayes.geneCount_upd, which stays as the reference.

    Every cell walks its own list from by_cell_index_kernel and adds the prob of
    each entry under that entry's gene. A cell's row is only ever touched by the
    thread that has the cell, so the cells can go on all cores with nothing shared.

    The row is summed in a small buffer with the dtype of the probs and written
    to out (float32) once. geneCount_upd does the same thing: it sums in the dtype
    of the probs, float32 on the very first call and float64 after, and then
    casts. Keeping to that is what makes the two agree to the last bit.

    Row 0 of out is the background and is left at zero, like in geneCount_upd.
    Its sums go to background instead. That one is done on a single thread
    because every spot has an entry there, it is one long list.
    """
    nC = indptr.shape[0] - 1
    nG = out.shape[1]
    for g in range(nG):
        out[0, g] = 0
    for c in prange(1, nC):
        row = np.zeros(nG, dtype=prob_flat.dtype)
        for i in range(indptr[c], indptr[c + 1]):
            row[gene_sorted[i]] += prob_flat[order[i]]
        for g in range(nG):
            out[c, g] = row[g]
    for i in range(indptr[0], indptr[1]):
        background[gene_sorted[i]] += prob_flat[order[i]]


@njit(parallel=True, cache=True)
def contributing_count_kernel(indptr, order, prob_flat, tol, kept):
    """For every cell, how many of its (spot, neighbour) entries have a prob above tol.

    First half of the contributing spots of cells_summary, the second half is
    contributing_fill_kernel. The counts are needed first, to know where in the
    output each cell starts. Cell 0, the background, is skipped: its row is
    dropped from cellData anyway and it is by far the longest list.
    """
    nC = indptr.shape[0] - 1
    for c in prange(1, nC):
        n = 0
        for i in range(indptr[c], indptr[c + 1]):
            if prob_flat[order[i]] > tol:
                n += 1
        kept[c] = n


@njit(parallel=True, cache=True)
def contributing_fill_kernel(indptr, order, gene_sorted, prob_flat, tol, nN, nG,
                             cell_off, out_spot, out_gene):
    """For every cell, its spots with a prob above tol, put in gene order.

    Cell c gets the stretch cell_off[c] to cell_off[c + 1] of out_spot (the row
    of the spot) and out_gene (its gene). Inside the stretch the genes go up, and
    the spots of one gene keep the order they have in the spot table. It is a
    counting sort by gene, one cell at a time: count the kept spots of each gene,
    turn the counts into start positions, then drop each spot into the next free
    slot of its gene.

    The cell lists come from by_cell_index_kernel, so nothing is sorted by cell
    here, that was done once at the start of the run. Cells go on all cores, a
    cell only writes inside its own stretch.
    """
    nC = indptr.shape[0] - 1
    for c in prange(1, nC):
        a = cell_off[c]
        if cell_off[c + 1] == a:
            continue
        nxt = np.zeros(nG + 1, dtype=np.int64)
        for i in range(indptr[c], indptr[c + 1]):
            if prob_flat[order[i]] > tol:
                nxt[gene_sorted[i] + 1] += 1
        for g in range(nG):
            nxt[g + 1] += nxt[g]
        for i in range(indptr[c], indptr[c + 1]):
            pos = order[i]
            if prob_flat[pos] > tol:
                g = gene_sorted[i]
                j = a + nxt[g]
                out_spot[j] = pos // nN
                out_gene[j] = g
                nxt[g] += 1
