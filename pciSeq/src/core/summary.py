import numpy as np
import pandas as pd
import numpy_groupies as npg
from collections.abc import Iterable
from ...src.core.utils.geometry import gaussian_ellipsoids_props, gaussian_contours
from ...src.core.utils.numba_kernels import contributing_count_kernel, contributing_fill_kernel
import logging

logger = logging.getLogger(__name__)


def cells_summary(cells, spots, genes, is3D):
    '''
    returns a dataframe summarising the main features of each cell, ie gene counts and cell types
    :param spots:
    :return:
    '''
    iCounts = np.argsort(-1 * cells.geneCount, axis=1)
    gene_names = genes.gene_panel[iCounts]
    gene_count = np.take_along_axis(cells.geneCount, iCounts, axis=1)

    iProb = np.argsort(-1 * cells.classProb, axis=1)
    class_names = cells.class_names[iProb]
    class_prob = np.take_along_axis(cells.classProb, iProb, axis=1)

    tol = 0.001

    logger.info('Start collecting data ...')

    isCount_nonZero = [d > tol for d in gene_count]
    name_list = [list(gene_names[i][d]) for (i, d) in enumerate(isCount_nonZero)]
    count_list = [((gene_count[i][d] * 1000).astype(np.int32) / 1000).tolist() for i, d in enumerate(isCount_nonZero)]

    # get spot IDs grouped by cell and gene, keeping a spot only if its probability for
    # that cell is above spot_tol.
    #
    # tol / 10 and not tol: tol is the cut-off for CellGeneCount, and a count just above
    # it can be the sum of several spots that are each below it. Cutting the spots at tol
    # too would show such a gene with a count and no spots. Not a guarantee, but any
    # mismatch left can only be around tol.
    spot_tol = tol / 10

    # for each cell one list per shown gene, in the same order as Genenames and
    # CellGeneCount: genes with higher read counts first, and only the genes with
    # counts above the threshold
    spot_id_list = contributing_spot_lists(cells, spots, genes, iCounts, isCount_nonZero, spot_tol)

    isProb_nonZero = [d > tol for d in class_prob]
    class_name_list = [list(class_names[i][d]) for (i, d) in enumerate(isProb_nonZero)]
    prob_list = [((class_prob[i][d] * 1000).astype(np.int32) / 1000).tolist() for i, d in enumerate(isProb_nonZero)]

    # the outline of every cell, 40 points each. All the cells in one call, see
    # gaussian_contours. cells.centroid hands out a copy of the whole table every
    # time it is read, so read it once here and not once per cell.
    centroid_xy = cells.centroid.values[:, :2]
    contour = gaussian_contours(centroid_xy, cells.cov[:, :2, :2], 3).astype(np.int64).tolist()

    df = pd.DataFrame({'Cell_Num': cells.centroid.index.tolist(),
                       'X': ((cells.centroid['x'] * 1000).astype(np.int32) / 1000).tolist(),
                       'Y': ((cells.centroid['y'] * 1000).astype(np.int32) / 1000).tolist(),
                       'Genenames': name_list,
                       'CellGeneCount': count_list,
                       'spot_id': spot_id_list, # the spot ids that when summed-up will generate CellGeneCount
                       'ClassName': class_name_list,
                       'Prob': prob_list,
                       'gaussian_contour': contour
                       })
    # the segmentation centroid as well, X0, Y0, Z0. X, Y, Z is the fitted centroid
    # and the two drift apart when cell_centroid_prior is small, so keep where
    # the cell started
    m0 = cells.ini_centroids()
    if is3D:
        df['sphere_scale'], df['sphere_rotation'] = sphere_props(cells)
        df['Z'] = ((cells.centroid['z'] * 1000).astype(np.int32) / 1000).tolist()
        df['Z0'] = ((m0['z'] * 1000).astype(np.int32) / 1000).tolist()
    else:
        # 2D has no z to report, but the viewer's arrow export asks for the
        # column whatever the data is, so give it zeros. Same idea as z_plane
        # on the spots side. Without this cellData_to_arrow refuses to write
        # and a 2D run comes out with no cells for the viewer.
        df['Z'] = np.float32(0.0)
        df['Z0'] = np.float32(0.0)
    # move column Z after X, Y, then X0, Y0, Z0 after that
    df.insert(3, 'Z', df.pop('Z'))
    df.insert(4, 'X0', ((m0['x'] * 1000).astype(np.int32) / 1000).tolist())
    df.insert(5, 'Y0', ((m0['y'] * 1000).astype(np.int32) / 1000).tolist())
    df.insert(6, 'Z0', df.pop('Z0'))

    df.set_index(['Cell_Num'])

    # Ignore the first row. It is the pseudocell to keep the misreads (ie the background)
    df = df[1:]
    logger.info('Data collected!')
    return df


def _column(values):
    """A plain column for the spot table, as a numpy array pandas takes as it is.

    spots_summary used to hand every column to pandas as a python list, and from
    a list pandas makes int64 for whole numbers and float64 for the rest,
    whatever the numpy type was. The columns here are cast to those two types so
    the table comes out with the very same dtypes as before, without the trip
    through python objects.
    """
    values = np.asarray(values)
    if values.dtype.kind in 'iu':
        return values.astype(np.int64, copy=False)
    if values.dtype.kind == 'f':
        return values.astype(np.float64, copy=False)
    return values


def spots_summary(spots, is3D):
    # check for duplicates (ie spots with the same coordinates with or without the same gene name).
    # is_duplicate = spots.data.duplicated(subset=['x', 'y'])

    idx = np.argsort(-1 * spots.parent_cell_prob, axis=1)
    p = np.take_along_axis(spots.parent_cell_prob, idx, axis=1).round(3)
    nbrs = np.take_along_axis(spots.parent_cell_id, idx, axis=1)
    max_nbrs = nbrs[:, 0]

    # The plain columns go in as numpy arrays, see _column. Only the two ragged
    # ones, neighbour_array and neighbour_prob, are still python lists, one list
    # per spot. They are what takes the time and the memory here now.
    data = spots.data
    out = pd.DataFrame({'gene_name': data.gene_name.to_numpy(dtype=object),
                        'gene_id': _column(spots.gene_id),
                        'spot_id': _column(data.index.to_numpy()),
                        'x': np.round(data.x.to_numpy().astype('float64'), 3),
                        'y': np.round(data.y.to_numpy().astype('float64'), 3),
                        'plane_id': _column(data.plane_id.to_numpy()),
                        'neighbour': _column(max_nbrs),
                        'neighbour_array': nbrs.tolist(),
                        'neighbour_prob': p.tolist(),
                        # 'omp_score': ((spots.data.score * 1000).astype(np.int32)/1000).tolist()
                        })
    # These three used to be written only for 3D, but the viewer reads all of
    # them whatever the data is, so a 2D run left it with nothing to draw. In 2D
    # z is zero for every spot anyway, and score and intensity are filled in by
    # the validator when the caller has none.
    out['z'] = np.round(data.z.to_numpy().astype('float64'), 3)
    out['omp_score'] = np.round(data.score.to_numpy().astype('float64'), 3)
    out['omp_intensity'] = np.round(data.intensity.to_numpy().astype('float64'), 3)
    # which cell the spot physically sits in, straight off the label image, 0 if none.
    # Not the same as 'neighbour': that is the cell the model decided the spot came
    # from, this is just where the segmentation put it. The model only ever looks at
    # it through InsideCellBonus, which is normally 0. Named after inside_cell in
    # label_processing, which is what works it out.
    out['inside_cell'] = _column(data.label.to_numpy())
    # move column z after x, y
    z_pos = out.columns.get_loc('y') + 1
    out.insert(z_pos, 'z', out.pop('z'))

    return out


def collect_data(cells, spots, genes, is3D):
    '''
    Collects data for the viewer
    :param cells:
    :param spots:
    :return:
    '''
    cell_df = cells_summary(cells, spots, genes, is3D)
    gene_df = spots_summary(spots, is3D)
    return cell_df, gene_df


def sphere_props(cells):
    # all the cells in one call, see gaussian_ellipsoids_props
    return gaussian_ellipsoids_props(cells.cov)


def contributing_spot_lists(cells, spots, genes, iCounts, shown, tol):
    """The spot_id column of cellData: for every cell, one list of spot ids per shown gene.

    iCounts is the gene order of each cell (highest count first) and shown says
    which of those genes make it to cellData, both as cells_summary has them. A
    spot is listed under a cell when its probability for that cell is above tol.
    A shown gene with no such spot gets an empty list.

    get_contributing_spots_2 is the reference for which spots belong where. It
    makes a python list for every cell and gene, nearly all of them empty, and
    that does not scale: 10,000 cells on 18,000 genes is 180 million lists. Here
    the spots above tol are laid out once in two flat arrays, cell after cell and
    gene after gene inside a cell, and python lists are only made for the genes
    that are shown. tests/test_contributing_spots_ab.py checks the two give the
    same column.
    """
    indptr, order, gene_sorted = spots.by_cell_index(cells.nC)
    prob = np.ascontiguousarray(spots.parent_cell_prob)
    prob_flat = prob.ravel()
    # same float type as the probs, so the cut is the one numpy makes in
    # parent_cell_prob > tol
    tol = prob.dtype.type(tol)

    kept = np.zeros(cells.nC, dtype=np.int64)
    contributing_count_kernel(indptr, order, prob_flat, tol, kept)
    cell_off = np.zeros(cells.nC + 1, dtype=np.int64)
    np.cumsum(kept, out=cell_off[1:])

    out_spot = np.empty(cell_off[-1], dtype=np.int64)
    out_gene = np.empty(cell_off[-1], dtype=np.int32)
    contributing_fill_kernel(indptr, order, gene_sorted, prob_flat, tol, prob.shape[1], genes.nG,
                             cell_off, out_spot, out_gene)

    # one big python list of the spot ids, every list handed out below is a slice of it
    ids = spots.data.index.values[out_spot].tolist()

    out = []
    for c in range(cells.nC):
        wanted = iCounts[c][shown[c]].tolist()
        a, b = int(cell_off[c]), int(cell_off[c + 1])
        if a == b:
            out.append([[] for _ in wanted])
            continue
        # where each gene of this cell starts and stops inside a to b
        g = out_gene[a:b]
        starts = np.concatenate(([0], np.flatnonzero(np.diff(g)) + 1))
        stops = np.concatenate((starts[1:], [b - a]))
        where = dict(zip(g[starts].tolist(), zip((starts + a).tolist(), (stops + a).tolist())))
        row = []
        for gene in wanted:
            se = where.get(gene)
            row.append(ids[se[0]:se[1]] if se is not None else [])
        out.append(row)
    return out


def get_contributing_spots(spot_ids, gene_id, parent_cell_id, parent_cell_prob, tol=0.001):
    """
    Get only the spot lists for each (cell, gene) combination.
    Stripped down version of aggregate_cell_gene_with_spots.

    Args:
        spot_ids: array of shape (n_spots,) - spot identifiers
        gene_id: array of shape (n_spots,) - gene category for each spot
        parent_cell_id: array of shape (n_spots, n_assignments) - cell labels
        parent_cell_prob: array of shape (n_spots, n_assignments) - probabilities

    Returns:
        spot_lists: array of shape (n_cells, n_genes) - lists of contributing spot_ids (prob > 0.001)
        cell_ids: array of unique cell IDs (row labels)
        gene_ids: array of unique gene IDs (column labels)
    """

    # Get unique cell and gene IDs
    unique_cells = np.unique(parent_cell_id.ravel())
    unique_genes = np.unique(gene_id)

    # Create mapping dictionaries
    cell_to_idx = {cell: idx for idx, cell in enumerate(unique_cells)}
    gene_to_idx = {gene: idx for idx, gene in enumerate(unique_genes)}

    n_cells = len(unique_cells)
    n_genes = len(unique_genes)

    # Expand arrays to match flattened structure
    spot_expanded = np.repeat(spot_ids, parent_cell_id.shape[1])
    gene_expanded = np.repeat(gene_id, parent_cell_id.shape[1])

    # Flatten the 2D arrays
    cells_flat = parent_cell_id.ravel()
    probs_flat = parent_cell_prob.ravel()

    # Filter valid entries (probabilities > 0.001 and valid cells)
    valid_mask = (probs_flat > tol) & np.isin(cells_flat, unique_cells)

    spot_valid = spot_expanded[valid_mask]
    gene_valid = gene_expanded[valid_mask]
    cells_valid = cells_flat[valid_mask]

    # Convert to indices
    gene_indices = np.array([gene_to_idx[g] for g in gene_valid])
    cell_indices = np.array([cell_to_idx[c] for c in cells_valid])

    # Create linear group indices
    group_indices = cell_indices * n_genes + gene_indices

    # Initialize spot lists arrays - REMOVED probability aggregation
    spot_lists_flat = np.empty(n_cells * n_genes, dtype=object)
    for i in range(n_cells * n_genes):
        spot_lists_flat[i] = []

    # Collect spots for each group - ONLY spot collection, no probability aggregation
    for i, group_id in enumerate(group_indices):
        spot_lists_flat[group_id].append(spot_valid[i].item())

    # Reshape results - REMOVED cell_gene_matrix and prob_lists
    spot_lists = spot_lists_flat.reshape(n_cells, n_genes)

    return spot_lists


def get_contributing_spots_2(cells, spots, genes, tol=0.0):
    """For every cell and gene, the ids of the spots of that gene that belong to the cell.

    A spot counts as belonging to a cell when its probability for that cell is above
    tol. With tol at 0 every candidate spot is listed, which is what this did before.
    Returns an object array, cells by genes, with a list of spot ids in each entry.
    """
    nN = spots.parent_cell_id.shape[1]
    cell_ids = spots.parent_cell_id.ravel()
    gene_ids = np.tile(spots.gene_id, (nN, 1)).T.ravel()

    spot_ids = np.tile(spots.data.index.values, (nN, 1)).T
    spot_ids = spot_ids.ravel()

    # parent_cell_prob lines up with parent_cell_id, one column per candidate cell
    keep = spots.parent_cell_prob.ravel() > tol
    group_idx = np.vstack((cell_ids[keep], gene_ids[keep]))

    agg = npg.aggregate_np(group_idx, spot_ids[keep], size=(cells.nC, genes.nG), func=list, fill_value=[], dtype=object)

    # # assert cells.geneCount.shape == agg.shape
    # out = []
    # for i, counts in enumerate(agg):
    #     counts = [list(d) for d in counts]
    #     out.append(counts)
    # # #     spot_id_list = agg[i][mask].tolist()
    # #     spot_id_list = agg[i].tolist()
    # #     out.append(counts)
    return agg