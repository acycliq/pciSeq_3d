"""
Checks that Spots.read splits the spots into kept and excluded the same way
as before.

The split is by the exclude_genes list. It used to be worked out with two
python loops over every spot, now it is one isin on the gene name column. The
older code is kept below, in _read_before, and the two must hand back the very
same two frames.
"""
import numpy as np
import pandas as pd
import pytest

from pciSeq.src.core.datatypes.spots import Spots


def _read_before(spots_df, exclude_genes):
    """NOTE: this is Spots.read before the two list comprehensions were
    replaced by isin. It is not used anywhere in pciSeq anymore, it lives here
    only to check the new code against."""
    gene_mask = [True if d not in exclude_genes else False for d in spots_df.gene_name]
    neg_gene_mask = [True if d in exclude_genes else False for d in spots_df.gene_name]
    spots_copy = spots_df.copy()
    spots_df = spots_copy.loc[gene_mask]
    spots_excluded_df = spots_copy.loc[neg_gene_mask]
    return spots_df, spots_excluded_df


def _read_now(spots_df, exclude_genes):
    # read only needs the config off the object, so there is no need to build
    # a whole Spots
    obj = Spots.__new__(Spots)
    obj.config = {'exclude_genes': exclude_genes}
    return obj.read(spots_df)


def _spots(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    genes = np.array(['Sst', 'Sst2', 'sst', 'Zbtb20', 'Ddit4l', 'Pvalb', 'Vip', 'Cck'], dtype=object)
    return pd.DataFrame({
        'gene_name': genes[rng.integers(0, len(genes), n)],
        'x': rng.random(n).astype(np.float32),
        'y': rng.random(n).astype(np.float32),
        'z_plane': rng.integers(0, 5, n).astype(np.float32),
    })


@pytest.mark.parametrize('exclude_genes', [
    [],                          # the default, nothing excluded
    ['Zbtb20', 'Ddit4l'],        # the aang run
    ['Sst'],                     # must not take Sst2 or sst with it
    ['NotInTheData'],
    ['Sst', 'Sst2', 'sst', 'Zbtb20', 'Ddit4l', 'Pvalb', 'Vip', 'Cck'],   # everything
])
def test_same_split(exclude_genes):
    spots = _spots()
    kept_0, excluded_0 = _read_before(spots, exclude_genes)
    kept, excluded = _read_now(spots, exclude_genes)
    pd.testing.assert_frame_equal(kept, kept_0, check_exact=True)
    pd.testing.assert_frame_equal(excluded, excluded_0, check_exact=True)
    assert len(kept) + len(excluded) == len(spots)


def test_the_input_frame_is_left_alone():
    spots = _spots(500)
    original = spots.copy()
    kept, _ = _read_now(spots, ['Sst'])
    kept.loc[kept.index[0], 'x'] = -1.0
    pd.testing.assert_frame_equal(spots, original)
