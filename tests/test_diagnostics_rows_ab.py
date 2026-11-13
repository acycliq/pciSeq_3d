"""
Checks that the rows of the spots table of diagnostics.db are the same as they
were before they were built a batch at a time.

export_diagnostics now gets each batch of rows from diagnostics_db._spot_rows.
Before that it made one tuple per spot, with int(), float() and tobytes() on
every value. That older loop is kept below, in _spot_rows_before, and the two
must give the very same list of tuples.
"""
import numpy as np
import pytest

from pciSeq.src.core.io.diagnostics_db import _spot_rows


def _spot_rows_before(index, gene_idx, xs, ys, zs, blob_arrays, start, end):
    """NOTE: this is the loop export_diagnostics used for the spots table before
    _spot_rows replaced it. It is not used anywhere in pciSeq anymore, it lives
    here only to check the new code against. The six named arrays of the old
    code are the entries of blob_arrays, in the same order."""
    batch = []
    for i in range(start, end):
        row = [
            int(index[i]),
            int(gene_idx[i]),
            int(xs[i]), int(ys[i]), float(zs[i]),
        ]
        for arr in blob_arrays:
            row.append(arr[i, :-1].tobytes())
        batch.append(tuple(row))
    return batch


def _arrays(n, nN, with_bonus, seed=0):
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(10 * n, size=n, replace=False)).astype(np.int64)   # ids with gaps
    gene_idx = rng.integers(0, 300, n).astype(np.int32)
    xs = rng.integers(0, 5000, n).astype(np.int32)
    ys = rng.integers(0, 5000, n).astype(np.int32)
    zs = (rng.random(n) * 60).astype(np.float32)
    blobs = [rng.integers(0, 40000, (n, nN)).astype(np.int32)]
    blobs += [rng.standard_normal((n, nN)).astype(np.float32) for _ in range(4)]
    if with_bonus:
        blobs.append(rng.standard_normal((n, nN)).astype(np.float32))
    return index, gene_idx, xs, ys, zs, blobs


@pytest.mark.parametrize('with_bonus', [False, True])
@pytest.mark.parametrize('start, end', [(0, 1000), (0, 1), (250, 777), (990, 1000), (5, 5)])
def test_same_rows(with_bonus, start, end):
    index, gene_idx, xs, ys, zs, blobs = _arrays(1000, 10, with_bonus)
    before = _spot_rows_before(index, gene_idx, xs, ys, zs, blobs, start, end)
    now = _spot_rows(index, gene_idx, xs, ys, zs, blobs, start, end)

    assert now == before
    assert len(now) == end - start
    # sqlite stores a value by its python type, so the types have to match too
    for a, b in zip(now, before):
        assert [type(v) for v in a] == [type(v) for v in b]


def test_a_blob_is_the_spot_row_without_the_background_column():
    index, gene_idx, xs, ys, zs, blobs = _arrays(50, 7, False)
    rows = _spot_rows(index, gene_idx, xs, ys, zs, blobs, 10, 20)
    ids = np.frombuffer(rows[3][5], dtype=np.int32)
    assert ids.tolist() == blobs[0][13, :-1].tolist()


def test_same_rows_when_the_arrays_are_not_contiguous():
    # a blob array can reach here as a view, eg a slice of a bigger array
    index, gene_idx, xs, ys, zs, blobs = _arrays(400, 10, True)
    blobs = [np.asfortranarray(b) for b in blobs]
    assert (_spot_rows(index, gene_idx, xs, ys, zs, blobs, 0, 400)
            == _spot_rows_before(index, gene_idx, xs, ys, zs, blobs, 0, 400))
