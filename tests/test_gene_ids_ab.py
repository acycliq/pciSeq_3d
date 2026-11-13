"""
Checks that the gene panel and the gene id of every spot are the same as they
were when they came from np.unique.

Spots now gets them from spots.gene_ids (a hash table, then a sort of the unique
names only) and Genes takes the panel from Spots. Before, Spots and Genes each
ran np.unique over the gene name of every spot. That older code is kept below,
in _gene_ids_before, and the two must give the very same arrays. The order of
the panel matters most: gene id g is row g of the single cell reference.
"""
import numpy as np
import pandas as pd
import pytest

from pciSeq.src.core.datatypes.spots import gene_ids


def _gene_ids_before(gene_names):
    """NOTE: this is how Spots.__init__ and Genes.__init__ got the gene panel,
    the gene ids and the counts per gene before gene_ids replaced it. It is not
    used anywhere in pciSeq anymore, it lives here only to check the new code
    against."""
    panel, gene_id, counts = np.unique(gene_names, return_inverse=True, return_counts=True)
    return panel, gene_id, counts


def _check(names):
    panel_0, id_0, counts_0 = _gene_ids_before(names)
    panel, gene_id, counts = gene_ids(names)
    assert panel.dtype == panel_0.dtype and np.array_equal(panel, panel_0)
    assert np.array_equal(gene_id, id_0)
    assert np.array_equal(counts, counts_0)
    # every spot points back at its own gene
    assert np.array_equal(panel[gene_id], np.asarray(names))


def test_same_on_a_shuffled_panel():
    rng = np.random.default_rng(0)
    panel = np.array(['g%05d' % i for i in range(3000)], dtype=object)
    _check(panel[rng.integers(0, 3000, 200_000)])


def test_same_on_names_that_sort_in_odd_ways():
    # capitals before lower case, digits before letters, a name that is the
    # start of another, and one gene that turns up late and only once
    names = ['b', 'B', 'a', '1a', '10', '2', 'Sst', 'Sst2', 'sst', 'Zbtb20', 'a', 'B', '10', 'aa', 'A']
    _check(np.array(names * 7 + ['Aaa1'], dtype=object))


def test_same_when_the_first_names_seen_are_the_last_in_the_alphabet():
    # factorize numbers the genes in the order it meets them. If the renumbering
    # to sorted order were left out this would be the first to show it
    _check(np.array(['z', 'y', 'x', 'w', 'a', 'z', 'a', 'm'], dtype=object))


def test_same_with_one_gene():
    _check(np.array(['only'] * 5, dtype=object))


def test_a_missing_gene_name_is_an_error():
    with pytest.raises(ValueError):
        gene_ids(np.array(['a', None, 'b'], dtype=object))


def test_spots_and_genes_agree_on_a_model(minimal_varbayes):
    vb = minimal_varbayes
    panel_0, id_0, counts_0 = _gene_ids_before(vb.spots.data.gene_name.values)
    assert np.array_equal(vb.genes.gene_panel, panel_0) and vb.genes.gene_panel.dtype == panel_0.dtype
    assert vb.genes.nG == len(panel_0)
    assert np.array_equal(vb.spots.gene_id, id_0) and vb.spots.gene_id.dtype == np.int32
    assert np.array_equal(vb.spots.counts_per_gene, counts_0)
    # the reference is lined up with the panel by position
    assert list(vb.single_cell.mean_expression.index) == list(panel_0)
