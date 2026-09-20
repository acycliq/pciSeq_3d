"""
read_tsv turns the list and dict columns back from text, and nothing more than that.

It used to do it with eval, which runs whatever is in the cell. That is fine for a
file pciSeq wrote and not fine for one somebody sent you. ast.literal_eval builds the
same lists, dicts, tuples and numbers and refuses anything else.
"""
import pandas as pd
import pytest

from pciSeq.src.core.io.tsv_export import read_tsv


def _write(tmp_path, cells):
    path = tmp_path / "some.tsv"
    pd.DataFrame({"a": cells}).to_csv(path, sep="\t", index=False)
    return path


def test_lists_dicts_and_tuples_come_back(tmp_path):
    path = _write(tmp_path, ["['Astro', 'Oligo']", "[0.71, 0.22]", "{'x': 1}", "(1, 2)"])
    out = read_tsv(path)
    assert out.a.tolist() == [['Astro', 'Oligo'], [0.71, 0.22], {'x': 1}, (1, 2)]


def test_plain_text_and_numbers_are_left_alone(tmp_path):
    path = tmp_path / "plain.tsv"
    pd.DataFrame({"gene": ["Synpr", "Sema5a"], "n": [3, 7]}).to_csv(path, sep="\t", index=False)
    out = read_tsv(path)
    assert out.gene.tolist() == ["Synpr", "Sema5a"]
    assert out.n.tolist() == [3, 7]


def test_code_hidden_in_a_cell_is_not_run(tmp_path):
    marker = tmp_path / "it_ran"
    path = _write(tmp_path, [f"(open({str(marker)!r}, 'w').close(),)"])
    with pytest.raises(ValueError):
        read_tsv(path)
    assert not marker.exists()
