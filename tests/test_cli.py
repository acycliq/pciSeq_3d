"""Tests for the command line entry point.

The loaders are checked against the shapes the old run_app_*.py scripts
actually used, since replacing those is the point of the CLI: a reference that
needs transposing, a tsv reference, spots in parquet needing a rename and a
two condition filter, and a segmentation in an npz under a named key.
"""

import json

import numpy as np
import pandas as pd
import pytest

from pciSeq.cli import (apply_overrides, build_parser, build_run, load_config,
                        main, read_masks, read_scrnaseq, read_spots)


@pytest.fixture()
def inputs(tmp_path):
    """The three inputs on disk, in the awkward variants seen in the wild."""
    # reference stored the way run_izzie_3d.py has it: classes as rows, so it
    # needs transposing to genes x classes
    ref = pd.DataFrame({'GeneA': [1.0, 2.0], 'GeneB': [3.0, 4.0]},
                       index=pd.Index(['Type_1', 'Type_2'], name='subclass'))
    ref.to_csv(tmp_path / 'ref.csv')

    # spots with the column names the raw files use, unfiltered
    spots = pd.DataFrame({
        'Gene': ['GeneA', 'GeneB', 'GeneA', 'GeneB'],
        'x': [1.0, 2.0, 3.0, 4.0],
        'y': [1.0, 2.0, 3.0, 4.0],
        'z_stack': [0, 0, 1, 1],
        'discriminability': [5.0, 1.0, 4.0, 9.0],
        'intensity': [0.9, 0.9, 0.1, 0.9],
    })
    spots.to_parquet(tmp_path / 'spots.parquet')

    seg = np.zeros((2, 8, 8), dtype=np.uint16)
    seg[:, 2:5, 2:5] = 1
    np.savez(tmp_path / 'seg.npz', seg=seg, other=np.zeros((2, 2, 2)))
    np.save(tmp_path / 'seg.npy', seg)
    return tmp_path


class TestConfig:

    def test_yaml_and_json_both_load(self, tmp_path):
        (tmp_path / 'a.yaml').write_text('opts:\n  rTheta: 2\n')
        (tmp_path / 'b.json').write_text('{"opts": {"rTheta": 2}}')
        assert load_config(tmp_path / 'a.yaml') == load_config(tmp_path / 'b.json')

    def test_unsigned_exponents_are_numbers(self, tmp_path):
        """PyYAML is YAML 1.1, where 1.0e6 is the string "1.0e6" unless the
        exponent is signed. rTheta: 1.0e6 reaching the model as text is a
        silent way to lose a run, so the loader handles the unsigned form."""
        (tmp_path / 'c.yaml').write_text(
            'opts:\n  rTheta: 1.0e6\n  rRho: 1e6\n  MisreadDensity: 1.0e-6\n'
            '  rSpot: 2\n  output_path: default\n')
        opts = load_config(tmp_path / 'c.yaml')['opts']

        assert opts['rTheta'] == 1e6 and isinstance(opts['rTheta'], float)
        assert opts['rRho'] == 1e6 and isinstance(opts['rRho'], float)
        assert opts['MisreadDensity'] == 1e-6
        # plain values keep their normal handling
        assert opts['rSpot'] == 2 and isinstance(opts['rSpot'], int)
        assert opts['output_path'] == 'default'

    def test_a_config_that_is_not_a_mapping_is_rejected(self, tmp_path):
        (tmp_path / 'bad.yaml').write_text('- just\n- a list\n')
        with pytest.raises(ValueError, match='mapping'):
            load_config(tmp_path / 'bad.yaml')


class TestOverrides:
    """--set is what turns a sweep into a shell loop, so the types have to
    survive the trip through the command line."""

    def test_values_keep_their_json_types(self):
        opts = apply_overrides({'rTheta': 2}, ['rTheta=5', 'save_data=false',
                                               'voxel_size=[0.28,0.28,0.7]'])
        assert opts['rTheta'] == 5 and isinstance(opts['rTheta'], int)
        assert opts['save_data'] is False
        assert opts['voxel_size'] == [0.28, 0.28, 0.7]

    def test_plain_words_stay_strings(self):
        # 'default' is not valid json, and it is a real value pciSeq uses
        assert apply_overrides({}, ['output_path=default'])['output_path'] == 'default'

    def test_the_original_is_not_mutated(self):
        base = {'rTheta': 2}
        apply_overrides(base, ['rTheta=99'])
        assert base['rTheta'] == 2

    def test_a_malformed_pair_is_rejected(self):
        with pytest.raises(ValueError, match='key=value'):
            apply_overrides({}, ['rTheta'])


class TestLoaders:

    def test_reference_can_be_transposed(self, inputs):
        df = read_scrnaseq({'path': str(inputs / 'ref.csv'), 'transpose': True})
        assert df.index.tolist() == ['GeneA', 'GeneB']       # genes as rows
        assert df.columns.tolist() == ['Type_1', 'Type_2']

    def test_reference_is_optional(self):
        assert read_scrnaseq(None) is None

    def test_spots_are_renamed_and_filtered(self, inputs):
        df = read_spots({
            'path': str(inputs / 'spots.parquet'),
            'rename': {'discriminability': 'score', 'Gene': 'gene_name',
                       'z_stack': 'z_plane'},
            'filter': 'score > 3.0 and intensity > 0.5',
        })
        assert 'gene_name' in df.columns and 'score' in df.columns
        # row 0 passes both, row 2 fails intensity, rows 1 and 3 fail score
        assert df.gene_name.tolist() == ['GeneA', 'GeneB']
        assert df.score.tolist() == [5.0, 9.0]

    def test_spots_need_no_rename_or_filter(self, inputs):
        assert len(read_spots({'path': str(inputs / 'spots.parquet')})) == 4

    def test_masks_from_npz_by_key(self, inputs):
        coo = read_masks({'path': str(inputs / 'seg.npz'), 'key': 'seg'})
        assert len(coo) == 2
        assert coo[0].shape == (8, 8)
        assert set(np.unique(coo[0].toarray())) == {0, 1}

    def test_npz_with_several_arrays_needs_a_key(self, inputs):
        with pytest.raises(ValueError, match='say which one'):
            read_masks({'path': str(inputs / 'seg.npz')})

    def test_masks_from_npy(self, inputs):
        assert len(read_masks({'path': str(inputs / 'seg.npy')})) == 2

    def test_a_single_plane_becomes_a_one_element_list(self, tmp_path):
        np.save(tmp_path / 'flat.npy', np.zeros((8, 8), dtype=np.uint16))
        assert len(read_masks({'path': str(tmp_path / 'flat.npy')})) == 1


class TestBuildRun:

    def test_a_whole_config_resolves(self, inputs):
        cfg = {
            'scrnaseq': {'path': str(inputs / 'ref.csv'), 'transpose': True},
            'spots': {'path': str(inputs / 'spots.parquet'),
                      'rename': {'discriminability': 'score', 'Gene': 'gene_name'}},
            'masks': {'path': str(inputs / 'seg.npz'), 'key': 'seg'},
            'opts': {'rTheta': 2},
        }
        spots, coo, ref, opts = build_run(cfg, ['rTheta=5'])
        assert len(spots) == 4
        assert len(coo) == 2
        assert ref.shape == (2, 2)
        assert opts == {'rTheta': 5}

    def test_missing_sections_say_which_one(self, inputs):
        with pytest.raises(ValueError, match="'masks'"):
            build_run({'spots': {'path': str(inputs / 'spots.parquet')}})


class TestCommandLine:

    def test_dry_run_prints_settings_without_reading_data(self, tmp_path, capsys):
        # the paths deliberately do not exist: a dry run must not touch them
        (tmp_path / 'c.yaml').write_text(
            'spots: {path: /nope/spots.csv}\n'
            'masks: {path: /nope/seg.npy}\n'
            'opts: {rTheta: 2}\n')

        assert main(['run', str(tmp_path / 'c.yaml'), '--dry-run',
                     '--set', 'rTheta=9']) == 0
        assert json.loads(capsys.readouterr().out) == {'rTheta': 9}

    def test_a_bad_config_exits_nonzero_without_a_traceback(self, tmp_path):
        (tmp_path / 'c.yaml').write_text('opts: {rTheta: 2}\n')   # no spots
        assert main(['run', str(tmp_path / 'c.yaml')]) == 1

    def test_run_is_a_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(['run', 'x.yaml', '--set', 'a=1'])
        assert args.command == 'run' and args.set == ['a=1']
