"""Command line entry point: run pciSeq from a config file.

    pciseq run analysis.yaml
    pciseq run analysis.yaml --set rTheta=5 --set mrf_beta=0
    pciseq run analysis.yaml --dry-run

The config says where the three inputs live and how to tidy them up, then what
settings to fit with. It replaces the hand written run_app_*.py scripts, which
all did the same four things (load a reference, load spots, load a
segmentation, call fit) and differed only in paths, a couple of column renames
and a filter.

Anything too odd to express here belongs in a small prep script that writes out
tidy inputs, rather than growing this file.

Config, everything except the paths optional::

    scrnaseq:
      path: scRNAseq.csv
      index_col: 0          # column holding the gene names, name or position
      sep: "\\t"             # for tsv
      transpose: false      # true if the file is cells x genes

    spots:
      path: spots.parquet
      rename: {discriminability: score, Gene: gene_name}
      filter: "score > 3.0 and intensity > 0.15"

    masks:
      path: gcamp_seg.npz
      key: seg              # only for npz, which holds several arrays

    opts:
      rTheta: 2
      save_data: true
      output_path: /path/to/output
"""
import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# PyYAML follows YAML 1.1, where a float in exponent form only parses if the
# exponent carries a sign: 1.0e-6 is a number but 1.0e6 is the string "1.0e6".
# That is a nasty way to lose a run, since rTheta: 1.0e6 would reach the model
# as text. This resolver adds the unsigned exponent form, which is what YAML
# 1.2 and json do and what anybody writing a config expects. It requires an
# exponent, so plain integers and decimals keep their normal handling.
_EXPONENT_FLOAT = re.compile(r'^[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)[eE][-+]?[0-9]+$')


def _yaml_loader():
    """safe_load, but numbers like 1.0e6 come out as numbers."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_implicit_resolver('tag:yaml.org,2002:float', _EXPONENT_FLOAT, list('-+0123456789.'))
    return Loader


def load_config(path: str) -> Dict[str, Any]:
    """Read a yaml or json config."""
    import yaml  # yaml is a superset of json, so it reads both

    with open(path) as fid:
        cfg = yaml.load(fid, Loader=_yaml_loader())
    if not isinstance(cfg, dict):
        raise ValueError(f'{path} should hold a mapping, got {type(cfg).__name__}')
    return cfg


def apply_overrides(opts: Dict[str, Any], overrides) -> Dict[str, Any]:
    """Fold --set key=value pairs into the opts.

    Values are read as json so numbers, booleans and lists come out as the
    right type, and anything that is not valid json is kept as a string. This
    is what makes a sweep a shell loop rather than a pile of config files.
    """
    opts = dict(opts)
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f'--set wants key=value, got {item!r}')
        key, _, raw = item.partition('=')
        try:
            opts[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            opts[key.strip()] = raw
    return opts


def read_scrnaseq(spec: Dict[str, Any]):
    """The single cell reference, as genes x cell types.

    Files come in both orientations and with the gene names in a column rather
    than the index, hence index_col and transpose.
    """
    if spec is None:
        return None

    path = os.path.expanduser(spec['path'])
    df = pd.read_csv(path, sep=spec.get('sep', ','),
                     index_col=spec.get('index_col', 0))
    if spec.get('transpose', False):
        df = df.T
    logger.info('reference: %d genes x %d classes from %s',
                df.shape[0], df.shape[1], path)
    return df


def read_spots(spec: Dict[str, Any]) -> pd.DataFrame:
    """The spots table, renamed and filtered as the config asks.

    filter is a pandas query string, so several conditions go in one line
    joined with 'and'.
    """
    path = os.path.expanduser(spec['path'])
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=spec.get('sep', ','))

    if spec.get('rename'):
        df = df.rename(columns=spec['rename'])

    if spec.get('filter'):
        before = len(df)
        df = df.query(spec['filter'])
        logger.info('spots: %d of %d kept by %r', len(df), before, spec['filter'])

    logger.info('spots: %d rows from %s', len(df), path)
    return df


def read_masks(spec: Dict[str, Any]):
    """The segmentation, one sparse matrix per z plane.

    Accepts .npy holding the stack, or .npz holding several arrays, in which
    case 'key' says which one. A 2D array is treated as a single plane.
    """
    from scipy.sparse import coo_matrix

    path = os.path.expanduser(spec['path'])
    loaded = np.load(path, allow_pickle=False)

    if hasattr(loaded, 'files'):  # npz
        key = spec.get('key')
        if key is None:
            if len(loaded.files) != 1:
                raise ValueError(
                    f'{path} holds {loaded.files}, say which one with key:')
            key = loaded.files[0]
        arr = loaded[key]
    else:
        arr = loaded

    if arr.ndim == 2:
        arr = arr[None, :, :]
    logger.info('masks: %d plane(s) of %d x %d from %s',
                arr.shape[0], arr.shape[1], arr.shape[2], path)
    return [coo_matrix(plane) for plane in arr]


def build_run(cfg: Dict[str, Any], overrides=None):
    """Turn a config into the four arguments fit() wants."""
    for key in ('spots', 'masks'):
        if key not in cfg:
            raise ValueError(f"config needs a '{key}' section")

    spots = read_spots(cfg['spots'])
    coo = read_masks(cfg['masks'])
    scrnaseq = read_scrnaseq(cfg.get('scrnaseq'))
    opts = apply_overrides(cfg.get('opts', {}), overrides)
    return spots, coo, scrnaseq, opts


def cmd_run(args) -> int:
    from pciSeq.app import fit
    from pciSeq.src.core.logger import setup_logger

    setup_logger()
    cfg = load_config(args.config)

    if args.dry_run:
        # resolve the opts without touching the data, so a typo in a sweep
        # shows up in a second rather than after the inputs have loaded
        opts = apply_overrides(cfg.get('opts', {}), args.set)
        print(json.dumps(opts, indent=2, default=str))
        return 0

    spots, coo, scrnaseq, opts = build_run(cfg, args.set)
    logger.info('settings: %s', json.dumps(opts, default=str))
    fit(spots=spots, coo=coo, scRNAseq=scrnaseq, opts=opts)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from pciSeq._version import __version__

    parser = argparse.ArgumentParser(
        prog='pciseq',
        description='Cell typing for spatial transcriptomics.',
    )
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser(
        'run', help='fit a dataset described by a config file',
        description='Run pciSeq on the inputs named in a yaml or json config.',
        epilog="example: pciseq run analysis.yaml --set rTheta=5",
    )
    run.add_argument('config', help='yaml or json config file')
    run.add_argument(
        '--set', action='append', metavar='KEY=VALUE',
        help='override one setting, repeatable. Values are read as json, so '
             'rTheta=5 is a number and save_data=false is a boolean.')
    run.add_argument(
        '--dry-run', action='store_true',
        help='print the resolved settings and stop, without loading any data')
    run.set_defaults(func=cmd_run)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as err:
        logger.error('%s', err)
        return 1


if __name__ == '__main__':
    sys.exit(main())
