---
title: Command line
description: Run pciSeq from a config file with the pciseq command, override settings for a single run, and drive parameter sweeps.
---

# Command line

`pciseq` runs a dataset described by a config file, so a run is a file you can
keep and share rather than a script you edit. It does the same thing as calling
[`fit`](./reference#fit) yourself, with the file loading built in.

```bash
pciseq run analysis.yaml
```

## The config file

Four sections. `spots` and `masks` are required, `scrnaseq` and `opts` are
optional.

```yaml
scrnaseq:
  path: scRNAseq.csv
  index_col: 0                 # column holding the gene names
  transpose: false             # true if the file is classes x genes

spots:
  path: spots.parquet
  rename: {discriminability: score, Gene: gene_name, z_stack: z_plane}
  filter: "score > 3.0 and intensity > 0.15"

masks:
  path: gcamp_seg.npz
  key: seg                     # only needed for npz holding several arrays

opts:
  voxel_size: [1, 1, 3.46]
  rTheta: 2
  save_data: true
  output_path: /path/to/output
```

Everything under `opts` goes straight to `fit`, so the keys are the ones on the
[configuration page](./configuration). Anything you leave out keeps its default.

### scrnaseq

The single cell reference, as genes by cell types. Written to disk both ways
round, hence `transpose`.

| key | meaning |
|-----|---------|
| `path` | csv or tsv file |
| `index_col` | column holding the gene names, by name or position. Default `0` |
| `sep` | column separator. Default `,`, use `"\t"` for tsv |
| `transpose` | `true` when the file has classes as rows |

### spots

| key | meaning |
|-----|---------|
| `path` | `.parquet`, or csv/tsv |
| `sep` | separator for text files |
| `rename` | old name to new name, for files whose columns are not what pciSeq expects |
| `filter` | a pandas query string. Several conditions join with `and` |

pciSeq needs `gene_name`, `x`, `y` and, for 3D, `z_plane`. `rename` is how you
get there without editing the data.

Keeping `filter` here rather than baking it into the file means the raw spots
stay unfiltered on disk and the threshold is visible next to the run it belongs
to.

### masks

| key | meaning |
|-----|---------|
| `path` | `.npy` holding the label volume, or `.npz` |
| `key` | which array to take from an npz. Required when the npz holds more than one |

A `(Z, Y, X)` volume becomes one plane per z. A 2D array is treated as a single
plane.

## Overriding settings for one run

`--set` changes one setting without touching the file. Repeat it as needed.

```bash
pciseq run analysis.yaml --set rTheta=5 --set mrf_beta=1.0
```

Values are read as JSON, so types come out right:

```bash
--set rTheta=5                       # number
--set save_data=false                # boolean
--set voxel_size=[0.28,0.28,0.7]     # list
--set output_path=default            # not valid JSON, so kept as a string
```

A key that is not a real pciSeq setting raises rather than being ignored, so a
typo stops the run instead of quietly doing the wrong thing.

## Sweeps

This is what `--set` is for. One config, a loop, one output directory per run:

```bash
for t in 2 5 15 25; do
  for b in 0 0.5 1.0; do
    pciseq run analysis.yaml \
      --set rTheta=$t --set mrf_beta=$b \
      --set output_path=out/rtheta${t}_beta${b}
  done
done
```

If you later fix a path in `analysis.yaml`, every run in the sweep inherits the
fix. Twelve near-identical config files would not.

Run them one at a time unless you know your memory budget: a large 3D dataset
can hold well over 8 GB per run.

## Checking a config

`--dry-run` prints the settings that would be used and stops, without reading
any data. Useful for catching a typo in a sweep in a second rather than after
the inputs have loaded.

```bash
pciseq run analysis.yaml --set rTheta=5 --dry-run
```

## A note on numbers in YAML

Write large numbers in exponent form as you would expect, `1.0e6`, and they will
be read as numbers. YAML's own rules say an exponent needs a sign, so plain
PyYAML would give you the string `"1.0e6"`; pciSeq handles the unsigned form so
`rTheta: 1.0e6` means what it looks like.

## Doing it from Python instead

The CLI is a convenience, not a separate implementation. This:

```bash
pciseq run analysis.yaml
```

is the same as loading the inputs yourself and calling
[`fit`](./reference#fit). If you want the config handling without the command
line:

```python
from pciSeq.cli import load_config, build_run
from pciSeq import fit

cfg = load_config("analysis.yaml")
spots, coo, scRNAseq, opts = build_run(cfg, overrides=["rTheta=5"])
cellData, geneData = fit(spots=spots, coo=coo, scRNAseq=scRNAseq, opts=opts)
```

## When the config is not enough

The config covers renaming, filtering and picking an array out of an npz, which
is most of what real datasets need. Anything stranger, a bespoke coordinate
transform, merging several files, stitching tiles, belongs in a small script
that writes out tidy inputs, which the config then points at. That keeps the
odd, dataset-specific work visible in its own file instead of spreading through
the config vocabulary.
