---
title: SpatialData store
description: The zarr store a run writes next to the tsv files, element by element, and how to query it.
---

# SpatialData store

When `save_data` is on, a run writes its results twice: once as tsv files for the
viewer, and once as a [SpatialData](https://scverse-spatialdata.readthedocs.io/)
zarr store at `<output>/data/spatialdata.zarr`. The zarr store is the copy meant
for the scverse tools: open it in napari via napari-spatialdata, or hand it to
squidpy, scanpy or plain anndata.

Everything on this page was written against spatialdata 0.5. The elements and
their columns are produced by `pciSeq/src/core/utils/spatialdata_export.py`; if
that file and this page ever disagree, believe the file.

## Do you need this file?

Maybe not, and that is fine. Everything needed to inspect a run is already in
the tsv files and the viewer. The zarr store is an optional copy for tools that
speak the scverse formats:

- **Just want the results?** Use the viewer or the tsv files and ignore this
  file entirely.
- **Want to look around in 3D?** One command, no python to write. Needs
  [napari-spatialdata](https://spatialdata.scverse.org/projects/napari/) installed
  (`pip install napari-spatialdata`):

  ```bash
  python -m napari_spatialdata view <output>/data/spatialdata.zarr
  ```

- **Want to script against it?** Skip to the [worked example](#worked-example)
  at the bottom of this page. Copy it, change the cell number. That snippet and
  `print(sdata)` are genuinely all the API knowledge an end user needs; the rest
  of this page is reference for when a specific question comes up.

## Opening the store

```python
import spatialdata as sd

sdata = sd.read_zarr('/path/to/output/data/spatialdata.zarr')
print(sdata)
```

```text
SpatialData object, with associated Zarr store: .../spatialdata.zarr
├── Labels
│     └── 'cell_labels': DataArray[zyx] (101, 4412, 6412)
├── Points
│     └── 'transcripts': DataFrame with shape: (<Delayed>, 10) (3D points)
└── Tables
      ├── 'cells': AnnData (26283, 207)
      └── 'reference': AnnData (38, 207)
with coordinate systems:
    ▸ 'microns', with elements:
        cell_labels (Labels), transcripts (Points)
```

The objects describe themselves: `print(sdata)` lists every element,
`print(sdata.tables['cells'])` lists every column and key the table carries. When
in doubt, print the thing.

## The elements

### `cell_labels` (Labels)

The segmentation, one uint32 label image over `(z, y, x)`. The value at a voxel
is the `Cell_Num` of the cell covering it, in the original label numbering. The
element carries a transformation into the `microns` coordinate system, so
napari-spatialdata renders it at true scale next to the transcripts.

### `transcripts` (Points)

One row per spot. Backed by parquet and lazy: call `.compute()` once to get a
pandas DataFrame.

| Column | Type | Meaning |
| --- | --- | --- |
| `x`, `y` | float32 | Spot position in pixels. |
| `z` | float32 | The z-plane index. Not the scaled `z` of `geneData`: the store keeps the plane, which is what the label image indexes. |
| `gene_name` | str | Gene name. Registered as the feature key, so `sdata.aggregate` style joins work. |
| `spot_id` | int | Spot id, same as `geneData.spot_id`. |
| `neighbour` | int | `Cell_Num` of the most likely parent cell for this spot. |
| `neighbour_prob` | float32 | Probability of that assignment. |
| `omp_score`, `omp_intensity` | float32 | Per-spot OMP diagnostics, 3D runs only. |
| `is_hard_misread` | bool | The spot's most likely parent is the background class. |

The ragged per-spot columns of `geneData` (`neighbour_array`, `neighbour_prob`
lists) are dropped on purpose: parquet handles ragged lists badly, and the
useful scalar, the probability of the assigned cell, is kept instead.

### `tables['cells']` (AnnData)

The cell typing results, one row per cell. This is the zarr twin of
`cellData.tsv`, reorganized into the anndata layout.

| Where | What |
| --- | --- |
| `obs` index | `Cell_Num`, **as a string**. `.loc[12265]` fails, `.loc['12265']` works. |
| `obs['cell_num']` | The same number as an int, for convenience. |
| `obs['region']`, `obs['class_name']`, `obs['class_prob']` | The annotating table wiring, the argmax class, and its probability. |
| `obs['is_pinned']`, `obs['pinned_at_iteration']` | Tie-freeze state: whether the class was pinned, and at which iteration. `-1` means never pinned. |
| `obsm['spatial']` | Centroid `(X, Y, Z)`. `X`, `Y` equal `cellData`'s; `Z` is in plane units, see the tip below. |
| `obsm['class_prob']` | The full class posterior, shape `(n_cells, n_classes)`. Columns line up with `uns['class_names']`. |
| `uns['class_names']` | Class names, the reference taxonomy plus `'Zero'`, the background class, appended last. |
| `X` | Sparse cell by gene count matrix, csr. Counts are soft, so entries can be fractional, exactly like `cellData.CellGeneCount`. |
| `var_names` | The gene panel, matching `cellData.Genenames`. |

::: tip Z units differ from cellData on purpose
`cellData.Z` is scaled by `voxel_size[2] / voxel_size[0]` so that distances are
Euclidean. The store's `obsm['spatial']` divides it back, because the store also
holds the label image, which indexes by plane. To go from the store to the
`cellData` value: `Z * voxel_size[2] / voxel_size[0]`. With isotropic voxels the
two are the same number.
:::

The table is a proper annotating table (`region = 'cell_labels'`,
`instance_key = 'cell_num'`), so scanpy and napari-spatialdata can join it
against the segmentation automatically.

### `tables['reference']` (AnnData)

The scRNAseq reference the run was called against, transposed to classes as rows
and genes as columns. It is there so the store says what produced the calls:
`X[i, j]` is the reference expression of gene `j` in class `i`.

## Provenance

`sdata.attrs['pciseq']` records what produced the store: the code (`branch`,
`commit`, `build_date`, `version`), when the store was written (`created_at`),
the `spatialdata_version`, and the full resolved run `config`.

## Mapping from the tsv columns

| tsv column | in the store |
| --- | --- |
| `cellData.Cell_Num` | `tables['cells'].obs` index, `obs['cell_num']`, label image values |
| `cellData.X`, `Y`, `Z` | `obsm['spatial']` (Z in plane units) |
| `cellData.ClassName`, `Prob` | `obsm['class_prob']` + `uns['class_names']`; argmax also flattened into `obs` |
| `cellData.Genenames`, `CellGeneCount` | `X` + `var_names` |
| `geneData` columns | `transcripts` points, see the table above |

## Worked example

One cell, looked up by `Cell_Num`, across the whole store:

```python
import numpy as np

adata = sdata.tables['cells']
cid = 12265
pos = int(np.flatnonzero(adata.obs.index.values == str(cid))[0])

obs = adata.obs.iloc[pos]        # class_name, class_prob, is_pinned, ...
xyz = adata.obsm['spatial'][pos] # centroid, Z in plane units
prob = adata.obsm['class_prob'][pos]
names = np.asarray(adata.uns['class_names'])
top3 = np.argsort(-prob)[:3]
[(names[i], round(float(prob[i]), 3)) for i in top3]
# [('037 DG Glut', 1.0), ('038 DG-PIR Ex IMN', 0.0), ('017 CA3 Glut', 0.0)]

counts = np.asarray(adata.X[pos].todense()).ravel()
adata.var_names[np.argmax(counts)], counts.max()   # top gene in this cell
# ('Sema5a', 6.0)
```

Spots whose most likely parent is this cell, and the label image check at the
centroid:

```python
tr = sdata['transcripts'].compute()
tr[tr.neighbour == cid]

x, y, z = adata.obsm['spatial'][pos]
int(sdata['cell_labels'][int(z), int(y), int(x)])   # == cid
```

::: warning Two ways to count a cell's spots
`len(tr[tr.neighbour == cid])` (hard argmax assignment) and
`adata.X[pos].sum()` (the model's soft counts) are close but not identical.
Both are correct; they answer different questions. The tsv files and the store
agree with each other on both quantities.
:::
