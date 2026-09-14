# Running pciSeq

pciSeq is run from Python by calling [`fit`](api/reference.md#fit) with the spots, the
segmentation and the cell type definitions, or from the
[command line](api/command-line.md) with the same inputs listed in a config file.

## From Python

```python
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import pciSeq

spots = pd.read_csv("spots.csv")                          # gene_name, x, y, z_plane
coo = [coo_matrix(plane) for plane in np.load("masks.npy")]  # one label matrix per plane
scRNAseq = pd.read_csv("scRNAseq.csv", index_col=0)        # genes x cell types

opts = {
    "voxel_size": [0.28, 0.28, 0.7],
    "exclude_genes": ["Zbtb20", "Ddit4l"],
    "output_path": "out/run1",
}

cellData, geneData = pciSeq.fit(spots=spots, coo=coo, scRNAseq=scRNAseq, opts=opts)
```

`fit` returns two DataFrames, one row per cell and one row per spot, described in
[Working with results](api/working-with-results.md). With `save_data` on, the default,
the same results are also written under `<output_path>/pciSeq/data/`.

## Inputs

| argument | type | required | content |
| --- | --- | --- | --- |
| `spots` | DataFrame | yes | One row per spot. Columns `gene_name`, `x`, `y`, and `z_plane` for 3D data. `score` and `intensity` are optional; they are carried through to the output and default to 1.0. |
| `coo` | list of `scipy.sparse.coo_matrix` | yes | The segmentation as a label image, one sparse matrix per z-plane, 0 for background. A 3D numpy array of shape `(planes, h, w)` is accepted and converted. |
| `scRNAseq` | DataFrame | yes | Mean expression per gene and cell type, genes as rows and cell types as columns, gene names in the index. Genes absent from the panel are dropped from the spots with a warning. |
| `opts` | dict | no | Settings to override, see [Configuration](api/configuration.md). Anything left out keeps its default. |

Coordinates in `spots` are in pixels of the label image. Spots and segmentation have to
be in the same frame.

## 2D and 3D

The dimensionality is read off `coo`. A single matrix, or a list with one element, is 2D.
A list with more than one plane is 3D.

- **2D.** `z_plane` may be left out of `spots`, it is set to 0. The legacy column name
  `Gene` is accepted for `gene_name`. `voxel_size` stays at its default `[1, 1, 1]`.
- **3D.** `z_plane` is required and indexes the plane in `coo`. `voxel_size` should be
  set to the physical size of a voxel as `[x, y, z]`, since the z step is usually
  coarser than the pixel and the distances along z are wrong without it.
  `remove_flat_cells`, on by default, drops cells that appear on a single plane.
