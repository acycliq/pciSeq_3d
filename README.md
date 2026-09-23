# pciSeq: Probabilistic Cell typing by In situ Sequencing

pciSeq assigns each RNA spot of an imaging-based spatial transcriptomics experiment to a
cell, and each cell to a cell type. The two assignments are estimated jointly: the type of
a cell depends on the spots inside it, and the cell of a spot depends on the types of the
cells around it. Both come back as probabilities. This branch works in 3D, on a stack of
segmented planes.

https://github.com/user-attachments/assets/ab9867e4-866e-4f81-88e5-359bdf7d7a64

*The live viewer, one picture per iteration. Cells are coloured by the class they are
currently assigned to, and the chart at the bottom right tracks convergence.*

## Inputs and outputs

**Inputs**

- **Spots.** A table of detected RNA spots with gene identity and position (`x`, `y`, and
  `z_plane` for 3D data).
- **Segmentation.** A label image giving the cell each pixel belongs to, typically from a
  DAPI nuclear stain.
- **Cell type definitions.** Mean expression per gene for each cell type, for example from
  a scRNA-seq experiment.

**Outputs**

- **`cellData`**, a probability distribution over the cell types for every cell.
- **`geneData`**, a probability distribution over the candidate parent cells and the
  background for every spot.

Both come back as pandas DataFrames. With `save_data` on they are also written to
`output_path` as tsv and feather files, as a SpatialData zarr store, and the fitted model
is pickled next to them.

## Install

```
pip install git+https://github.com/acycliq/pciSeq_3d.git@dev_3d
```

Python 3.10 or newer.

## Quick start

```python
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import pciSeq

spots = pd.read_csv("spots.csv")                             # gene_name, x, y, z_plane
coo = [coo_matrix(plane) for plane in np.load("masks.npy")]  # one label matrix per plane
scRNAseq = pd.read_csv("scRNAseq.csv", index_col=0)          # genes x cell types

opts = {
    "voxel_size": [0.28, 0.28, 0.7],
    "output_path": "out/run1",
    "realtime_viewer": True,     # watch it run at http://127.0.0.1:5001
}

cellData, geneData = pciSeq.fit(spots=spots, coo=coo, scRNAseq=scRNAseq, opts=opts)
```

## Explaining the calls

A fitted model can be inspected after the run. [`check_cell`](https://acycliq.github.io/pciSeq_3d/api/reference#check-cell) and
[`check_spot`](https://acycliq.github.io/pciSeq_3d/api/reference#check-spot) break a call
into the terms behind it: why a cell was given its class, why a spot was assigned to a
cell. They also serve as diagnostics for miscalls. Both are walked through in the
documentation, in
[How a cell's call was made](https://acycliq.github.io/pciSeq_3d/explaining-the-calls/why-a-cell-got-its-type)
and [How a spot's call was made](https://acycliq.github.io/pciSeq_3d/explaining-the-calls/why-a-spot-got-its-cell).

## Documentation

[Documentation](https://acycliq.github.io/pciSeq_3d/): settings, model, outputs, live
viewer.

## Citation

Qian, X. et al. Probabilistic cell typing enables fine mapping of closely related cell
types in situ. *Nature Methods* 17, 101 to 106 (2020).

## Licence

MIT, see [LICENSE](LICENSE).
