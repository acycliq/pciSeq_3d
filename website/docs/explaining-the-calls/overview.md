
# Explaining the calls

pciSeq returns a probability for every cell type of every cell and for every candidate
cell of every spot. This section shows how to find out why the model arrived at those
probabilities: which genes, which prior and which neighbours made a cell a given type,
and which terms sent a spot to a cell or to the background.

Two methods of the fitted model answer these questions:

| question | method | page |
| --- | --- | --- |
| Why was this cell assigned this type, and not another? | [`check_cell`](../api/reference.md#check-cell) | [Why a cell got its type](why-a-cell-got-its-type.md) |
| Why was this spot assigned to this cell, or to the background? | [`check_spot`](../api/reference.md#check-spot) | [Why a spot went where it went](why-a-spot-went-where-it-went.md) |

Both read the values the model used in its last iteration, so what they show agrees with
the probabilities in `cellData` and `geneData`.

## The fitted model

`check_cell` and `check_spot` are methods of the fitted model, a `VarBayes` object.
[`fit`](../api/reference.md#fit) does not return it, but with `save_data=True`, the
default, it is saved to `<output_path>/pciSeq/data/debug/pciSeq.pickle`:

```python
import pandas as pd

obj = pd.read_pickle('<output_path>/pciSeq/data/debug/pciSeq.pickle')
```

[`cell_type`](../api/reference.md#cell-type) returns the model directly. A model saved by
an earlier version of pciSeq may lack the values these methods read.

## The example data

The examples in this section use the CA1 dataset of
[Qian et al. (2020)](https://doi.org/10.1038/s41592-019-0631-4), which is in the pciSeq
repository: a 2D section of mouse hippocampus with 72,336 spots of 92 genes, a
segmentation of 3,481 cells, and a single-cell reference with 71 cell types. The
following downloads the data, runs pciSeq and loads the fitted model:

```python
import io
import urllib.request

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import pciSeq

url = 'https://raw.githubusercontent.com/acycliq/pciSeq_3d/dev_3d/pciSeq/data/mouse/ca1'

spots = pd.read_csv(f'{url}/iss/spots.csv')
with urllib.request.urlopen(f'{url}/segmentation/label_image.coo.npz') as r:
    coo = load_npz(io.BytesIO(r.read()))

# one column per single cell, the first row holds its cell type
sc = pd.read_csv(f'{url}/scRNA/scRNAseq.csv.gz', header=None, index_col=0, dtype=object)
sc = sc.rename(columns=sc.iloc[0]).iloc[1:].astype(np.uint32)

cellData, geneData = pciSeq.fit(spots=spots, coo=coo, scRNAseq=sc,
                                opts={'output_path': 'ca1_run'})
obj = pd.read_pickle('ca1_run/pciSeq/data/debug/pciSeq.pickle')
```

The run takes under a minute. The reference holds one column per single cell, and
pciSeq averages the columns of each type into the cell type definitions. With the
default settings the figures on the following pages are reproduced exactly.
