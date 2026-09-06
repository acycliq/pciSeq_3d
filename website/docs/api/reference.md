# API reference

::: warning Auto-generated
This page is generated from the pciSeq source by `website/gen_api.py`.
Edit the docstrings in the source, not this file.
:::

Everything here is reachable as `pciSeq.<name>` (plus `VarBayes`, the
model object that [`fit`](#fit) and [`cell_type`](#cell-type) build and
return). The main entry point is [`fit`](#fit). For what the output
DataFrames hold and worked examples of the model attributes, see
[Working with results](./working-with-results).

## `fit`

`pciSeq.app.fit`

```python
fit(*args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]
```

Main entry point for pciSeq cell typing analysis.

``spots`` and ``coo`` may be given as the first two positional arguments or as
keywords. Keyword form is preferred.

**Parameters**

- **`spots`** *(pd.DataFrame)*
  The spots to assign. Needs the columns 'gene_name', 'x' and 'y', plus 'z_plane' for 3D data. Optional 'score' and 'intensity' columns from the spot caller are carried through to geneData; they default to 1.0 and nothing in the model reads them.
- **`coo`** *(list of scipy.sparse.coo_matrix)*
  The label image, one sparse matrix per z-plane. A list with more than one plane is treated as 3D.
- **`scRNAseq`** *(pd.DataFrame)*
  Single-cell reference data used to annotate the cell types, genes by cell classes. Required.
- **`opts`** *(dict, optional)*
  Any config values you want to override, e.g. {'max_iter': 500}. See the configuration page for the full list of keys and their defaults.

**Returns**

- **`cellData`** *(pd.DataFrame)*
  One row per cell: the class probabilities, the gene counts and the cell geometry.
- **`geneData`** *(pd.DataFrame)*
  One row per spot: which cell it was assigned to and with what probability.

**Raises**

- **`ValueError`**
  If spots or coo are missing or malformed.
- **`RuntimeError`**
  If cell typing fails. Not converging is not a failure on its own: the loop runs to max_iter, logs the convergence status and returns its results.

**Notes**

Positional and keyword forms can be mixed; the keywords win if both are given.


## `cell_type`

`pciSeq.app.cell_type`

```python
cell_type(cells: pd.DataFrame, spots: pd.DataFrame, scRNAseq: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, VarBayes]
```

Perform cell typing using Variational Bayes algorithm.

**Parameters**

- **`cells`** *(pd.DataFrame)*
  Preprocessed cell data containing cell locations and boundaries
- **`spots`** *(pd.DataFrame)*
  Preprocessed spot data containing gene expressions and coordinates
- **`scRNAseq`** *(pd.DataFrame)*
  Single-cell RNA sequencing reference data, genes by cell classes. Required.
- **`config`** *(Dict[str, Any])*
  Configuration dictionary containing algorithm parameters

**Returns**

- **`Tuple[pd.DataFrame, pd.DataFrame, VarBayes]`**
  - cellData: DataFrame containing cell typing results - geneData: DataFrame containing gene assignment results - varBayes: The fitted VarBayes model instance

**Raises**

- **`ValueError`**
  If input data is invalid or incompatible
- **`RuntimeError`**
  If cell typing algorithm fails to converge


## `stage_data`

`pciSeq.src.preprocess.main.stage_data`

```python
stage_data(spots: pd.DataFrame, coo: List[coo_matrix], cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict]]
```

Process spots and label images for cell typing analysis.

**Parameters**

- **`spots`** *(pd.DataFrame)*
  Spot data with columns: ['gene_name', 'x', 'y', 'z_plane']
- **`coo`** *(List[coo_matrix])*
  List of sparse matrices containing cell segmentation
- **`cfg`** *(Dict)*
  Configuration dictionary with processing parameters

**Returns**

- **`cells`** *(pd.DataFrame)*
  Cell properties including position and size
- **`borders_future`** *(Future)*
  Future resolving to (cell_boundaries, cell_boundaries_list). Border extraction runs in the background and only blocks when .result() is called.
- **`processed_spots`** *(pd.DataFrame)*
  Processed spots with cell assignments
- **`label_map`** *(Optional[Dict])*
  Label remapping if labels were reordered


## `attach_to_log`

`pciSeq.src.core.logger.attach_to_log`

```python
attach_to_log()
```

exists only for backwards compatibility.
Replaced by setup_logger


## `setup_logger`

`pciSeq.src.core.logger.setup_logger`

```python
setup_logger(level=None)
```

Configure pciSeq logging with colored console output.

WARNING: This function clears all existing root logger handlers and replaces
them with pciSeq's own handler. If pciSeq is embedded inside a larger
application that has its own logging setup, do NOT call this function —
the parent application's handlers will be wiped out. Only call setup_logger()
when pciSeq is the top-level application.

Args:
    level: logging level (e.g. logging.DEBUG, logging.INFO). Defaults to INFO.

Returns:
    The configured 'pciSeq' logger instance.


## `to_spatialdata`

`pciSeq.src.core.utils.spatialdata_export.to_spatialdata`

```python
to_spatialdata(cellData: pd.DataFrame, geneData: pd.DataFrame, coo: List[coo_matrix], varBayes, cfg: Dict)
```

Assemble the run into an in-memory SpatialData object.

Args:
    cellData: cell typing results, one row per cell, original labels.
    geneData: spot results, one row per spot, original labels.
    coo: the segmentation, one sparse plane per z, as fit() received it.
    varBayes: the fitted model, read for the arrays the two frames do not
        carry (class posterior, spot probabilities, gene panel, reference).
    cfg: the resolved config. voxel_size and label_map are used here.


## `write_spatialdata`

`pciSeq.src.core.utils.spatialdata_export.write_spatialdata`

```python
write_spatialdata(cellData, geneData, coo, varBayes, cfg, out_dir: str) -> str
```

Write the run as a SpatialData zarr store and return its path.


## `add_image`

`pciSeq.src.core.utils.spatialdata_export.add_image`

```python
add_image(store_path: str, img, name: str='background', voxel_size=None, scale_factors=(2, 2, 2)) -> None
```

Add a background image to an existing SpatialData store.

Works like inserting into a database: the store already holds the run
(spots, labels, tables) and this writes one more element into it, without
touching anything else. Kept separate from write_spatialdata because fit()
never sees the image, the same reason stage_image is its own entry point.

Args:
    store_path: path to an existing spatialdata.zarr written by pciSeq.
    img: the image to add. A 2D array (H, W), a 3D stack (Z, H, W), or a
        3D stack with channels (Z, H, W, C). Same shapes stage_image takes.
    name: element name inside the store.
    voxel_size: [x, y, z]. Left as None it is read from the store's own
        provenance, so the microns transform automatically matches the
        other elements. Pass it only for a store that lacks the metadata.
    scale_factors: the multiscale pyramid, each level relative to the one
        before. The default (2, 2, 2) gives four scales. None writes a
        single scale.


## `add_boundaries`

`pciSeq.src.core.utils.spatialdata_export.add_boundaries`

```python
add_boundaries(store_path: str, labels_name: str='cell_labels', name: str='cell_boundaries', voxel_size=None) -> None
```

Add per-plane cell boundary polygons to an existing SpatialData store.

Same insert pattern as add_image, but this one needs no data at all: the
boundaries are derived from the segmentation, and the store already holds
the segmentation as the labels element. They are extracted here with the
same chain code tracing the pipeline uses, so what goes in matches what
pciSeq would have drawn.

Shapes in SpatialData are strictly 2D, so a 3D run gets one shapes element
per plane, cell_boundaries_plane_000 and so on, each a set of polygons
indexed by the cell label. A cell spanning 12 planes appears as 12
polygons, one per element, all under its own label, which is exactly how
pciSeq thinks of boundaries anyway. A 2D run gets a single element.

Args:
    store_path: path to an existing spatialdata.zarr written by pciSeq.
    labels_name: the labels element to trace.
    name: element name, used as a prefix on 3D runs.
    voxel_size: [x, y, z]. Left as None it is read from the store's own
        provenance, same as add_image.


## `stage_image`

`pciSeq.src.tiling.stage_image.stage_image`

```python
stage_image(img, out_dir=None, zoom_levels=8, name=None, description=None, plane_prefix='plane_', use_buffer=True, tint=None, progress=True)
```

Turn an image (or z-stack) into an MBTiles file the viewer can read.

This builds the tiled, multi-resolution background that pciSeq Viewer uses
as its slippy-map base layer. Give it a single 2D image or a whole 3D stack
and it writes one `.mbtiles` file.

**Parameters**

- **`img`** *(np.ndarray or str)*
  The image to tile: a 2D array (H, W), a 3D stack (Z, H, W), a 3D stack with channels (Z, H, W, C), or a path to a 2D image file (legacy).
- **`out_dir`** *(str, optional)*
  Directory for the `.mbtiles` file. Defaults to the system temp directory.
- **`zoom_levels`** *(int, optional)*
  Number of zoom levels to produce. Default is 8.
- **`name`** *(str, optional)*
  Short identifier for the dataset. Also used as the output filename, e.g. `name="S10_gcamp_10"` writes `S10_gcamp_10.mbtiles`. If empty, the file is named `output.mbtiles`.
- **`description`** *(str, optional)*
  Longer description of the dataset.
- **`plane_prefix`** *(str, optional)*
  Prefix for the per-plane names. Default is "plane_".
- **`use_buffer`** *(bool, optional)*
  If True (the default) the tiles are built in memory and inserted straight into the MBTiles database. If False they are written to disk first, which uses less memory but more disk I/O.
- **`tint`** *(str, optional)*
  Hex colour like "#00FF00" the viewer uses to tint this grayscale layer. If omitted, the layer is shown in plain grayscale.
- **`progress`** *(bool, optional)*
  If True (the default) show two per-plane tqdm bars, one for tiling and one for the db writing. Set to False for headless/quiet runs.

**Returns**

- **`str`**
  Path to the created `.mbtiles` file.

**Notes**

Requires libvips. If it is not installed, `pciSeq.stage_image()` falls back
to a stub that only logs a warning.


## `tile_maker`

`pciSeq.src.tiling.stage_image.tile_maker`

```python
tile_maker(img, zoom_levels=8, out_dir='./tiles', plane_prefix='plane_', progress_bar=None)
```

Makes a pyramid of tiles from an image.

Args:
    img: One of:
        - str: path to a 2D image file (TIFF, PNG, JPEG, etc.)
        - numpy array (H, W): single 2D grayscale image
        - numpy array (Z, H, W): 3D stack of grayscale images
        - numpy array (Z, H, W, C): 3D stack with channels
    zoom_levels: (int) Number of zoom levels to produce. Default is 8.
    out_dir: (str) Output folder for the tile pyramid. Will be deleted and recreated if exists.
    plane_prefix: (str) Prefix for plane subdirectories when processing 3D images.
                  Default is "plane_" resulting in "plane_0", "plane_1", etc.
    progress_bar: (tqdm, optional) if given, ticked once per plane instead of
                  logging a per-plane line. Used by stage_image to drive its bars.

Returns:
    dict with keys:
        - 'original_dims': [width, height] of the original input image
        - 'num_planes': number of planes processed
        - 'zoom_levels': number of zoom levels


## `VarBayes`

`pciSeq.src.core.main.VarBayes`

```python
VarBayes(cells_df: pd.DataFrame, spots_df: pd.DataFrame, scRNAseq: pd.DataFrame, config: Dict[str, Any])
```

Implements Variational Bayes algorithm for spatial transcriptomics analysis.

This class performs cell type assignment and spot-to-cell mapping using a
probabilistic model with variational inference.

Args:
    cells_df: DataFrame containing cell information
    spots_df: DataFrame containing spot information
    scRNAseq: Single-cell RNA sequencing reference data
    config: Configuration dictionary containing algorithm parameters

::: tip Obtaining a fitted instance
`VarBayes` is not instantiated directly in normal use. [`fit`](#fit) and [`cell_type`](#cell-type) construct and run it. `cell_type` returns the fitted instance; `fit` does not, but when `save_data=True` (the default) the fitted model is serialised to `<output_path>/data/debug/pciSeq.pickle` (`output_path` defaults to a temporary directory). The attributes and methods below operate on a loaded instance; [Working with results](./working-with-results) walks through the main ones with examples.

```python
import pandas as pd

obj = pd.read_pickle('<output_path>/data/debug/pciSeq.pickle')

obj.metadata
obj.check_cell(my_label=42, user_class='Astro')
```
:::

### Attributes

- <a id="metadata"></a>**`metadata`** *(dict)*
  Provenance recorded when the model is built, saved alongside the results so a run can be traced back to the code that produced it. Contains:
    - `version`: the pciSeq version
    - `branch`: the git branch
    - `commit`: the git commit hash
    - `build_date`: the package build date
    - `created_at`: a UTC timestamp for when the run was created

### Methods

#### `check_spot`

```python
check_spot(spot_id)
```

#### `check_cell`

```python
check_cell(my_label, user_class, top_n=10, show_plot=True)
```

#### `read_tsv`

```python
read_tsv(filepath)
```

#### `heatmap_counts_per_class`

```python
heatmap_counts_per_class()
```

Display the interactive heatmap.

#### `cells.gene_reads_per_class`

```python
cells.gene_reads_per_class()
```

Calculate total (weighted by class prob) gene reads for each class.

Returns:
    np.ndarray: Shape (G, K) total reads per class and gene

#### `cells.mean_gene_reads_per_class`

```python
cells.mean_gene_reads_per_class()
```

Calculate the average gene reads for each cell class/type in a soft clustering setup.

In soft clustering, each cell belongs to multiple classes with probabilities $w_{ck}$.
The average number of reads for gene $g$ in class $k$ is computed as:

$$\overline{r}_{gk} = \frac{\sum_{c=1}^{C} x_{cg} \cdot w_{ck}}{\sum_{c=1}^{C} w_{ck}}$$

Where:
    - $x_{cg}$: Number of reads for gene $g$ in cell $c$
    - $w_{ck}$: Probability that cell $c$ belongs to class $k$
    - The numerator is the total weighted sum of reads for gene $g$ in class $k$
    - The denominator is the total probability mass of class $k$

Returns:
    np.ndarray: Shape (G, K), where:
        G = number of genes
        K = number of cell classes/types
