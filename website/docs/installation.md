# Installation

Install straight from the repository:

```bash
pip install git+https://github.com/acycliq/pciSeq_3d.git@dev_3d
```

For development, clone it and install in editable mode so edits to the source are picked up
without reinstalling:

```bash
git clone https://github.com/acycliq/pciSeq_3d.git
cd pciSeq_3d
pip install -e .
```

Python 3.10 or later. The dependencies are pulled in by pip. The image tiling functions need
libvips, which arrives with the `pyvips[binary]` wheel; on a platform with no such wheel the
rest of the package still imports and only `tile_maker` and `stage_image` raise, with a
message pointing at the libvips install page. `read_tiles` does not need libvips.

## Versions

```python
import pciSeq
pciSeq.__version__     # '0.0.66.dev0'
pciSeq.__commit__      # short git hash, eg '6db6dc8'
pciSeq.__branch__      # 'dev_3d'
pciSeq.__build_date__  # 'unknown' on a source checkout
```

The version string lives in `pciSeq/_version.py`. `__commit__` and `__branch__` are read
from git when the package is imported from a checkout, otherwise from the values baked in
when it was built. On the command line, `pciseq --version` prints the version string.

The model changes between commits. Two results are directly comparable only when the
commit matches, or when the changes in between are known. Every run records it, see below.

## Run provenance

Every run records the code that produced it, so a result can be traced back to it:

```python
varBayes.metadata
# {'version':    '0.0.66.dev0',
#  'branch':     'dev_3d',
#  'commit':     '6db6dc8',
#  'build_date': 'unknown',
#  'created_at': '2026-09-04T12:47:51Z'}
```

`varBayes` is the fitted model returned by [`cell_type`](api/reference.md#cell-type), or
loaded from the pickle when `save_data` is on. The same dictionary is written into the saved
output in two more places, so it survives without the live object:

- `diagnostics.db`, under the `pciSeq_provenance` key of the `metadata` table:

  ```python
  import sqlite3, json

  con = sqlite3.connect('<output_path>/pciSeq/data/viewer_data/diagnostics/diagnostics.db')
  row = con.execute(
      "select value from metadata where key = 'pciSeq_provenance'").fetchone()
  print(json.loads(row[0]))
  ```

- the [SpatialData store](api/spatialdata-store.md#provenance), as `sdata.attrs['pciseq']`,
  together with the resolved run config, without `label_map`.

`commit` is the one to quote when comparing two runs.
