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
| `coo` | a `scipy.sparse.coo_matrix`, a list of them, or a numpy array | yes | The segmentation as a label image: each pixel holds the label of the cell it belongs to, and background pixels are labelled 0. Pass it as one sparse matrix per z-plane, or as a dense numpy array of shape `(h, w)` or `(planes, h, w)`, which is converted to sparse internally. |
| `scRNAseq` | DataFrame | yes | Mean expression per gene and cell type, genes as rows and cell types as columns, gene names in the index. Spots whose gene is absent from `scRNAseq` are dropped, with a warning. |
| `opts` | dict | no | Settings to override, see [Configuration](api/configuration.md). Anything left out keeps its default. |

Coordinates in `spots` are in pixels of the label image. Spots and segmentation have to
be in the same frame.

## 2D and 3D

The dimensionality is read off `coo`. A single matrix, or a list with one element, is 2D.
A list with more than one plane is 3D.

- **2D.** `z_plane` may be omitted from `spots`; it is then set to 0. The legacy column name
  `Gene` is accepted for `gene_name`. `voxel_size` stays at its default `[1, 1, 1]`.
- **3D.** `z_plane` is required and indexes the plane in `coo`. `voxel_size` should be
  set to the physical size of a voxel as `[x, y, z]`, since the z step is usually
  coarser than the pixel and the distances along z are wrong without it.
  `remove_flat_cells`, on by default, drops cells that appear on a single plane.

## Settings

Every setting and its default is in `pciSeq/config.py`; the
[configuration page](api/configuration.md) is generated from it. `opts` overrides
individual keys, the rest keep their defaults.

The settings are of two kinds:

- Experiment settings describe the data and follow from it: `voxel_size`,
  `exclude_genes`, `output_path`, the viewer options.
- Hyperparameters set the behaviour of the model. They are chosen by the user, usually after
  comparing several values.

## Fine-tuning

The hyperparameters below are listed in the order in which they are typically adjusted.

| Setting | Controls |
| --- | --- |
| [`Inefficiency`](api/configuration.md#inefficiency) | Global scale factor applied to the cell type definitions. Every mean expression value is multiplied by it, which sets the baseline expected number of spots per cell, assuming no further correction is needed. With definitions from scRNA-seq, it is the sensitivity of the spatial assay relative to scRNA-seq. At 0.2, a gene with mean expression 100 in a class has a baseline expected number of about 20 spots in a cell of that class. |
| [`rTheta`](api/configuration.md#rtheta) | Strength of the prior on the cell scale factor $\theta_c$, which rescales the expected counts of every class to the cell's own total count. Low values, towards 1, let the data set $\theta_c$; high values, in practice an order of magnitude above the typical count of a cell, hold $\theta_c$ at 1. A low value lets a near-empty cell be assigned a real class on one or two spots; see [the prior strength](the-model/scale-factors.md#r-theta). |
| [`mrf_beta`](api/configuration.md#mrf-beta) | Strength of the spatial term. Larger values give more weight to the classes of a cell's neighbours. The bonus a class receives is `mrf_beta` times the sum of the neighbours' probabilities for that class, each neighbour weighted by a Gaussian function of its distance; the weights sum to `nNeighbors`. `0` disables the term. Useful where cell types are strongly localised, in layers or regions, as in the hippocampus. |
| [`MisreadDensity`](api/configuration.md#misreaddensity), [`rRho`](api/configuration.md#rrho) | The prior mean of the per-gene misread density, in spots per unit volume, and the strength of that prior. `rRho` counts as that many spots against the gene's background count, so at the default of 1 the prior has very little effect and the density is set by the data. |
| [`cell_type_weights`](api/configuration.md#cell-type-weights) | The prior probability of each class. Types not listed share the remaining probability equally, so the default `{'Zero': 0.5}` gives Zero half the prior mass and splits the other half among the real types. There is no `default` key. |
| [`cell_centroid_prior`](api/configuration.md#cell-centroid-prior), [`cell_cov_prior`](api/configuration.md#cell-cov-prior) | How far the fitted centre and shape of a cell may move from the segmentation. Each is a pseudo-count: the segmentation centroid counts as `cell_centroid_prior` spots and the spherical prior shape as about `cell_cov_prior` spots, against the spots assigned to the cell. At the defaults of 1000, far more than a cell holds, the centre and shape stay close to the segmentation. Run with the defaults first; lowering them towards the spot count of a typical cell lets the spots move and reshape the cells, at the cost of more iterations, and the calls should be compared with the default run to see whether the change is worth it. See [choosing the prior strengths](the-model/gaussian.md#choosing-the-prior-strengths). |

## Compute requirements

The variational loop is multithreaded. The heavy steps are compiled with numba and run
on every core the process is allowed to use, which on a personal machine is every core
of the machine. pciSeq runs on the CPU. A GPU is neither required nor used.

Memory grows with the product of cells, genes and classes: the largest arrays hold one
float32 value per cell, gene and class, and there are several of them.

As a reference point, a 3D run on 37,310 cells, 2.1 million spots, 316 genes and
38 classes over 69 planes of 6432 x 8545 pixels:

| measurement | value |
| --- | --- |
| peak RAM | 31 GB |
| iterations to converge | 92 |
| output on disk | 8.3 GB, of which 6.6 GB is the debug pickle |

Seconds per iteration on three machines:

| machine | CPU | cores | clock | RAM | OS | seconds per iteration |
| --- | --- | --- | --- | --- | --- | --- |
| workstation | Xeon w7-2475X | 20 | 4.8 GHz | 128 GB DDR5 | Ubuntu 24.04 | 2.5 |
| cluster node | 2 x Xeon Gold 6240 | 36 | 2.6 GHz | 1.5 TB DDR4 | RHEL 9.6 | 4.8 |
| laptop chip | Core i7-1165G7 | 4 | 4.7 GHz | 64 GB DDR4 | Ubuntu 26.04 | 7 |

All three ran Python 3.10. The core count alone does not predict the time: the
machine with the most cores is not the fastest. The cluster node was a large memory
node; the run used 31 GB of RAM of its 1.5 TB.
To leave cores free for other work on a shared machine, set `NUMBA_NUM_THREADS` in the
shell before starting Python:

```bash
export NUMBA_NUM_THREADS=8
```

## Cluster jobs

On a cluster, a run is one multithreaded process on one node, and the scheduler
decides how many cores of that node the job gets. The reference run above was on
Myriad, one of UCL's HPC clusters, which uses the Slurm scheduler. A job there is a
Slurm script that asks for a node, plus what it runs on that node: either a config
file through `pciseq run`, as on the [command line page](api/command-line.md), or a
Python script that calls `fit`.

Both forms are shown below for a dataset named aang. Only the Slurm scripts are
specific to the cluster; the config file and the Python script run unchanged on a
local machine once the paths are edited, and the live viewer, off here because a
compute node has no screen, can be turned on.

<figure class="diagram">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 176" role="img" aria-label="run_aang_myriad_cli.sh, a Slurm script, runs pciseq run aang_myriad.yaml, a config file; run_aang_myriad.sh, a Slurm script, runs python run_aang_myriad.py, a Python script" style="max-width: 760px; width: 100%;">
  <defs>
    <marker id="cj-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path class="aq-local-head" d="M 0 0 L 10 5 L 0 10 z" />
    </marker>
  </defs>
  <text class="aq-cap" x="12" y="14">SUBMITTED WITH SBATCH</text>
  <text class="aq-cap" x="432" y="14">RUN ON THE NODE</text>
  <g>
    <rect class="aq-box" x="12" y="24" width="276" height="58" rx="7" />
    <text class="aq-title aq-mono" x="150" y="49" text-anchor="middle" style="font-size:14px">run_aang_myriad_cli.sh</text>
    <text class="aq-sub" x="150" y="70" text-anchor="middle">Slurm script</text>
    <path class="aq-local" d="M 290 53 H 426" marker-end="url(#cj-head)" fill="none" />
    <text class="aq-label" x="359" y="46" text-anchor="middle">runs</text>
    <rect class="aq-box" x="432" y="24" width="316" height="58" rx="7" />
    <text class="aq-title aq-mono" x="590" y="49" text-anchor="middle" style="font-size:14px">pciseq run aang_myriad.yaml</text>
    <text class="aq-sub" x="590" y="70" text-anchor="middle">the command line, with a config file</text>
  </g>
  <g>
    <rect class="aq-box" x="12" y="106" width="276" height="58" rx="7" />
    <text class="aq-title aq-mono" x="150" y="131" text-anchor="middle" style="font-size:14px">run_aang_myriad.sh</text>
    <text class="aq-sub" x="150" y="152" text-anchor="middle">Slurm script</text>
    <path class="aq-local" d="M 290 135 H 426" marker-end="url(#cj-head)" fill="none" />
    <text class="aq-label" x="359" y="128" text-anchor="middle">runs</text>
    <rect class="aq-box" x="432" y="106" width="316" height="58" rx="7" />
    <text class="aq-title aq-mono" x="590" y="131" text-anchor="middle" style="font-size:14px">python run_aang_myriad.py</text>
    <text class="aq-sub" x="590" y="152" text-anchor="middle">a Python script that calls fit</text>
  </g>
</svg>
<figcaption>The two Slurm scripts and the commands they execute on the node.</figcaption>
</figure>

The Python script runs pciSeq from a clone of the repository in the home folder
rather than an installed copy, its first lines make sure of that, and it prints the
commit so the log records which code ran.

::: code-group

```yaml [aang_myriad.yaml]
# aang on UCL Myriad through the command line. the data was rsynced to ~/data.
# no live viewer, a compute node has no screen. run with the slurm script
# run_aang_myriad_cli.sh, which calls: pciseq run aang_myriad.yaml
scrnaseq:
  path: ~/data/scRNAseq_final.csv
  index_col: 0

spots:
  path: ~/data/aang_1.5.0/aang_1.5.0_pciSeq/aang_1.5.0_pciseq_omp_0score.csv
  rename: {z_stack: z_plane, Gene: gene_name}
  # goes straight to pandas DataFrame.query, so it must be an expression that
  # pandas can parse: comparisons joined with and/or, not plain english and
  # not a comma list
  filter: "score > 0.4 and intensity > 0.15"

masks:
  path: ~/data/aang_1.5.0/aang_1.5.0_pciSeq/aang_1.5.0_masks.npy

opts:
  realtime_viewer: false
  Inefficiency: 0.1
  MisreadDensity: 1.0e-6
  voxel_size: [0.28, 0.28, 0.7]
  rTheta: 2
  mrf_beta: 1.5
  exclude_genes: [Zbtb20, Ddit4l]
  output_path: /home/skgtdni/pciseq_runs/aang_cli
```

```python [run_aang_myriad.py]
import os
import sys
import subprocess

# aang on UCL Myriad. the clone of pciSeq_3d sits in the
# home folder there, and the data was rsynced to ~/data. no realtime viewer,
# a compute node has no screen.
REPO = os.path.expanduser("~/pciSeq_3d")
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

import pciSeq

# realpath on both sides, on myriad ~ is /home/<user> but python reports
# /myriadfs/home/<user>
if not os.path.realpath(pciSeq.__file__).startswith(os.path.realpath(REPO)):
    raise SystemExit("ABORT: pciSeq came from %s, not %s" % (os.path.dirname(pciSeq.__file__), REPO))

SPOTS = os.path.expanduser("~/data/aang_1.5.0/aang_1.5.0_pciSeq/aang_1.5.0_pciseq_omp_0score.csv")
MASKS = os.path.expanduser("~/data/aang_1.5.0/aang_1.5.0_pciSeq/aang_1.5.0_masks.npy")
SCRNA = os.path.expanduser("~/data/scRNAseq_final.csv")
OUT = os.path.expanduser("~/pciseq_runs/aang")


if __name__ == "__main__":
    pciSeq.setup_logger()

    # say which code this is, so the log records it.
    commit = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %s"],
                            capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", REPO, "branch", "--show-current"],
                            capture_output=True, text=True).stdout.strip()
    print("running pciSeq from %s, branch %s, commit %s" % (REPO, branch, commit))

    scRNAseq = pd.read_csv(SCRNA)
    scRNAseq = scRNAseq.set_index("Unnamed: 0")

    spots = pd.read_csv(SPOTS)
    spots = spots.rename(columns={"z_stack": "z_plane", "Gene": "gene_name"})
    spots = spots[spots.score > 0.4]
    spots = spots[spots.intensity > 0.15]

    coo = np.load(MASKS)
    coo = [coo_matrix(d) for d in coo]

    opts_3D = {
        "realtime_viewer": False,
        "Inefficiency": 0.1,
        "MisreadDensity": 1e-6,
        "voxel_size": [0.28, 0.28, 0.7],
        "rTheta": 2,
        "mrf_beta": 1.5,
        "exclude_genes": ["Zbtb20", "Ddit4l"],
        "output_path": OUT,
    }

    pciSeq.fit(spots=spots, coo=coo, scRNAseq=scRNAseq, opts=opts_3D)
```

:::

The Slurm scripts, one for each file above. Every `#SBATCH` line is a request to the
scheduler; the lines after them run on the node once it is granted:

::: code-group

```bash [run_aang_myriad_cli.sh]
#!/bin/bash -l
# slurm job for aang on myriad-ng through the command line: pciseq run with
# the config file aang_myriad.yaml. one standard node, all 36 cores (the numba
# kernels use them), 64GB of RAM is twice the 31GB peak seen so far.
# submit with: sbatch run_aang_myriad_cli.sh
#SBATCH --job-name=aang_cli
#SBATCH --partition=myriad
#SBATCH --account=allusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

# size the thread pools of numba and of the blas behind numpy from what
# slurm gave us, rather than leave them to work it out from the node
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/venvs/pciseq/bin/activate
cd ~/jobs
pciseq run aang_myriad.yaml
```

```bash [run_aang_myriad.sh]
#!/bin/bash -l
# slurm job for run_aang_myriad.py on myriad-ng. one standard node, all 36
# cores (the numba kernels use them), 64GB is twice the 31GB peak seen so far.
# submit with: sbatch run_aang_myriad.sh
#SBATCH --job-name=aang
#SBATCH --partition=myriad
#SBATCH --account=allusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

# size the thread pools of numba and of the blas behind numpy from what
# slurm gave us, rather than leave them to work it out from the node
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/venvs/pciseq/bin/activate
cd ~/jobs
python run_aang_myriad.py
```

:::

| flag | description |
| --- | --- |
| `--job-name` | the name shown in the queue, and `%x` in the log file name |
| `--partition` | the group of nodes to run on |
| `--account` | the allocation the job is charged to |
| `--nodes=1`, `--ntasks=1` | one node and one process on it; pciSeq is one process with many threads |
| `--cpus-per-task=36` | the cores for that process, here all the cores of a standard node |
| `--mem` | the RAM reserved for the job |
| `--time` | the wall clock time reserved for the job |
| `--output` | the log file; `%x` is the job name and `%j` the job number |

The two `export` lines set the thread counts to what the job was granted.
`--cpus-per-task=36` asks for 36 cores; Slurm grants them and writes the number into
`SLURM_CPUS_PER_TASK`. Left to itself, numba sizes its thread pool from the cores the
operating system lets the process run on. Whether a cluster restricts that to the
job's cores or leaves the whole node visible depends on how the cluster is set up, so
the script does not rely on it and states the count. `OMP_NUM_THREADS` does the same
for a second thread pool: the linear algebra library behind numpy and scipy has its
own threads and sizes them from that variable. `export` is what makes a shell variable
visible to the Python process started after it.

The run log is written to `pciSeq.log` in a `pciSeq` folder under the system temporary
directory; the path is printed at the start of the run. On a machine shared between
users that folder may belong to someone else, in which case the run stops at once with
a permission error. Setting `TMPDIR` to a writable directory moves the log.
