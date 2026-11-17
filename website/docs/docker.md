# Docker

The repository holds a Dockerfile, `docker/Dockerfile`, for running pciSeq without installing
Python or any package on the machine. The image contains Python 3.10, pciSeq with the `mcp`
extra, and the dependencies at the exact versions listed in `docker/constraints.txt`. pciSeq
Viewer is a desktop application and is not part of the image.

## Requirements

- [Docker](https://docs.docker.com/get-docker/). Python, pip and git are not required; they
  are inside the image.
- The two files of the `docker/` folder, `Dockerfile` and `constraints.txt`, obtained by
  cloning the repository or by downloading them.
- An internet connection during the build, which downloads Python, the dependencies and
  pciSeq. A run needs no connection.
- About 2 GB of disk space for the image. The build takes a few minutes.

The image has been built and run on Linux, x86_64. Windows and macOS have not been tested.
On those systems Docker Desktop runs containers inside a virtual machine with a fixed memory
allowance, which has to be large enough for the dataset.

On Linux, Docker commands need either `sudo` or membership of the `docker` group.

## Image build

The image is built from the top of the repository. `PCISEQ_REF` is the tag, branch or commit
of pciSeq to install; the code is fetched from GitHub, not copied from the checkout.

```bash
docker build --build-arg PCISEQ_REF=v0.1.2 -t pciseq:0.1.2 docker/
```

The last argument, `docker/`, is the folder that holds the two files, here the `docker` folder
of the repository. If the two files were downloaded to another folder, that folder is given
in its place. `-t` sets the name of the image, used in the commands below.

A quick check that the image works is to ask it for the version of pciSeq it holds:

```bash
docker run --rm pciseq:0.1.2 --version
```

## Run from a config file

The image starts the `pciseq` command (see [Command line](./api/command-line.md)).

A container cannot see the files of the machine. One folder is made visible to it with `-v`,
and inside the container that folder has another name, `/data`. The example below assumes
the config file and the inputs are in one folder:

```
/home/user/my_run/
    config.yaml
    spots.csv
    masks.npy
    scRNAseq.csv
```

pciSeq runs inside the container, so every path in the config file is the path inside the
container and starts with `/data`, `output_path` included:

```yaml
scrnaseq:
  path: /data/scRNAseq.csv
  index_col: 0

spots:
  path: /data/spots.csv

masks:
  path: /data/masks.npy

opts:
  realtime_viewer: false
  voxel_size: [0.28, 0.28, 0.7]
  output_path: /data/out
```

There is no `/data` folder on the machine. The paths above exist only inside the container,
where the option `-v /home/user/my_run:/data` of the run shows the folder of the machine
under that name. Nothing is copied; each file has two names:

| On the machine | Inside the container |
| --- | --- |
| `/home/user/my_run/masks.npy` | `/data/masks.npy` |
| `/home/user/my_run/config.yaml` | `/data/config.yaml` |
| `/home/user/my_run/out/` | `/data/out/` |

Inputs kept in several folders are given with one `-v` each, every folder under its own name
inside the container.

The run:

```bash
docker run --rm -u $(id -u):$(id -g) -v /home/user/my_run:/data pciseq:0.1.2 run /data/config.yaml
```

| Part | Meaning |
| --- | --- |
| `--rm` | The container is removed when the run ends. The image and the output files remain. |
| `-u $(id -u):$(id -g)` | The run is made as the current user, so the output files belong to that user and not to root. |
| `-v /home/user/my_run:/data` | The folder on the machine, left of the colon, is visible inside the container as `/data`. |
| `pciseq:0.1.2` | The image, by the name given at the build. |
| `run /data/config.yaml` | The arguments of the `pciseq` command, which the image starts. The same as `pciseq run /data/config.yaml` on a machine with pciSeq installed. |

The output is written to `/data/out` inside the container, which is
`/home/user/my_run/out` on the machine, where pciSeq Viewer opens it. A file written outside
`/data` is lost when the container is removed.

The live viewer has not been tested from inside the image. `realtime_viewer` is left at
`false`, its default, and the run is inspected afterwards with pciSeq Viewer.

## Run from a Python script

A Python script that calls `pciSeq.fit` (see [Running pciSeq](./running-pciseq.md)) is run
the same way, with `python` in place of the `pciseq` command. The script is placed in the
mounted folder, next to the inputs, and its paths start with `/data` too. The script below
does the same run as the config file above:

::: code-group

```python [my_script.py]
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

import pciSeq

# /data is the folder given with -v, see the run command below
spots = pd.read_csv("/data/spots.csv")
masks = np.load("/data/masks.npy")
coo = [coo_matrix(plane) for plane in masks]
scRNAseq = pd.read_csv("/data/scRNAseq.csv", index_col=0)

opts = {
    "realtime_viewer": False,
    "voxel_size": [0.28, 0.28, 0.7],
    "output_path": "/data/out",
}

pciSeq.fit(spots=spots, coo=coo, scRNAseq=scRNAseq, opts=opts)
```

:::

The run, with `--entrypoint python` so that the image starts `python` and not `pciseq`:

```bash
docker run --rm -u $(id -u):$(id -g) -v /home/user/my_run:/data --entrypoint python pciseq:0.1.2 /data/my_script.py
```

## HPC clusters

Most HPC clusters do not permit Docker. They provide Apptainer, which runs the same image from a
single file with the extension `.sif`. The `.sif` file is the Apptainer form of the image:
one read-only file holding everything the Docker image holds. It is made once from the Docker
image and then used for every run.

The steps take place on two machines: a machine with Docker, where the image was built, and
the HPC cluster.

1. On the machine with Docker, the image is written to a file. The file is about 2 GB.

   ```bash
   docker save pciseq:0.1.2 -o pciseq_0.1.2.tar
   ```

2. The file is copied to the cluster, here to the home folder of the user. `user` and
   `cluster.address` stand for the login name and the address of the cluster.

   ```bash
   scp pciseq_0.1.2.tar user@cluster.address:~/
   ```

3. On the cluster, the `.sif` file is made from it. After this step the `.tar` file is no
   longer needed, on either machine.

   ```bash
   apptainer build pciseq_0.1.2.sif docker-archive://pciseq_0.1.2.tar
   ```

4. On the cluster, the run. It is the same as the Docker run above, with `--bind` in place of
   `-v`.

   ```bash
   apptainer run --bind /path/to/my_run:/data pciseq_0.1.2.sif run /data/config.yaml
   ```

`/path/to/my_run` is a folder on the cluster. The config file and the inputs have to be on
the cluster too, in that folder, and the paths in the config file start with `/data` as before.
The output is written to that folder, `/path/to/my_run/out` in the example config.

Apptainer always runs as the user who starts it, so there is no equivalent of the `-u` option
and the output files belong to that user.

The command of step 4 can be typed on a login node for a small test. A real dataset is run
through the scheduler of the cluster.

### Scheduler

The image replaces the installation of pciSeq on the cluster and nothing else. The request to
the scheduler is not affected: the node, the number of cores, the memory and the time limit
are asked for as in [Cluster jobs](./running-pciseq.md#cluster-jobs). The container runs
inside the job and has the cores and the memory that the scheduler granted.

In a Slurm script, the lines that activate a Python environment and start pciSeq are
replaced by one `apptainer` line. The `#SBATCH` lines and the two `export` lines stay as they
are; Apptainer passes the environment variables of the job on to pciSeq.

The script below is for Myriad, as on the Cluster jobs page. It assumes that the `.sif` file
is in the home folder and that the config file and the inputs are in `~/my_run`.

::: code-group

```bash [run_apptainer.sh]
#!/bin/bash -l
# slurm job on myriad-ng: pciseq run with a config file, inside the apptainer
# image. one standard node, all 36 cores, 64GB.
# submit with: sbatch run_apptainer.sh
#SBATCH --job-name=my_run
#SBATCH --partition=myriad
#SBATCH --account=allusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log

# size the thread pools of numba and of the blas behind numpy from what
# slurm gave us. apptainer hands these on to the container
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ~/my_run on the cluster is /data inside the container, so the paths in
# config.yaml start with /data
apptainer run --bind ~/my_run:/data ~/pciseq_0.1.2.sif run /data/config.yaml
```

:::

A job is submitted from the login node of the cluster, from the folder that holds the script:

```bash
sbatch run_apptainer.sh
```

For a Python script in place of the config file, the last line of the job script becomes
`apptainer exec --bind ~/my_run:/data ~/pciseq_0.1.2.sif python /data/my_script.py`. `exec`
runs the program that is named, here `python`, where `run` starts `pciseq`.

`squeue -u <user>` lists the job while it waits or runs. The messages of the run are written
to a log file in the same folder, `my_run_<job number>.log`, and the output of pciSeq to
`~/my_run/out` when `output_path` in the config file is `/data/out`.

## Dependency versions

`docker/constraints.txt` applies to the image only. The dependencies in `setup.py` carry no
exact versions, so that pciSeq can be installed with pip next to other packages.

The file was recorded with v0.1.2. An image of a later commit is built with the same file; a
dependency added after v0.1.2 is not in the list and is installed at its newest version.

The same file can be passed to pip to reproduce the environment of the image outside Docker:

```bash
pip install -c docker/constraints.txt git+https://github.com/acycliq/pciSeq_3d.git@v0.1.2
```
