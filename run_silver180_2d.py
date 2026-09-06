"""silver180 squashed down to 2D, to eyeball the 2D path in the viewer.

The spots get flattened: the z_plane column is dropped, so every spot lands on
one plane and the validator fills z_plane with 0. The masks are the single mid
plane of the stack, passed as one coo_matrix rather than a list, which is what
puts the run in 2D mode.

Tolerance is loose on purpose, this is a look-at-it run, not a real one.
"""
import os
import sys

REPO = "/home/dimitris/dev/python/pciseq_apr_502608c"
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from pciSeq.app import fit
from pciSeq.src.core.logger import setup_logger
import pciSeq

if not os.path.dirname(pciSeq.__file__).startswith(REPO):
    raise SystemExit("ABORT: pciSeq came from %s" % os.path.dirname(pciSeq.__file__))

SPOTS = "/home/dimitris/data/Christina/pciSeq_HC_silver/input_and_nb_for_pciSeq/pciseq_omp_noThresh_coppafisher180_ompFix.csv"
MASKS = "/home/dimitris/data/Christina/pciSeq_HC_silver/input_and_nb_for_pciSeq/input/DAPI_3DmaxV1_optimized.npy"
OUT = "/home/dimitris/pciseq_runs/twod_check/silver180_flat"

if __name__ == "__main__":
    setup_logger()

    scRNAseq = pd.read_csv(os.path.join('~', 'data', 'Izzie',
                                        'Aang_coppa_v1_0_0_output', 'scRNAseq_final.csv'))
    scRNAseq = scRNAseq.set_index("Unnamed: 0")

    spots = pd.read_csv(SPOTS)
    spots = spots.rename(columns={"z_stack": "z_plane", "Gene": "gene_name"})
    spots = spots[spots.score > 0.4]
    spots = spots[spots.intensity > 0.15]
    # flatten: drop z entirely, the validator puts z_plane=0 back for 2D
    spots = spots.drop(columns=["z_plane"])
    print("spots: %d, columns %s" % (len(spots), list(spots.columns)))

    masks = np.load(MASKS)
    mid = masks.shape[0] // 2
    coo = coo_matrix(masks[mid])          # ONE matrix, not a list, so is3D=False
    print("masks %s, mid plane %d, cells on it %d"
          % (masks.shape, mid, len(np.unique(masks[mid])) - 1))

    opts_2D = {
        "realtime_viewer": False,
        "save_data": True,
        "Inefficiency": 0.1,
        "MisreadDensity": {"default": 1e-6},
        "nNeighbors": 9,
        "InsideCellBonus": 0,
        "CellCallTolerance": 0.5,
        "voxel_size": [1, 1, 1],          # z is 0 anyway, keep Sz=1 so the
                                          # roi volume and the inverse stay sane
        "remove_flat_cells": True,        # skipped in 2D, every cell is flat
        "SpotReg": 0.1,
        "rTheta": 2,
        "cell_type_weights": None,
        "cell_type_prior": "uniform",
        "mrf_beta": 1.5,
        "rRho": 1,
        "max_iter": 10,
        "exclude_genes": [],
        "output_path": OUT,
    }

    fit(spots=spots, coo=coo, scRNAseq=scRNAseq, opts=opts_2D)
