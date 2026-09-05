"""The one entry point that writes a finished run to disk, in every format."""

import os
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict

from .paths import get_out_dir
from .tsv_export import write_tsv
from .arrow_export import write_arrow
from .provenance import serialise
from ..utils.spatialdata_export import write_spatialdata

logger = logging.getLogger(__name__)



def write_data(cellData: pd.DataFrame, geneData: pd.DataFrame,
               cellBoundaries: pd.DataFrame, cellBoundaries_list: pd.DataFrame,
               coo, varBayes: Any, cfg: Dict) -> None:

    dst = get_out_dir(cfg['output_path'])
    out_dir = os.path.join(dst, 'data')

    # flag the spots whose best guess is the background column, ie the model
    # reckons they are misreads rather than belonging to any cell. Goes on
    # geneData before it is written so it lands in both the tsv and the arrow.
    try:
        prob = varBayes.spots.parent_cell_prob
        if prob is not None and len(prob):
            is_misread = np.argmax(prob, axis=1) == (prob.shape[1] - 1)
            geneData = geneData.copy()
            geneData['is_hard_misread'] = is_misread.astype(np.uint8)
    except Exception as e:
        logger.warning('Could not attach is_hard_misread to geneData: %s', e)

    write_tsv(cellData, geneData, cellBoundaries, out_dir)
    write_arrow(geneData, cellData, cellBoundaries_list, out_dir)
    # the same results again as a SpatialData zarr store, so the scverse tools
    # (napari-spatialdata, squidpy, scanpy) can open the run
    write_spatialdata(cellData, geneData, coo, varBayes, cfg, out_dir)

    # Save debug info
    serialise(varBayes, os.path.join(out_dir, 'debug'))
