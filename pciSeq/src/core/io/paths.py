"""Where a run's output goes, and where the package lives."""

import os
import tempfile
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional
from email.parser import BytesHeaderParser

logger = logging.getLogger(__name__)



def get_out_dir(path: Optional[str] = None, sub_folder: str = '') -> str:
    """Get or create output directory path.

    Args:
        path: Base path, or None for default temp directory
        sub_folder: Optional subdirectory name

    Returns:
        str: Path to output directory

    Notes:
        - Uses system temp directory if path is None or 'default'
        - Creates directories if they don't exist
    """
    if path is None or path == 'default':
        out_dir = Path(tempfile.gettempdir()) / 'pciSeq'
    else:
        out_dir = Path(path) / sub_folder / 'pciSeq'

    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)



def get_pciSeq_install_dir():
    p = subprocess.run(['pip', 'show', 'pciSeq'], stdout=subprocess.PIPE)
    h = BytesHeaderParser().parsebytes(p.stdout)
    assert h['Location'] is not None, 'Could not locate pciSeq installation folder, maybe the package is not installed.'
    return os.path.join(h['Location'], 'pciSeq')



def log_file(cfg: Dict) -> None:
    """Setup the logger file handler if it doesn't exist.

    Args:
        cfg: Configuration dictionary containing output path

    Notes:
        - Only adds FileHandler if one doesn't already exist
        - Creates log file in output directory
        - Uses standard logging format
    """
    root_logger = logging.getLogger()

    # Only add FileHandler if none exists
    if root_logger.handlers and not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        logfile = os.path.join(get_out_dir(cfg['output_path']), 'pciSeq.log')
        fh = logging.FileHandler(logfile, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        root_logger.addHandler(fh)
        logger.info('Writing to %s' % logfile)
