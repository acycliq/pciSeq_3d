"""What produced a run: version, branch, commit, and the pickled model."""

import os
import pickle
import logging
from typing import Dict, Any

from .diagnostics_db import export_diagnostics

logger = logging.getLogger(__name__)



def collect_metadata() -> Dict:
    """Collect metadata about the environment and analysis run."""
    import platform
    import subprocess
    import sys
    from datetime import datetime

    # Git commit of the pciSeq code
    git_commit = None
    try:
        pciSeq_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=pciSeq_dir,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        pass

    # Key package versions
    pkg_versions = {}
    for pkg in ['numpy', 'scipy', 'pandas', 'pciSeq']:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            pass

    metadata = {
        'date': datetime.now().isoformat(),
        'git_commit': git_commit,
        'hostname': platform.node(),
        'os': f'{platform.system()} {platform.release()}',
        'python_version': sys.version.split()[0],
        'package_versions': pkg_versions,
    }
    return metadata



def serialise(varBayes: Any, debug_dir: str) -> None:
    """Pickle variable Bayes object to debug directory.

    Args:
        varBayes: Object to serialize
        debug_dir: Directory to save pickle file
    """
    varBayes._metadata = collect_metadata()

    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    pickle_dst = os.path.join(debug_dir, 'pciSeq.pickle')
    with open(pickle_dst, 'wb') as outf:
        pickle.dump(varBayes, outf)

    pickle_mb = os.path.getsize(pickle_dst) / (1024 * 1024)
    logger.info('Saved at %s (%.1f MB)', pickle_dst, pickle_mb)

    # Export diagnostics database to diagnostics folder (sibling of arrow folder)
    # This allows the viewer to auto-discover it alongside arrow data
    # the viewer reads everything under viewer_data, the db included
    viewer_dir = os.path.join(os.path.dirname(debug_dir), 'viewer_data')
    export_diagnostics(varBayes, viewer_dir)
