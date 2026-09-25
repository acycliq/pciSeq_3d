import subprocess
import os
from pciSeq._version import __version__


def _resolve_git_info():
    """Return (commit, branch, commit_date). Try live git first, fall back to
    values baked at build time, fall back to 'unknown'.

    commit_date is when that commit was made, so how old the code is. There used to
    be a build_date as well, but from a source checkout it was just the last time
    setup.py ran, which had nothing to do with the commit, so it was dropped."""
    pkg_dir = os.path.dirname(__file__)

    def _git(args):
        try:
            return subprocess.check_output(
                ['git'] + args,
                cwd=pkg_dir,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return ''

    commit = _git(['rev-parse', '--short', 'HEAD'])
    branch = _git(['rev-parse', '--abbrev-ref', 'HEAD'])
    if commit and branch:
        return commit, branch, _git(['log', '-1', '--format=%cI']) or 'unknown'

    # a pip install has no .git, so read what setup.py baked in when it was built.
    # getattr because a _build_info.py from before commit_date will not have it
    try:
        from pciSeq import _build_info as bi
        return (getattr(bi, '__commit__', '') or 'unknown',
                getattr(bi, '__branch__', '') or 'unknown',
                getattr(bi, '__commit_date__', '') or 'unknown')
    except ImportError:
        return 'unknown', 'unknown', 'unknown'


__commit__, __branch__, __commit_date__ = _resolve_git_info()


from pciSeq.app import fit
from pciSeq.app import cell_type
from pciSeq.src.preprocess.main import stage_data
from pciSeq.src.core.logger import attach_to_log, setup_logger
from pciSeq.src.core.utils.spatialdata_export import to_spatialdata, write_spatialdata, add_image, add_boundaries
import logging

logger = logging.getLogger(__name__)


def _check_libvips():
    """Is libvips there?

    setup.py asks for pyvips[binary], which ships the libvips binaries with the
    wheel, so on any platform with a wheel this is True and the tiling functions
    import normally. It is still worth checking: a platform with no wheel would
    otherwise fail with an ImportError from somewhere deep in the import chain
    rather than a message saying what is wrong.
    """
    try:
        import pyvips
        return True
    except (ImportError, OSError):
        # ImportError: pyvips is not installed at all. OSError: pyvips is there but
        # the libvips library it wraps is not.
        return False


# reading tiles back only needs sqlite and pillow, so it is not behind the libvips check
from pciSeq.src.tiling.read_tiles import read_tiles

if _check_libvips():
    from pciSeq.src.tiling.stage_image import tile_maker, stage_image
else:
    _NO_VIPS = ('%s() needs libvips, which normally arrives with the '
                'pyvips[binary] dependency. Reinstalling pciSeq should fix it. '
                'If your platform has no pyvips wheel, see '
                'https://www.libvips.org/install.html')

    # raise, do not just warn. A warning and a None back let a script carry on with
    # no tiles made and nothing to say why.
    def tile_maker(*args, **kwargs):
        raise ImportError(_NO_VIPS % 'tile_maker')

    def stage_image(*args, **kwargs):
        raise ImportError(_NO_VIPS % 'stage_image')
