import subprocess
import os
from pciSeq._version import __version__


def _resolve_git_info():
    """Return (commit, branch). Try live git first, fall back to values
    baked at build time, fall back to 'unknown'."""
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
        return commit, branch

    try:
        from pciSeq._build_info import __commit__ as c, __branch__ as b
        return c or 'unknown', b or 'unknown'
    except ImportError:
        return 'unknown', 'unknown'


def _resolve_build_date():
    """Date setup.py was last run (write time of _build_info.py).
    'unknown' if the package was imported from a source checkout that
    has never been built."""
    try:
        from pciSeq._build_info import __build_date__
        return __build_date__ or 'unknown'
    except ImportError:
        return 'unknown'


__commit__, __branch__ = _resolve_git_info()
__build_date__ = _resolve_build_date()


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
    except OSError:
        return False


if _check_libvips():
    from pciSeq.src.tiling.stage_image import tile_maker, stage_image
else:
    _NO_VIPS = ('>>>> %s() needs libvips, which normally arrives with the '
                'pyvips[binary] dependency. Reinstalling pciSeq should fix it. '
                'If your platform has no pyvips wheel, see '
                'https://www.libvips.org/install.html <<<<')

    def tile_maker(*args, **kwargs):
        logger.warning(_NO_VIPS, 'tile_maker')

    def stage_image(*args, **kwargs):
        logger.warning(_NO_VIPS, 'stage_image')
