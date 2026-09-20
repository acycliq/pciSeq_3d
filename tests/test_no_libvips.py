"""
What happens when libvips is not there.

Two ways it can be missing: pyvips is not installed at all (ImportError), or pyvips
is installed but the libvips library it wraps is not (OSError). Either way pciSeq has
to import fine, read_tiles has to work since it never needed libvips, and the two
tiling functions have to raise and say what is wrong.

They used to log a warning and hand back None, so a script carried on with no tiles
made. And only the OSError case was caught, so with pyvips missing altogether
`import pciSeq` itself fell over.

Each case runs in a fresh python, faking a missing module inside the test process
would leak into every test that runs after it.
"""
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECK = textwrap.dedent("""
    import pciSeq
    assert callable(pciSeq.read_tiles)
    for fn in (pciSeq.stage_image, pciSeq.tile_maker):
        try:
            out = fn('whatever')
        except ImportError as err:
            assert 'libvips' in str(err) and fn.__name__ in str(err), str(err)
        else:
            raise SystemExit(f'{fn.__name__} did not raise, it returned {out!r}')
    print('ok')
""")


def _run(tmp_path, fake_pyvips):
    """Put a fake pyvips in front of the real one and run CHECK in a new python."""
    (tmp_path / "pyvips.py").write_text(fake_pyvips)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), REPO]))
    return subprocess.run([sys.executable, "-c", CHECK], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=300)


@pytest.mark.parametrize("fake_pyvips", [
    "raise ImportError('No module named pyvips')",
    "raise OSError('cannot load library libvips.so.42')",
], ids=["pyvips_not_installed", "libvips_library_missing"])
def test_pciSeq_imports_and_the_tiling_functions_raise(tmp_path, fake_pyvips):
    done = _run(tmp_path, fake_pyvips)
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("ok")
