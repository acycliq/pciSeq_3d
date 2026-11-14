"""
There is one stamp on a run, and it is built in one place.

The model used to carry two. `metadata` was set when the model was built, and was the
one the diagnostics db and the SpatialData store wrote out. `_metadata` was a second,
older one that serialise() slapped on at pickling time. Nothing read it, it got its
commit a different way (so the two could disagree) and it quietly put the machine's
hostname into every pickle.

Now io/provenance.py builds the stamp, VarBayes keeps it as `metadata`, and that is
the lot.
"""
import json
import pickle

from pciSeq.src.core.io import run_metadata

KEYS = {'version', 'branch', 'commit', 'commit_date', 'tag', 'created_at',
        'python_version', 'os', 'package_versions'}


def test_the_stamp_has_the_keys_the_docs_list():
    assert set(run_metadata()) == KEYS


def test_the_old_keys_are_still_there_under_the_same_names():
    # the db and the store have been writing these out for a while, anything reading
    # them has to keep working. build_date was dropped in September 2026 on purpose,
    # it was misleading from a checkout; old runs still carry it and the viewer's
    # About box falls back to it
    assert {'version', 'branch', 'commit', 'created_at'} <= set(run_metadata())


def test_commit_date_is_the_date_of_the_commit():
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    want = subprocess.check_output(['git', 'log', '-1', '--format=%cI'], cwd=repo, text=True).strip()
    assert run_metadata()['commit_date'] == want


def _git_tag_here():
    # the tag HEAD is exactly on, or '' when it is on none (git exits non zero then)
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(['git', 'describe', '--tags', '--exact-match'], cwd=repo,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        return ''


def test_tag_is_the_tag_of_the_commit_or_empty():
    # on a release commit this is the tag name, like 'v0.1.2'. Anywhere else it is
    # an empty string, not None and not 'unknown', so every value stays a string
    tag = run_metadata()['tag']
    assert isinstance(tag, str)
    assert tag == _git_tag_here()


def test_tag_is_never_the_nearest_tag_with_a_distance():
    # git describe without --exact-match gives things like v0.1.2-7-gabc1234 a few
    # commits after a release. That reads like a release and is not one.
    import re
    assert not re.search(r'-\d+-g[0-9a-f]+$', run_metadata()['tag'])


def test_it_goes_to_json_as_it_is():
    # both the diagnostics db and the SpatialData store json.dumps it
    stamp = run_metadata()
    assert json.loads(json.dumps(stamp)) == stamp


def test_the_libraries_the_numbers_depend_on_are_in():
    versions = run_metadata()['package_versions']
    assert set(versions) == {'numpy', 'scipy', 'pandas', 'numba'}
    assert all(isinstance(v, str) and v for v in versions.values())


def test_nothing_that_names_the_machine_or_the_user():
    import getpass
    import platform
    text = json.dumps(run_metadata()).lower()
    assert 'hostname' not in text
    assert platform.node().lower() not in text
    assert getpass.getuser().lower() not in text


def test_the_model_carries_that_stamp_and_only_that_one(minimal_varbayes):
    vb = minimal_varbayes
    assert set(vb.metadata) == KEYS

    back = pickle.loads(pickle.dumps(vb))
    assert back.metadata == vb.metadata
    assert not hasattr(back, '_metadata')
