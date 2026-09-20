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

KEYS = {'version', 'branch', 'commit', 'build_date', 'created_at',
        'python_version', 'os', 'package_versions'}


def test_the_stamp_has_the_keys_the_docs_list():
    assert set(run_metadata()) == KEYS


def test_the_old_keys_are_still_there_under_the_same_names():
    # the db and the store have been writing these five out for a while, anything
    # reading them has to keep working
    assert {'version', 'branch', 'commit', 'build_date', 'created_at'} <= set(run_metadata())


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
