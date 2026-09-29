"""
The realtime viewer server.

It is optional and off by default, so nothing else in the suite touches it. These
tests cover the parts that break quietly: that the server actually serves, that
stop() frees the port (it used to only flip a flag), and that send_update never
raises into the model loop no matter what it is handed.
"""
import logging
import socket
import time
import urllib.request

import numpy as np
import pytest

from pciSeq.src.realtime_viewer import RealtimeViewerServer


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _bound(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) == 0


@pytest.fixture
def server():
    srv = RealtimeViewerServer(port=_free_port(), auto_open_browser=False)
    yield srv
    srv.stop()


def test_it_serves_while_running(server):
    server.start()
    time.sleep(0.8)
    base = 'http://127.0.0.1:%d' % server.port
    assert urllib.request.urlopen(base + '/health', timeout=3).status == 200
    assert urllib.request.urlopen(base + '/', timeout=3).status == 200


def test_socketio_is_still_wired_up(server):
    """We serve through make_server rather than socketio.run, so check the
    socketio middleware is still in the chain."""
    server.start()
    time.sleep(0.8)
    url = 'http://127.0.0.1:%d/socket.io/?EIO=4&transport=polling' % server.port
    assert urllib.request.urlopen(url, timeout=3).status == 200


def test_stop_frees_the_port(server):
    """It used to only set a flag, leaving the port bound for the whole process."""
    server.start()
    time.sleep(0.8)
    assert _bound(server.port)
    server.stop()
    time.sleep(0.5)
    assert not _bound(server.port), 'stop() left the port bound'
    assert server._is_running is False


def test_the_port_can_be_reused(server):
    """Two runs in one process have to be able to use the same port."""
    server.start(); time.sleep(0.8); server.stop(); time.sleep(0.5)
    again = RealtimeViewerServer(port=server.port, auto_open_browser=False)
    try:
        again.start()
        time.sleep(0.8)
        assert _bound(again.port)
    finally:
        again.stop()


def test_starting_twice_is_a_no_op(server, caplog):
    server.start()
    time.sleep(0.5)
    with caplog.at_level(logging.WARNING):
        server.start()
    assert 'already running' in caplog.text.lower()


def test_stopping_when_not_started_is_safe(server):
    server.stop()          # must not raise


def test_it_says_so_when_no_model_is_wired(server, caplog):
    """app.py attaches the model after construction. Until it does, send_update
    should say the reference is missing rather than fall over inside."""
    server.start()
    time.sleep(0.5)
    with caplog.at_level(logging.WARNING):
        server.send_update(np.zeros((3, 2)), 1, 0.5)
    assert 'VarBayes reference not set' in caplog.text


def test_send_update_never_raises_into_the_loop(server, caplog):
    """It is called from main_loop, so nothing here may kill the run."""
    server.start()
    time.sleep(0.5)
    server._varbayes_ref = object()          # wired, but useless
    with caplog.at_level(logging.ERROR):
        server.send_update(None, 0, 0.5)     # must not propagate
    assert 'Failed to send update' in caplog.text


# --------------------------------------------------------------------------- #
# What send_update puts on the wire. The browser code reads these exact event
# names and keys, so this is a contract: change it here and viewer.html and the
# js in static/js have to change with it.
# --------------------------------------------------------------------------- #

EXPECTED_EVENTS = [
    ('geometry_init_begin', ['cell_call_tolerance', 'chunk_size', 'class_names',
                             'img_dim', 'is_3d', 'mcr', 'num_cells', 'version',
                             'voxel_size']),
    ('geometry_init_chunk', ['cell_ids', 'centroids_x', 'centroids_y',
                             'centroids_z', 'end', 'radii', 'start']),
    ('geometry_init_end',   []),
    ('classes_update_begin', ['chunk_size', 'delta', 'iteration', 'num_cells']),
    ('classes_update_chunk', ['cell_classes', 'end', 'prob', 'start']),
    ('classes_update_end',   ['iteration']),
]


def _wired(vb, port):
    """A server with the model attached and emit captured, without a real socket."""
    srv = RealtimeViewerServer(port=port, auto_open_browser=False)
    srv._varbayes_ref = vb
    srv._is_running = True
    sent = []
    srv.socketio.emit = lambda ev, data=None, **kw: sent.append((ev, data))
    return srv, sent


def _run_one_iteration(vb):
    vb.initialise_state()
    for step in (vb.geneCount_upd, vb.rho_upd, vb.eta_upd, vb.theta_upd,
                 vb.gamma_upd, vb.cell_to_cellType, vb.spots_to_cell_numba):
        step()


def test_send_update_emits_the_expected_events(minimal_varbayes):
    _run_one_iteration(minimal_varbayes)
    srv, sent = _wired(minimal_varbayes, _free_port())
    srv.send_update(minimal_varbayes.cells.classProb, 3, 0.25)

    assert [e for e, _ in sent] == [e for e, _ in EXPECTED_EVENTS]
    for (name, keys), (got_name, payload) in zip(EXPECTED_EVENTS, sent):
        assert sorted(payload) == keys, 'payload of %s changed' % got_name


def test_geometry_is_only_sent_once(minimal_varbayes):
    """The geometry does not change between iterations, so it goes out once."""
    _run_one_iteration(minimal_varbayes)
    srv, sent = _wired(minimal_varbayes, _free_port())
    srv.send_update(minimal_varbayes.cells.classProb, 1, 0.5)
    srv.send_update(minimal_varbayes.cells.classProb, 2, 0.4)
    first = [e for e, _ in sent if e.startswith('geometry_init')]
    assert len(first) == 3, 'geometry should be sent once, got %d events' % len(first)


def test_an_empty_label_map_is_not_a_label_map(minimal_varbayes):
    """{} means no renumbering happened. Treating it as a map indexes into
    nothing and raises."""
    _run_one_iteration(minimal_varbayes)
    minimal_varbayes.config['label_map'] = {}
    srv, sent = _wired(minimal_varbayes, _free_port())
    srv.send_update(minimal_varbayes.cells.classProb, 1, 0.5)
    assert sent, 'send_update produced nothing with an empty label_map'


def test_the_last_update_is_cached_for_late_joiners(minimal_varbayes):
    """handle_connect replays this to a client that arrives mid run. It used to
    be built after the emit and referenced a variable that had moved, so the
    NameError was swallowed by the broad except and the cache stayed empty."""
    _run_one_iteration(minimal_varbayes)
    srv, _ = _wired(minimal_varbayes, _free_port())
    srv.send_update(minimal_varbayes.cells.classProb, 7, 0.125)

    cached = getattr(srv, '_last_update', None)
    assert cached is not None, 'nothing was cached'
    assert cached['iteration'] == 7
    assert cached['delta'] == pytest.approx(0.125)
    assert sorted(cached) == ['cell_classes', 'chunk_size', 'delta',
                              'iteration', 'num_cells', 'prob']


# --------------------------------------------------------------------------- #
# The viewer has to work with no internet. The browser libraries used to come
# from cdn.socket.io, unpkg and d3js.org, and deck.gl was pinned to @latest so an
# upstream release could break it without anything changing here.
# --------------------------------------------------------------------------- #

import pathlib

STATIC = pathlib.Path(__file__).parent.parent / 'pciSeq' / 'src' / 'realtime_viewer' / 'static'


def test_no_page_reaches_out_to_the_internet():
    offenders = []
    for f in list(STATIC.glob('*.html')) + list(STATIC.glob('js/*.js')):
        for i, line in enumerate(f.read_text(errors='ignore').splitlines(), 1):
            if 'src="http' in line or 'href="http' in line:
                offenders.append('%s:%d' % (f.name, i))
    assert not offenders, 'external resources referenced at %s' % ', '.join(offenders)


@pytest.mark.parametrize('lib', ['socket.io.min.js', 'd3.v7.min.js', 'deck.gl.min.js'])
def test_the_vendored_libraries_are_there(lib):
    f = STATIC / 'vendor' / lib
    assert f.exists(), '%s is missing, the viewer will not load' % lib
    assert f.stat().st_size > 10_000, '%s looks truncated' % lib


def test_the_server_serves_the_vendored_files(server):
    server.start()
    time.sleep(0.8)
    base = 'http://127.0.0.1:%d' % server.port
    for path in ('/viewer.css', '/vendor/socket.io.min.js',
                 '/vendor/d3.v7.min.js', '/vendor/deck.gl.min.js'):
        assert urllib.request.urlopen(base + path, timeout=5).status == 200, path


def test_the_real_tolerance_reaches_the_viewer(minimal_varbayes):
    """The convergence chart draws its threshold line at this value. It used to
    be left at the browser's placeholder of 0.2 because the server never sent
    it, so the line sat in the wrong place on every run."""
    _run_one_iteration(minimal_varbayes)
    minimal_varbayes.config['CellCallTolerance'] = 0.02
    srv, sent = _wired(minimal_varbayes, _free_port())
    srv.send_update(minimal_varbayes.cells.classProb, 1, 0.5)

    begin = next(d for e, d in sent if e == 'geometry_init_begin')
    assert begin['cell_call_tolerance'] == pytest.approx(0.02)
    assert srv._geometry_cache['cell_call_tolerance'] == pytest.approx(0.02)


def test_class_indices_above_255_reach_the_viewer(minimal_varbayes):
    """A big taxonomy has more than 255 classes. The assigned class used to be cast to
    uint8 before it was sent, so the index wrapped round and class 256 was drawn as
    class 0, class 300 as class 44, and so on. It is uint16 now, and the browser keeps
    it in a Uint16Array."""
    vb = minimal_varbayes
    _run_one_iteration(vb)
    nC = vb.cells.classProb.shape[0]

    # 300 classes, and every cell wins a class past the old limit: 250, 251, 252, ...
    wanted = 250 + np.arange(nC)
    big = np.zeros((nC, 300), dtype=np.float32)
    big[np.arange(nC), wanted] = 1.0

    srv, sent = _wired(vb, _free_port())
    srv.send_update(big, 1, 0.5)

    got = [c for _, d in sent if isinstance(d, dict) and 'cell_classes' in d
           for c in d['cell_classes']]
    # row 0 is the background pseudo cell, the server leaves it out
    assert got == wanted[1:].tolist()
    assert max(got) > 255


# --------------------------------------------------------------------------- #
# The chat panel. The model itself is stubbed (tests/test_mcp_chat.py covers the
# tool loop); what matters here is the wiring: that a question reaches the loop,
# that the answer comes back as chat_event, and that the key never goes out on
# the wire or into the settings route's answer.
# --------------------------------------------------------------------------- #

def test_a_question_runs_a_turn_and_the_steps_come_back(minimal_varbayes, monkeypatch):
    _run_one_iteration(minimal_varbayes)
    srv, sent = _wired(minimal_varbayes, _free_port())

    seen = {}

    def fake_turn(live, messages, on_event=None, **kw):
        seen['question'] = messages[-1]['content']
        seen['live'] = live
        on_event({'type': 'text', 'text': 'iteration 3, still moving'})
        on_event({'type': 'done'})
        return {'text': 'iteration 3, still moving',
                'messages': messages + [{'role': 'assistant', 'content': 'ok'}]}

    from pciSeq.src.mcp import chat
    monkeypatch.setattr(chat, 'run_turn', fake_turn)
    srv._on_chat_message({'text': 'how far along is it?'})

    assert seen['question'] == 'how far along is it?'
    events = [d for e, d in sent if e == 'chat_event']
    assert [d['type'] for d in events] == ['text', 'done']
    # the conversation is kept, so the next question has the context
    assert len(srv._chat_history) == 2


def test_the_chat_gets_a_snapshot_each_iteration(minimal_varbayes):
    """changed_cells compares with the last snapshot, so something has to take
    one every iteration. Only once the chat has been used: no chat, no cost."""
    _run_one_iteration(minimal_varbayes)
    srv, _ = _wired(minimal_varbayes, _free_port())
    srv.send_update(minimal_varbayes.cells.classProb, 1, 0.5)
    assert srv._live is None

    from pciSeq.src.mcp.live import Live
    srv._live = Live(minimal_varbayes)
    minimal_varbayes.iter_num = 2            # main_loop sets this in a real run
    srv.send_update(minimal_varbayes.cells.classProb, 2, 0.4)
    assert srv._live.changed_cells()['compared_with_iteration'] == 2


def test_a_failing_turn_is_reported_and_not_remembered(minimal_varbayes, monkeypatch):
    """A bad key must not leave the question stuck in the history, or every later
    question replays it."""
    _run_one_iteration(minimal_varbayes)
    srv, sent = _wired(minimal_varbayes, _free_port())

    def boom(*a, **kw):
        raise RuntimeError('the model answered 401: invalid x-api-key')

    from pciSeq.src.mcp import chat
    monkeypatch.setattr(chat, 'run_turn', boom)
    srv._on_chat_message({'text': 'hello'})

    errs = [d for e, d in sent if e == 'chat_event' and d['type'] == 'error']
    assert '401' in errs[0]['error']
    assert srv._chat_history == []


def test_no_model_wired_up_says_so(minimal_varbayes):
    srv, sent = _wired(minimal_varbayes, _free_port())
    srv._varbayes_ref = None
    srv._on_chat_message({'text': 'hello'})
    assert [d['type'] for e, d in sent if e == 'chat_event'] == ['error']


def test_the_settings_route_never_hands_out_the_key(minimal_varbayes, tmp_path, monkeypatch):
    from pciSeq.src.mcp import chat_settings
    monkeypatch.setattr(chat_settings, 'config_path', lambda: tmp_path / 'chat.json')
    srv, _ = _wired(minimal_varbayes, _free_port())
    client = srv.app.test_client()

    saved = client.post('/chat/settings', json={'provider': 'anthropic',
                                                'api_key': 'sk-secret'})
    assert saved.status_code == 200
    assert 'sk-secret' not in saved.get_data(as_text=True)
    got = client.get('/chat/settings')
    assert got.get_json()['has_key'] is True
    assert 'sk-secret' not in got.get_data(as_text=True)
    # a service we do not know needs its address, and says so rather than 500ing
    bad = client.post('/chat/settings', json={'provider': 'other'})
    assert bad.status_code == 400 and 'address' in bad.get_json()['error']
