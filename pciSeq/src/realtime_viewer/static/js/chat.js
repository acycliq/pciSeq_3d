/**
 * Chat Module
 * Ask the model about the run while it is still running.
 *
 * A dock along the bottom, the same shape as the chat in the desktop viewer:
 * a Connection tab for the provider and key, and a Chat tab for the talking.
 *
 * The page does no thinking of its own. It sends the question over the socket,
 * the server runs the tool loop (pciSeq/src/mcp/chat.py) and sends back each
 * step as a chat_event. The API key lives on the server and never comes here,
 * the Connection tab only ever learns whether one is set.
 */

(function() {
    'use strict';

    window.pciSeq = window.pciSeq || {};

    let providers = [];
    let busy = false;          // a question is in flight, do not send another

    function el(id) {
        return document.getElementById(id);
    }

    function socket() {
        return window.pciSeq.socket;
    }

    // ------------------------------------------------------------ messages

    function addLine(who, text) {
        const div = document.createElement('div');
        div.className = 'chat-' + who;
        div.textContent = text;
        const box = el('chat-messages');
        box.appendChild(div);
        box.scrollTop = box.scrollHeight;
        return div;
    }

    /** The tool call, written the way it would be called, like the desktop one. */
    function addToolLine(name, input) {
        const args = Object.entries(input || {})
            .map(([k, v]) => k + '=' + JSON.stringify(v))
            .join(', ');
        return addLine('tool', name + '(' + args + ')');
    }

    function send() {
        const input = el('chat-input');
        const text = input.value.trim();
        if (!text || busy) return;
        addLine('user', text);
        input.value = '';
        busy = true;
        el('chat-send').disabled = true;
        showThinking();
        socket().emit('chat_message', { text: text });
    }

    /** The thinking line always sits at the bottom, so it keeps showing while
     *  the model works out its answer after the tool calls, not only before
     *  them. Moving the same element is what keeps it last. */
    function showThinking() {
        let t = el('chat-thinking');
        if (!t) {
            t = addLine('thinking', 'thinking...');
            t.id = 'chat-thinking';
        }
        const box = el('chat-messages');
        box.appendChild(t);
        box.scrollTop = box.scrollHeight;
    }

    function clearThinking() {
        const t = el('chat-thinking');
        if (t) t.remove();
    }

    function finish() {
        busy = false;
        el('chat-send').disabled = false;
        clearThinking();
    }

    /** One step of the turn, as the server runs it. */
    function onEvent(ev) {
        if (ev.type === 'tool_call') {
            addToolLine(ev.name, ev.input);
            showThinking();                 // back to the bottom, under the call
        } else if (ev.type === 'tool_result' && ev.is_error) {
            addLine('tool', '  ' + ev.name + ' could not answer');
            showThinking();
        } else if (ev.type === 'text') {
            clearThinking();
            // the model can speak more than once in a turn, before and after a
            // tool call, so each text block gets its own line
            addLine('assistant', ev.text);
            showThinking();                 // it may not be finished yet
        } else if (ev.type === 'error') {
            clearThinking();
            addLine('error', ev.error);
            finish();
        } else if (ev.type === 'done') {
            finish();
        }
    }

    // ---------------------------------------------------------- connection

    function paintSettings(s) {
        if (s.error) {
            el('chat-key-state').textContent = s.error;
            return;
        }
        providers = s.providers || [];
        const sel = el('chat-provider');
        sel.innerHTML = '';
        providers.forEach(p => {
            const o = document.createElement('option');
            o.value = p.id;
            o.textContent = p.label;
            sel.appendChild(o);
        });
        sel.value = s.provider;
        el('chat-model').value = s.model || '';
        el('chat-base-url').value = s.base_url || '';
        el('chat-key').value = '';
        onProviderChange();

        let state;
        if (s.has_key && s.key_from_env) {
            state = 'a key is set, taken from ' + s.env_name;
        } else if (s.has_key) {
            state = 'a key is saved in ' + s.config_file;
        } else {
            state = 'no key yet' + (s.env_name ? ', or set ' + s.env_name : '');
        }
        el('chat-key-state').textContent = state;
    }

    /** The address box only makes sense for a service we do not know. */
    function onProviderChange() {
        const other = el('chat-provider').value === 'other';
        document.querySelectorAll('.chat-other-only').forEach(n => {
            n.style.display = other ? '' : 'none';
        });
    }

    /** Picking a provider moves the model box to that provider's own model.
     *  Without this the box keeps the model of the provider you came from and
     *  saves it, so you end up asking Z.ai for a Claude model. */
    function onProviderPicked() {
        const p = providers.find(x => x.id === el('chat-provider').value);
        if (p) el('chat-model').value = p.model || '';
        onProviderChange();
    }

    function loadSettings() {
        fetch('/chat/settings')
            .then(r => r.json())
            .then(paintSettings)
            .catch(e => { el('chat-key-state').textContent = String(e); });
    }

    function saveSettings() {
        fetch('/chat/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: el('chat-provider').value,
                model: el('chat-model').value,
                base_url: el('chat-base-url').value,
                api_key: el('chat-key').value
            })
        })
            .then(r => r.json())
            .then(s => {
                paintSettings(s);
                if (!s.error) showTab('chat');
            })
            .catch(e => { el('chat-key-state').textContent = String(e); });
    }

    // ------------------------------------------------------------- the dock

    function showTab(name) {
        document.querySelectorAll('.chat-tab').forEach(b => {
            b.classList.toggle('active', b.dataset.tab === name);
        });
        document.querySelectorAll('.chat-tab-body').forEach(b => {
            b.classList.toggle('active', b.id === 'chat-tab-' + name);
        });
        if (name === 'connection') loadSettings();
        if (name === 'chat') el('chat-input').focus();
    }

    function openDock(open) {
        const dock = el('chat-dock');
        // an inline height from an earlier drag beats the class, so it has to go
        // when closing, or the dock never shuts
        const h = savedHeight();
        dock.style.height = (open && h) ? h + 'px' : '';
        dock.classList.toggle('open', open);
        el('chat-toggle').classList.toggle('open', open);
        if (open) {
            el('chat-input').focus();
            if (!providers.length) loadSettings();
        }
    }

    // the same limits as the desktop viewer: never shorter than this, and always
    // leave a strip of map above the dock
    const MIN_HEIGHT = 140;
    const HEIGHT_KEY = 'pciSeqLiveChatHeight';

    function clamp(h) {
        return Math.max(MIN_HEIGHT, Math.min(h, window.innerHeight - 120));
    }

    /** The height you dragged it to, kept for next time, as the desktop does. */
    function savedHeight() {
        try {
            const h = Number(localStorage.getItem(HEIGHT_KEY));
            return h > 0 ? clamp(h) : 0;
        } catch (e) {
            return 0;
        }
    }

    /** Drag the top edge to make the dock taller or shorter. */
    function initResize() {
        const dock = el('chat-dock');
        const handle = el('chat-resize-handle');
        let dragging = false;

        handle.addEventListener('mousedown', e => {
            dragging = true;
            dock.classList.add('resizing');
            e.preventDefault();
        });
        document.addEventListener('mousemove', e => {
            if (!dragging) return;
            dock.style.height = clamp(window.innerHeight - e.clientY) + 'px';
        });
        document.addEventListener('mouseup', () => {
            if (!dragging) return;
            dragging = false;
            dock.classList.remove('resizing');
            try {
                localStorage.setItem(HEIGHT_KEY, String(parseInt(dock.style.height, 10)));
            } catch (e) {
                // a browser with storage off still resizes, it just forgets
            }
        });
    }

    function initialize() {
        el('chat-toggle').addEventListener('click', () => {
            openDock(!el('chat-dock').classList.contains('open'));
        });
        el('chat-close').addEventListener('click', () => openDock(false));
        el('chat-expand').addEventListener('click', () => {
            // the inline height a drag leaves behind would win over the class
            el('chat-dock').style.height = '';
            el('chat-dock').classList.toggle('expanded');
        });
        el('chat-send').addEventListener('click', send);
        el('chat-save').addEventListener('click', saveSettings);
        el('chat-provider').addEventListener('change', onProviderPicked);
        el('chat-reset').addEventListener('click', () => {
            socket().emit('chat_reset');
            el('chat-messages').innerHTML = '';
            finish();
        });
        document.querySelectorAll('.chat-tab').forEach(b => {
            b.addEventListener('click', () => showTab(b.dataset.tab));
        });

        // enter sends, shift enter is a new line
        el('chat-input').addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send();
            }
        });

        initResize();
    }

    window.pciSeq.chat = { initialize: initialize, handleEvent: onEvent };

})();
