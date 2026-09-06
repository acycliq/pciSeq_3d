# Vendored browser libraries

The viewer used to pull these from CDNs, which meant it did not work offline and,
in deck.gl's case, tracked `@latest` so an upstream release could break it without
anything changing here.

| file | version | source |
| --- | --- | --- |
| `socket.io.min.js` | 4.5.4 | https://cdn.socket.io/4.5.4/socket.io.min.js |
| `d3.v7.min.js` | 7 | https://d3js.org/d3.v7.min.js |
| `deck.gl.min.js` | 9.4.0 | https://unpkg.com/deck.gl@9.4.0/dist.min.js |

To update one, download the new file over the old one and change the version in
this table. Keep the versions pinned, no `@latest`.
