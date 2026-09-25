"""The MCP server: the tools in tools.py, exposed over the protocol.

This is the only file in pciSeq that imports mcp. All the logic lives in tools.py,
this just registers each function and keeps track of which run is open, so the tools
stay usable from a notebook or the viewer with no mcp anywhere.

It is an optional extra, so a plain pciSeq install does not pull in a web server:

    pip install "pciSeq_3d[mcp] @ git+https://github.com/acycliq/pciSeq_3d.git@dev_3d"

That puts a `pciseq-mcp` command on the path. Point Claude Code, Claude Desktop or any
other MCP client at it over stdio, for example in ~/.claude.json:

    "mcpServers": {"pciSeq": {"type": "stdio", "command": "pciseq-mcp", "args": []}}

Use the full path to the command if the client does not share your shell's PATH, which
is usually the case for desktop apps. The docstrings below are what the agent reads to decide which tool to call,
so they say what the numbers mean, not how they are computed.
"""
import functools
import io
import json
from pathlib import Path
from typing import Optional

try:
    from mcp.server.mcpserver import Image, MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ModuleNotFoundError as e:  # pragma: no cover
    # the tools in tools.py work without any of this, only the server needs it
    raise ModuleNotFoundError(
        'the pciSeq MCP server needs the mcp library, which is an optional extra so\n'
        'that a plain install does not pull in a web server. Add it with:\n'
        '    pip install "pciSeq_3d[mcp]"\n'
        'or straight from github:\n'
        '    pip install "pciSeq_3d[mcp] @ git+https://github.com/acycliq/pciSeq_3d.git@dev_3d"'
    ) from e

from . import docs as _docs
from .tools import Run, open_run as _open

server = MCPServer(
    name='pciSeq',
    instructions=(
        'Tools for inspecting a finished pciSeq run. Call open_run first with the run '
        'folder, then ask about cells and spots. Cell labels are always the labels of '
        'the segmentation the user knows, never internal indices. Counts are soft, '
        'weighted by assignment probability, unless a tool says it is a hard count. '
        'When you explain a result, explain it the way a teacher would: say what '
        'happened and why in plain words, and use the numbers to support the story '
        'rather than as the story. explain_cell returns a narrative field written that '
        'way; build on it, do not just repeat the table.'
    ),
)

# the run the tools are answering about. One at a time, set by open_run.
_run: Optional[Run] = None


def _need_run() -> Run:
    if _run is None:
        raise RuntimeError('no run is open, call open_run with the run folder first')
    return _run


# the errors tools.py raises on purpose, with a message meant for whoever asked:
# no such cell, no such gene, an old run without containment data, no run open.
# mcp hides the message of any other exception from the client, so the agent would
# only see 'Error executing tool spots_in_cell' and never learn what to do about it.
# Re-raising as ToolError sends the message through.
_EXPECTED = (KeyError, ValueError, NotImplementedError, RuntimeError, FileNotFoundError)


def _tool(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _EXPECTED as e:
            # KeyError repr()s its message, which puts quotes round it
            msg = e.args[0] if e.args else str(e)
            raise ToolError(str(msg)) from e
    return server.tool()(wrapped)


@_tool
def open_run(path: str) -> dict:
    """Open a finished pciSeq run so the other tools can answer questions about it.

    Give the run's output folder, or any folder above it that contains exactly one
    run. Returns a summary: how many cells, spots, genes and classes, the pciSeq
    version that made it, and whether it carries containment data (runs before
    September 2026 do not, and spots_in_cell will say so).
    """
    global _run
    _run = _open(path)
    return _run.summary()


@_tool
def run_info() -> dict:
    """What produced the open run and how it ended: pciSeq version and commit, the
    resolved settings (mrf_beta, rTheta, Inefficiency, nNeighbors, voxel_size and the
    rest), the number of iterations, and whether the loop converged, with a sentence
    saying so. Use it for 'what settings did this run use' and 'did it converge'.
    Older runs carry only the version and commit and the answer says so."""
    return _need_run().run_info()


@_tool
def cell(label: int) -> dict:
    """The headline facts about one cell: its class probabilities, its top genes, its
    total counts and its scale factor theta.

    label is the cell's number in the segmentation, the one shown in the viewer.
    Counts are soft, each spot contributes its probability of belonging to the cell.
    """
    return _need_run().cell(label)


@_tool
def explain_cell(label: int, vs_class: Optional[str] = None, top_n: int = 10) -> dict:
    """Why a cell was given its class, gene by gene.

    Compares the assigned class against another (the runner up by default, or
    vs_class). Reports the three parts of the score for each, the gene
    log-likelihood, the class prior and the spatial term, and lists the genes that
    pushed hardest for each side. The narrative field tells the story in plain
    words, with the evidence as odds rather than units.

    Use this for questions like 'why is cell 2413 Ndnf Gaba' or 'why is cell 18223
    not CA1'. label is the segmentation label.
    """
    return _need_run().explain_cell(label, vs_class=vs_class, top_n=top_n)


@_tool
def explain_spot(spot_id: int) -> dict:
    """Why a spot was assigned to the cell it was, term by term.

    One row per candidate cell plus the background: the Gaussian fit to the cell
    centroid, the class expression term, the cell scale, the cell-gene scale, the
    gene efficiency, the inside-cell bonus, their sum, and the resulting probability.
    The background row carries the gene's misread density instead.

    A spot can sit outside every cell and still go to one, because the position is
    scored against the centroid, not the mask. spot_id is the index in geneData.
    """
    return _need_run().explain_spot(spot_id)


@_tool
def cell_counts(label: int, gene: Optional[str] = None) -> dict:
    """How many reads a cell holds, in total or for one gene.

    These are SOFT counts: each spot contributes its probability of belonging to the
    cell, so they are estimates and not whole numbers. For the number of spots
    physically inside the cell's boundary use spots_in_cell instead.
    """
    return _need_run().cell_counts(label, gene=gene)


@_tool
def spots_in_cell(label: int, gene: Optional[str] = None) -> dict:
    """How many spots physically sit inside a cell's segmentation mask.

    This is a HARD count with no probabilities: a spot either falls inside the mask
    or it does not. It is a different number from cell_counts, because the model can
    assign a spot to a cell it is not inside, and vice versa. Needs a run made after
    September 2026; older runs raise with a clear message.
    """
    return _need_run().spots_in_cell(label, gene=gene)


@_tool
def spots_of_cell(label: int, min_prob: Optional[float] = None) -> dict:
    """Which spots belong to a cell, with their probabilities, sorted highest first.

    Two definitions, and the answer says which it used. Without min_prob: the spots
    whose MOST LIKELY parent is this cell, the argmax. That is not a hard assignment,
    the lowest probability in the list can be well under 0.5. With min_prob, say
    0.0001: EVERY spot with probability above it on this cell, whose probabilities
    add up to the cell's soft counts. The second list is usually many times longer
    than the first.

    Use this for 'which spots are assigned to cell 18223' or 'list the spots of
    cell 18223 with their probabilities'.
    """
    return _need_run().spots_of_cell(label, min_prob=min_prob)


@_tool
def cell_row(label: int) -> dict:
    """The cellData.tsv row of one cell, value for value: Cell_Num, X, Y, Z,
    Genenames, CellGeneCount, spot_id, ClassName, Prob. Use this when the question
    is about what the saved file says for a cell. The counts are soft."""
    return _need_run().cell_row(label)


@_tool
def spot_row(spot_id: int) -> dict:
    """The geneData.tsv row of one spot, value for value: gene, position, plane,
    neighbour, neighbour_array, neighbour_prob, omp_score, omp_intensity,
    is_hard_misread and, on runs from September 2026 on, inside_cell."""
    return _need_run().spot_row(spot_id)


@_tool
def cell_image(label: int, context: bool = False, plane: Optional[int] = None,
               width: int = 1200, channel: Optional[str] = None, neighbours: bool = False,
               save_as: Optional[str] = None, mbtiles: Optional[str] = None) -> list:
    """A picture of a cell on the background image (DAPI or another stain), for 'show
    me cell 18223', 'show me cell 18223 on the DAPI' or 'where is cell 18223 in the
    section'.

    context=False gives a close-up: the cell outlined in red, every other cell on
    that plane in blue, the nuclei underneath. context=True gives the whole section
    with a ring round the cell, to show where it sits in the tissue. Ask for both
    when the user wants to see a cell; that is the pair on the why-a-cell-got-its-type
    docs page.

    Which cells are outlined depends on the question. neighbours=False, every cell
    near it, is right for 'show me the cell' or 'is it segmented properly'.
    neighbours=True outlines only the cells the spatial term of the model listened
    to, which is what 'why did its neighbours make it this class' or 'show me its
    neighbours' needs; the answer lists them and says which sit on another plane.

    The image is stitched from the viewer's tile pyramid (the .mbtiles in
    viewer_data), so it is a close visual copy, not the raw pixels. One plane at a
    time: the cell's 3D neighbours on other planes are not drawn, which is why a
    cell can have fewer blue cells around it than neighbours. The plane is the
    centroid's when the run knows its voxel size, else the one where the cell is
    biggest; pass plane to choose. save_as writes the png to that path as well,
    which is how a user in a terminal gets to see it.

    A run can have more than one background image, for example DAPI and GCaMP, one
    .mbtiles each. channel picks one by its name ('DAPI', 'GCaMP'); part of the name
    is enough. If the user did not say which image and the run has several, do not
    guess: the tool refuses and lists them, so ask the user which one they want.
    If the run has only one background image it is always used, whatever channel
    says, because the file cannot tell which stain it is; background_note in the
    answer says so, pass that on to the user.
    """
    im, info = _need_run().cell_image(label, context=context, plane=plane, width=width,
                                      channel=channel, neighbours=neighbours,
                                      mbtiles=mbtiles)
    return _picture(im, info, save_as)


@_tool
def plane_image(plane: Optional[int] = None, bbox: Optional[list] = None,
                width: int = 1200, channel: Optional[str] = None,
                save_as: Optional[str] = None, mbtiles: Optional[str] = None) -> list:
    """The background image (DAPI or another stain) of one plane with nothing drawn
    on it, for 'show me the whole image', 'show me the DAPI of plane 54' or 'show me
    the region around x 5000 to 6000, y 500 to 1200'.

    With no bbox it is the whole plane, untrimmed. bbox is [x0, y0, x1, y1] in image
    pixels, the same coordinates as the cells and spots. The plane defaults to the
    middle of the stack. For a picture about one cell use cell_image instead, it
    finds the plane and the window for you and draws the outlines.

    Stitched from the viewer's tile pyramid (the .mbtiles in viewer_data), so it is
    a close visual copy, not the raw pixels. The answer says the scale, so a point
    of the image can be placed on the picture. save_as writes the png to that path
    as well, which is how a user in a terminal gets to see it.

    A run can have more than one background image, for example DAPI and GCaMP, one
    .mbtiles each. channel picks one by its name ('DAPI', 'GCaMP'); part of the name
    is enough. If the user did not say which image and the run has several, do not
    guess: the tool refuses and lists them, so ask the user which one they want.
    If the run has only one background image it is always used, whatever channel
    says, because the file cannot tell which stain it is; background_note in the
    answer says so, pass that on to the user.
    """
    im, info = _need_run().plane_image(plane=plane, bbox=bbox, width=width,
                                       channel=channel, mbtiles=mbtiles)
    return _picture(im, info, save_as)


def _picture(im, info, save_as):
    """What an image tool hands back: the png itself, so the agent can look at it,
    and the facts as text. Written to save_as too when asked."""
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    if save_as:
        path = Path(save_as).expanduser()
        path.write_bytes(buf.getvalue())
        info['saved_as'] = str(path)
    return [Image(data=buf.getvalue(), format='png'), json.dumps(info, indent=2)]


@_tool
def docs(query: str, n: int = 5) -> dict:
    """Search the pciSeq documentation and return the paragraphs that match.

    Use it to check how something works before explaining it: what a term means,
    what a config option does, how a file is laid out. Each hit names the page it
    came from; read the whole page as the resource pciseq-docs://<page> when the
    paragraph is not enough. Plain keyword search, no index.
    """
    hits = _docs.search_docs(query, n=n)
    for h in hits:
        h['resource'] = 'pciseq-docs://' + h['page']
    return {'query': query, 'hits': hits,
            'note': 'no docs pages found on this machine' if _docs.docs_root() is None else ''}


@server.resource('pciseq-docs://index', name='pciSeq documentation index',
                 description='Every documentation page, with its title. Read a page as '
                             'pciseq-docs://<page>.', mime_type='text/markdown')
def docs_index() -> str:
    lines = ['# pciSeq documentation', '']
    for page in _docs.list_pages():
        lines.append('- pciseq-docs://%s  %s' % (page, _docs.page_title(_docs.read_page(page))))
    return '\n'.join(lines) if len(lines) > 2 else 'no docs pages found on this machine'


# One resource per page, registered at import. A URI template such as
# pciseq-docs://{path} would be neater but matches a single path segment, so every
# page inside a folder (api/, the-model/, ...) would be unreachable. Static URIs
# also let the agent see every page with its title in list_resources.
def _page_reader(page):
    # a static resource's function must take no arguments, so the page is closed
    # over here rather than passed in
    def reader() -> str:
        return _docs.read_page(page)
    return reader


def _register_pages():
    for page in _docs.list_pages():
        title = _docs.page_title(_docs.read_page(page)) or page
        server.resource('pciseq-docs://' + page, name=title,
                        description='pciSeq documentation, %s' % page,
                        mime_type='text/markdown')(_page_reader(page))


_register_pages()


def main() -> None:
    server.run(transport='stdio')


if __name__ == '__main__':
    main()
