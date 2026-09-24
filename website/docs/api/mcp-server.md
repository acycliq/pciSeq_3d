---
description: Ask an agent about a finished pciSeq run over the Model Context Protocol.
---

# MCP server

The MCP server connects an AI assistant, such as Claude Code or Claude Desktop, to a
finished pciSeq run. The assistant can then answer questions about the run: why a cell
was given its class, why a spot went to the cell it did, how many reads a cell holds,
which spots sit inside it. The connection uses the
[Model Context Protocol](https://modelcontextprotocol.io).

The server reads the run's output folder. It needs `diagnostics.db` and the viewer
files, which `fit` writes by default, and uses `cellData.tsv` and `geneData.tsv` when
they are present. The fitted pickle is not used.

## Installation

The server is an optional extra, since it brings in a web server stack that the rest
of pciSeq does not need:

```bash
pip install "pciSeq_3d[mcp] @ git+https://github.com/acycliq/pciSeq_3d.git@dev_3d"
```

This adds a `pciseq-mcp` command to the path. `pciSeq_3d[all]` installs every extra.

## Registering the server

The agent starts the server itself over stdio. For Claude Code, add it to
`~/.claude.json`:

```json
"mcpServers": {
  "pciSeq": {"type": "stdio", "command": "pciseq-mcp", "args": []}
}
```

For Claude Desktop, the same entry goes in `claude_desktop_config.json`, and the server
then appears in the app under **Connectors**. Desktop applications do not usually share
the shell's `PATH`, so give the full path to the command there; `which pciseq-mcp`
prints it.

A new session of the agent lists the server under its tools. The first call in a
session is `open_run` with the run folder; every other tool refers to the run that is
open.

## Tools

All tools take and return the cell labels of the input segmentation, the same numbers
the viewer shows. The internal labels pciSeq uses when it renumbers a segmentation
never appear, see [Cell identifiers](./working-with-results.md#cell-identifiers).

| Tool | Returns |
| --- | --- |
| `open_run(path)` | Opens a run and returns its size, the pciSeq version that made it, and whether it carries containment data. |
| `cell(label)` | The class probabilities, top genes, total counts and scale factor of one cell. |
| `explain_cell(label, vs_class=None)` | The score of the assigned class against another, split into the gene log-likelihood, the class prior and the spatial term, with the genes that pushed hardest for each side. `vs_class` defaults to the runner up. |
| `explain_spot(spot_id)` | One row per candidate cell plus the background: the six score terms, their sum and the resulting probability. |
| `cell_counts(label, gene=None)` | The reads a cell holds, in total or for one gene. |
| `spots_in_cell(label, gene=None)` | The spots whose pixel falls inside the cell's segmentation mask. |
| `spots_of_cell(label, min_prob=None)` | The spots whose most likely parent is the cell, each with its probability. With `min_prob`, every spot with probability above it. |
| `cell_row(label)` | The `cellData.tsv` row of one cell, value for value. |
| `spot_row(spot_id)` | The `geneData.tsv` row of one spot, value for value. |

`explain_cell` and `explain_spot` return the same numbers as
[`check_cell`](./reference.md#check-cell) and [`check_spot`](./reference.md#check-spot),
which the [Explaining the calls](../explaining-the-calls/overview.md) pages work
through by hand.

### Soft and hard counts

Two of the tools count the reads of a cell and give different numbers on purpose.

`cell_counts` is the sum of the assignment probabilities of the spots, the same
quantity as `CellGeneCount` in `cellData`. It is not a whole number. A spot that
belongs to a cell with probability 0.3 contributes 0.3.

`spots_in_cell` counts the spots whose pixel falls inside the cell's mask. It is a
whole number and no probability enters it. The model scores a spot's position against
the cell centroid, not the mask, so a spot outside every cell can still be assigned to
one, and many are.

Every answer from either tool says which of the two it is.

## Examples

The run is a fit of the Espio section used throughout the documentation. The agent is
asked in plain language; the tool calls and the substance of the answers are shown.

**Why is cell 2413 `048 RHP-COA Ndnf Gaba`?** The agent calls `explain_cell(2413)`:

```
assigned   048 RHP-COA Ndnf Gaba   p = 1.000
compared   047 Sncg Gaba           p = 0.000

                    assigned   compared
gene log-likelihood  -674.94    -708.54
class prior            -4.33      -4.33
spatial term            4.80       0.01

for the assigned class   Ndnf +15.3   Rgs5 +15.3   Ttr +10.7   Vip +4.9
for the compared class   Npy   -7.6   Kit  -7.2   Rgs10 -3.9
```

The gene log-likelihood decides it, by 34 nats. Ndnf and Rgs5 each contribute 15 nats
in favour; Npy and Kit argue for `Sncg Gaba` but not by enough.

**Why did spot 1642419 go to cell 18223?** The agent calls `explain_spot(1642419)`
and reports the Synpr spot at 0.74 on cell 18223, 0.13 on cell 21574 and 0.10 on cell
17371, matching [Table 3.1](../explaining-the-calls/why-a-spot-got-its-cell.md#table-3-1),
with the six score terms behind each number.

**How many Ndnf reads does cell 2413 have?** `cell_counts(2413, gene='Ndnf')` returns
7.77, with the note that this is a soft count, the sum of the assignment probabilities
of the Ndnf spots.

**Which spots belong to cell 18223?** `spots_of_cell(18223)` returns the 31 spots
whose most likely parent is the cell, with probabilities from 0.81 down to 0.295, and
notes that the lowest is well under one half: the cell is those spots' best guess,
not a certainty. `cell_row(18223)` lists 542 spots under `spot_id`, every spot with
probability above 0.0001 on the cell. Their probabilities sum to the cell's 38.5
counts. The two lists answer different questions.

## Notes

Runs made before September 2026 have no `inside_cell` column, so `spots_in_cell`
refuses with a message saying so. The other tools work on any run that has
`diagnostics.db`.

Probabilities in `cellData.tsv` and `geneData.tsv` are kept to three decimals, and
`cell_row` and `spot_row` return what the files hold. `explain_spot` recomputes them
at full precision, so it is the tool to use when a probability below 0.0005 matters.

The tools are plain Python functions in `pciSeq.src.mcp.tools`, usable without the
server:

```python
from pciSeq.src.mcp.tools import open_run
run = open_run('<output_path>')
run.explain_cell(2413)
```
