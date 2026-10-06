---
description: The four ways to access the AI agent that explains a run, and what each one uses.
# standard column and margins; only the diagram runs wide, as on the scaling factors demo
pageClass: demo-wide
---

# Accessing the AI agent

pciSeq saves with its results, for every cell and every spot, the terms from which its
assignment was computed. An agent, a language model equipped with functions that read
those files, explains an assignment from them: why a cell was assigned to its class, why
a spot was assigned to one cell rather than another, what value a setting took.

There are four ways to put a question to it. The diagram shows what each one uses.
Choosing a way keeps its parts and fades the rest; the person marks where the question
is typed.

<DemoFrame src="/ways-to-ask.html" wide :min="420"
           title="The ways to ask about a run" />

| Way | When | Reads | Requires |
| :--- | :--- | :--- | :--- |
| An AI assistant with `pciseq-mcp` | once the run has completed | the saved files, from the output folder of the run | the `mcp` extra, see [MCP server](./api/mcp-server.md) |
| An AI assistant connected to [pciSeq Viewer](./viewer/overview.md) | once the run has completed | the saved files, as loaded in pciSeq Viewer | pciSeq Viewer running; its address registered with the assistant once |
| The chat panel in pciSeq Viewer | once the run has completed | the saved files, as loaded in pciSeq Viewer | an API key |
| The chat of the [live viewer](./api/live-viewer.md) | during the run | the fit in progress | an API key |

The agent can also show the source code behind a quantity, for example the adjustment
for anisotropy. Through pciSeq Viewer, from its chat panel or from an AI assistant
connected to it, the file is read from the pciSeq repository on GitHub at the commit
that produced the run, so the code shown is the code that computed the saved numbers;
this needs a network connection, and the request names a file and a commit and carries
nothing from the run. The live viewer's chat reads the installed files, which are the
code running the fit. An AI assistant with `pciseq-mcp` cannot yet be given the code
that produced the run. The [code map](./api/code-map.md) names the function and line for
each quantity.

Where an API key is needed, it is saved on the user's machine. The files `pciSeq.fit`
wrote for the run stay on that machine too; only the question and the results of the
functions are sent to the language model.
