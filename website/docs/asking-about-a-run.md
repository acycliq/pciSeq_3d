---
description: The four ways to ask an AI agent about a run, and what each one uses.
# standard column and margins; only the diagram runs wide, as on the scaling factors demo
pageClass: demo-wide
---

# Asking about a run

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

The two assistant ways use the language model of the assistant. The two chats take an
API key from Anthropic. In every case the saved files stay on the user's machine; only
the question and the results of the functions are sent to the language model.
