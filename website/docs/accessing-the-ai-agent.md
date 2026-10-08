---
description: The four ways to access the AI agent that explains a run, and what each one uses.
# standard column and margins; only the diagram runs wide, as on the scaling factors demo
pageClass: demo-wide
---

<script setup>
import { ref } from 'vue'

// the path of a question: which step is picked, 0 is all of them
const step = ref(0)
const steps = [
  { name: 'All', says: 'The path of one question. Dashed arrows cross the network to the language model; solid arrows stay on the user\'s machine.' },
  { name: '1 Question', says: 'The question is sent to the language model with the list of functions: the name, the description and the arguments of each.' },
  { name: '2 Function to call', says: 'The language model picks one from the list and replies with its name and arguments, here explain_cell for cell 5016.' },
  { name: '3 Function runs', says: 'The function is run on the user\'s machine and reads the saved files.' },
  { name: '4 Result', says: 'The result of the function is sent to the language model.' },
  { name: '5 Answer', says: 'The language model writes the answer from that result.' },
]
const lit = (...on) => step.value === 0 || on.includes(step.value)

// connecting an AI assistant: which of the two ways is shown
const way = ref('viewer')
</script>

# Accessing the AI agent

pciSeq saves with its results, for every cell and every spot, the terms from which its
assignment was computed. An agent, a language model equipped with functions that read
those files, explains an assignment from them: why a cell was assigned to its class, why
a spot was assigned to one cell rather than another, what value a setting took.

There are four ways to put a question to it. The diagram shows what each one uses.
Choosing a way keeps its parts and fades the rest; the person marks where the question
is typed.

<DemoFrame src="/agent_access.html" wide :min="420"
           title="The ways to ask about a run" />

| Way | When | Reads | Requires |
| :--- | :--- | :--- | :--- |
| An AI assistant with `pciseq-mcp` | once the run has completed | the saved files, from the output folder of the run | two steps, see <a href="#connecting-an-ai-assistant" @click="way = 'mcp'">With pciseq-mcp</a> |
| An AI assistant connected to [pciSeq Viewer](./viewer/overview.md) | once the run has completed | the saved files, as loaded in pciSeq Viewer | two steps, see <a href="#connecting-an-ai-assistant" @click="way = 'viewer'">With pciSeq Viewer</a> |
| The chat panel in pciSeq Viewer | once the run has completed | the saved files, as loaded in pciSeq Viewer | an API key |
| The chat of the [live viewer](./api/live-viewer.md) | during the run | the fit in progress | an API key |

## Connecting an AI assistant

An AI assistant such as Claude Code is told once where the pciSeq functions are. This is
called registering, and it is one command. What is registered depends on whether the run
is open in pciSeq Viewer.

| | The run is open in pciSeq Viewer | Without pciSeq Viewer |
| :--- | :--- | :--- |
| `mcp` extra | not needed | required |
| Registered with the assistant | the address of pciSeq Viewer | the `pciseq-mcp` program |
| Then | start the assistant and ask | start the assistant and name the output folder of the run in the question |
| Steps | tab "With pciSeq Viewer" | tab "With pciseq-mcp" |

"Without pciSeq Viewer" covers a machine where it is not installed or not used, for
example a remote machine reached over ssh.

The two are independent: each has its own command, and neither requires the other. The
same applies to any other AI application that supports the Model Context Protocol; only
the form of the registration command differs.

<div class="way-box">
<div class="way-tabs" role="group" aria-label="Way to connect an AI assistant">
  <button type="button" :class="{ on: way === 'viewer' }" :aria-pressed="way === 'viewer'" @click="way = 'viewer'">With pciSeq Viewer</button>
  <button type="button" :class="{ on: way === 'mcp' }" :aria-pressed="way === 'mcp'" @click="way = 'mcp'">With pciseq-mcp</button>
</div>

<div v-show="way === 'viewer'" class="way-panel">

1. Register the address of pciSeq Viewer with the assistant, once.

   ```bash
   claude mcp add --transport http pciseq-viewer http://127.0.0.1:8317/mcp
   ```

2. Open pciSeq Viewer and load the run, by selecting its `viewer_data` folder.

Then start the assistant and ask. Neither `pciseq-mcp` nor the `mcp` extra is used.

</div>

<div v-show="way === 'mcp'" class="way-panel">

1. Install pciSeq with the `mcp` extra. This adds the `pciseq-mcp` program.

   ```bash
   pip install "pciSeq_3d[mcp] @ git+https://github.com/acycliq/pciSeq_3d.git@dev_3d"
   ```

2. Register the program with the assistant, once. The assistant then starts it by itself
   when needed.

   ```bash
   claude mcp add --scope user pciseq -- pciseq-mcp
   ```

Then start the assistant and name the output folder of the run in the question. The
commands for other assistants are on the [MCP server](./api/mcp-server.md) page.

</div>
</div>

## The path of a question

In every way a question is answered in the same five steps. The language model does
not read the files of the run. It names a function, and the function is run on the
user's machine. Choosing a step keeps its parts and fades the rest.

The language model has no knowledge of these functions of its own. The list of them is
sent with every question, and the language model chooses from the descriptions in it.

<figure class="diagram">
<div class="aq-steps" role="group" aria-label="Step to show">
  <button v-for="(b, i) in steps" :key="i" type="button" :class="{ on: step === i }" :aria-pressed="step === i" @click="step = i">{{ b.name }}</button>
</div>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 392" role="img" aria-label="The path of one question: it is sent to the language model with the list of functions, the language model names one, that function is run on the user's machine and reads the saved files, and its result goes back for the answer">
  <rect class="aq-machine" x="14" y="26" width="536" height="352" rx="7" />
  <text class="aq-cap" x="16" y="16">THE USER'S MACHINE</text>
  <text class="aq-cap" x="670" y="16" text-anchor="middle">OVER THE NETWORK</text>
  <g :class="{ 'aq-dim': !lit(1, 2, 3, 4, 5) }">
    <rect class="aq-box" x="34" y="46" width="196" height="150" rx="4" />
    <text class="aq-title" x="132.0" y="117.0" text-anchor="middle">Chat or AI assistant</text>
    <text class="aq-sub" x="132.0" y="136.0" text-anchor="middle">where the question is typed</text>
  </g>
  <g :class="{ 'aq-dim': !lit(1, 2, 4, 5) }">
    <rect class="aq-box aq-llm-box" x="590" y="46" width="160" height="150" rx="4" />
    <text class="aq-title" x="670.0" y="126.0" text-anchor="middle">Language model</text>
  </g>
  <g :class="{ 'aq-dim': !lit(1) }">
    <line class="aq-net" x1="230" y1="72" x2="582" y2="72" />
    <polygon class="aq-net-head" points="590,72 581,67.5 581,76.5" />
    <text class="aq-label" x="410.0" y="65" text-anchor="middle"><tspan class="aq-num">1</tspan>  question, with the list of functions</text>
  </g>
  <g :class="{ 'aq-dim': !lit(2) }">
    <line class="aq-net" x1="590" y1="107" x2="238" y2="107" />
    <polygon class="aq-net-head" points="230,107 239,102.5 239,111.5" />
    <text class="aq-label" x="410.0" y="100" text-anchor="middle"><tspan class="aq-num">2</tspan>  function to call</text>
  </g>
  <g :class="{ 'aq-dim': !lit(4) }">
    <line class="aq-net" x1="230" y1="142" x2="582" y2="142" />
    <polygon class="aq-net-head" points="590,142 581,137.5 581,146.5" />
    <text class="aq-label" x="410.0" y="135" text-anchor="middle"><tspan class="aq-num">4</tspan>  result of the function</text>
  </g>
  <g :class="{ 'aq-dim': !lit(5) }">
    <line class="aq-net" x1="590" y1="177" x2="238" y2="177" />
    <polygon class="aq-net-head" points="230,177 239,172.5 239,181.5" />
    <text class="aq-label" x="410.0" y="170" text-anchor="middle"><tspan class="aq-num">5</tspan>  answer</text>
  </g>
  <g :class="{ 'aq-dim': !lit(1, 2, 3) }">
    <rect class="aq-box" x="34" y="244" width="330" height="120" rx="4" />
    <text class="aq-title" x="48" y="266">Functions</text>
    <text class="aq-sub" x="124" y="266">a name, a description and arguments each</text>
    <rect class="aq-pick" :class="{ on: step === 2 || step === 3 }" x="42" y="276" width="314" height="22" rx="3" />
    <text class="aq-mono" x="50" y="292">explain_cell</text>
    <text class="aq-sub" x="146" y="292">why a cell was assigned to its class</text>
    <text class="aq-mono" x="50" y="314">explain_spot</text>
    <text class="aq-sub" x="146" y="314">why a spot was assigned to its cell</text>
    <text class="aq-mono" x="50" y="336">find_cells</text>
    <text class="aq-sub" x="146" y="336">the cells matching a filter</text>
    <text class="aq-sub" x="50" y="356">and the rest of them</text>
  </g>
  <g :class="{ 'aq-dim': !lit(3) }">
    <line class="aq-local" x1="132" y1="196" x2="132" y2="236" />
    <polygon class="aq-local-head" points="132,244 127.5,235 136.5,235" />
    <text class="aq-label" x="142" y="225"><tspan class="aq-num">3</tspan>  runs the function</text>
    <line class="aq-local" x1="364" y1="304" x2="402" y2="304" />
    <polygon class="aq-local-head" points="410,304 401,299.5 401,308.5" />
    <text class="aq-label" x="386" y="296" text-anchor="middle">reads</text>
    <rect class="aq-box aq-files" x="410" y="281" width="124" height="46" rx="4" />
    <text class="aq-title" x="472.0" y="309.0" text-anchor="middle">Saved files</text>
  </g>
</svg>
<figcaption>{{ steps[step].says }}</figcaption>
</figure>

1. The question is sent to the language model, together with the list of functions it
   may call. An entry of the list is a name, a description and the arguments the
   function takes. This is the entry for `explain_cell`, the description shortened:

   ```
   name         explain_cell
   description  Why a cell was given its class, gene by gene. [...] Use this for
                questions like 'why is cell 2413 Ndnf Gaba' or 'why is cell 18223
                not CA1'.
   arguments    label     The cell label, as in the segmentation.
                vs_class  Class to compare against. Defaults to the runner up.
                top_n     How many genes to list for each side, default 10.
   ```

   An AI assistant receives the same list when it connects to `pciseq-mcp` or to
   pciSeq Viewer.
2. The language model replies with the function to call and its arguments, for example
   `explain_cell` for cell 5016.
3. The function is run on the user's machine. It reads the saved files, or, in the
   live viewer's chat, the fit in progress.
4. The result of the function is sent to the language model.
5. The language model writes the answer from that result.

Steps 2 to 4 are repeated when an answer needs more than one function.

In every way the agent can also show the source code behind a quantity, for example
the adjustment for anisotropy, and in every way the code shown is the code that produced
the numbers. The live viewer's chat reads the installed files, which are the code
running the fit. For a completed run, `pciseq-mcp` and pciSeq Viewer read the file at
the commit the run recorded, from the pciSeq repository on GitHub. This needs a network
connection; the request names a file and a commit and carries nothing from the run. A run that does not record its commit is refused rather than
shown another version. The [code map](./api/code-map.md) names the function and line
for each quantity.

Where an API key is needed, it is saved on the user's machine. The files `pciSeq.fit`
wrote for the run stay on that machine too; only the question and the results of the
functions are sent to the language model.
