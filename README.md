# ContextFlow

ContextFlow is a context-first Python framework for building transparent LLM agents.

## Goals

- Keep raw `messages[]` fully inspectable
- Compose context explicitly from reusable nodes
- Parse model responses into typed objects
- Add persistence, memory, and observability incrementally

## Quickstart

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
uvicorn contextflow.api.main:app --reload
```

## Qwen3 Examples (OpenAI-Compatible)

Install one backend (or both):

```bash
pip install openai
pip install litellm
```

Set endpoint env vars (DashScope example):

```bash
set OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
set DASHSCOPE_API_KEY=sk-xxx
set QWEN_MODEL=qwen-flash
```

Local OpenAI-compatible servers still work (set `OPENAI_BASE_URL` + `OPENAI_API_KEY`).

One-time agent call:

```bash
python examples/agent_once_qwen.py
```

Edit variables at the top of the file (`BACKEND`, `MODEL`, `BASE_URL`, `API_KEY`, `USER_INPUT`) for your own setup.

Multi-turn conversation call:

```bash
python examples/agent_stream_qwen.py
```

This script starts a chat loop and keeps conversation history across turns. Type `exit` to stop.

You can now skip manual `create_client(...)` wiring and let `Agent` resolve the client from provider fields.
You can also merge backend and model as `backend/model`, e.g. `openai/qwen3-max`:

```python
agent = Agent(
	model="openai/qwen3-max",
	name="chat_agent",
	description="Multi-turn chat assistant.",
	instruction="You are concise and helpful.",
	base_url=os.getenv("QWEN_BASE_URL"),
	api_key=os.getenv("QWEN_API_KEY"),
)

result = await agent.run_once(user_input="hello")
```

`run_once(..., llm_client=...)` is still supported when you want to inject a custom client.

## Simple Agent Definition (ADK-style)

ContextFlow now includes a lightweight `Agent` abstraction for reusable definitions:

```python
from contextflow import Agent

def get_current_time(city: str) -> dict:
	"""Returns the current time in a specified city."""
	return {"status": "success", "city": city, "time": "10:30 AM"}

root_agent = Agent(
	model="openai/qwen-flash",
	name="root_agent",
	description="Tells the current time in a specified city.",
	instruction="You are a helpful assistant that tells the current time in cities.",
	tools=[get_current_time],
)
```

Runnable example:

```bash
python examples/simple_agent_define.py
```

## Web Debug CLI (`contextflow web`)

Use the debug web UI against a script-defined root agent:

```bash
contextflow web examples/agents/simple_agent_define.py
```

For multi-agent scripts, define your main coordinator as `root_agent` (recommended),
or pass an explicit variable path:

```bash
contextflow web examples/agents/multi_agent_async.py root_agent
contextflow web examples/agents/multi_agent_async.py --agent orchestrator
```

Selection behavior:

- If provided, `<agent_path>` / `--agent` is used directly.
- Otherwise, auto-discovery prefers `root_agent`, then `agent`, then `orchestrator`.
- If none of those names exists, the first declared module-level `Agent` is used.

Multi-agent debug tip:

- If your script has custom orchestration logic, expose `async def debug_chat(message: str) -> str`.
- `contextflow web` will call this handler so the web chat follows the same first-hop flow as your script.
- The debug UI tags each captured LLM call with `agent_name` and renders per-agent chat/context traces.

## Sandbox (Isolated Workspace + Agent Attachment)

ContextFlow includes `AgentSandbox` for isolated file/code operations and easy agent binding.

Use `AgentSandbox.from_agent_sandbox(...)` for strong runtime isolation via Kubernetes Agent Sandbox.
`AgentSandbox.create(...)` is local workspace isolation for development convenience.

Local workspace mode:

```python
from contextflow import Agent, AgentSandbox

sandbox = AgentSandbox.create("examples/_sandbox_smoke")
sandbox.write_text("hello.txt", "hi")

agent = Agent(
	model="qwen-flash",
	name="sandboxed_agent",
	description="Assistant with sandbox tools",
	instruction="Use sandbox tools to inspect and edit files.",
)

sandboxed_agent = sandbox.attach_agent(agent)
print([tool.name for tool in sandboxed_agent.tool_specs() if tool.name.startswith("sandbox_")])
```

Kubernetes agent-sandbox mode (using `k8s-agent-sandbox`):

```python
from contextflow import AgentSandbox

with AgentSandbox.from_agent_sandbox(
	template_name="python-sandbox-template",
	namespace="default",
	gateway_name=None,
) as sandbox:
	out = sandbox.run("echo 'hello from k8s sandbox'")
	print(out.stdout)
```

Install optional dependency:

```bash
pip install -e ".[sandbox]"
```

Runnable example:

```bash
python examples/sandbox_attach_agent.py
```

Interactive code-writing chat example:

```bash
python examples/sandbox_chat_codegen.py
```

Ask in natural language, e.g. `create app.py that prints hello`, and the agent will use sandbox tools to write/update files in `examples/_sandbox_smoke`.

## Agent State Machine

Each `Agent` can optionally carry an `AgentStateMachine` — a lightweight finite-state machine that lets you:

- **Gate execution**: only allow `run_once()` in specific states (e.g. `"running"`)
- **Control flow**: transition between states with guards that can block invalid moves
- **React to changes**: register `on_enter` / `on_exit` / `on_change` hooks (sync or async)
- **Force overrides**: bypass transition rules with `force_transition()` for recovery paths
- **Inspect history**: every transition is logged with a timestamp

### Quick start

```python
from contextflow import Agent, AgentStateMachine

# 1. Define the state machine
sm = AgentStateMachine(
    initial="idle",
    transitions={
        "idle":    ["running"],
        "running": ["paused", "done", "error"],
        "paused":  ["running", "done"],
        "error":   ["idle"],
    },
)

# Only allow LLM calls when the agent is in "running" state
sm.allow_run_when("running")

# 2. Register hooks (decorators, sync or async)
@sm.on_enter("running")
def start_running(old, new, ctx):
    print(f"Agent started! (was {old})")

@sm.on_enter("error")
def on_error(old, new, ctx):
    print(f"ERROR from {old}: {ctx.get('reason', '?')}")

# 3. Register guards (return truthy to allow, falsy to block)
@sm.guard("running", "done")
def must_have_output(from_st, to_st, ctx):
    return bool(ctx.get("output"))

# 4. Attach to agent
agent = Agent(
    model="openai/qwen-flash",
    name="my_agent",
    description="Stateful assistant",
    instruction="You are helpful.",
    state_machine=sm,
)
```

### Using the state machine at runtime

```python
# Read current state
print(agent.state)                        # "idle"

# Transition (validates table + runs guards)
await agent.transition_to("running")      # "running"

# run_once() is now allowed
result = await agent.run_once(user_input="Hello")

# Transition with context (guards can inspect it)
await agent.transition_to("done", context={"output": result.output_text})

# Force-transition (skips table & guards, hooks still fire)
await agent.force_transition("idle")

# Inspect full history
for entry in agent.state_machine.history:
    print(f"  {entry.from_state} → {entry.to_state} at {entry.timestamp}")
```

### API summary

| Method / Property | Description |
|---|---|
| `AgentStateMachine(initial, transitions)` | Create a machine. Pass `transitions=None` for open mode (any transition). |
| `sm.allow_run_when(*states)` | Restrict `run_once()` to these states only. No args = no restriction. |
| `sm.current` | Current state string. |
| `sm.can_transition(target)` | Check if `target` is reachable (ignoring guards). |
| `sm.can_run()` | Check if `run_once()` is allowed. |
| `await sm.transition_to(state, context=)` | Transition with validation + guards + hooks. |
| `await sm.force_transition(state, context=)` | Skip table & guards; hooks still fire. |
| `sm.history` | List of `StateTransition` records. |
| `sm.reset(state=)` | Clear history and optionally set a new state. |
| `@sm.on_enter(state)` | Decorator: hook fires when entering a state. |
| `@sm.on_exit(state)` | Decorator: hook fires when leaving a state. |
| `@sm.on_change` | Decorator: hook fires on every transition. |
| `@sm.guard(from, to)` | Decorator: must return truthy to allow the transition. |

### Exceptions

| Exception | When |
|---|---|
| `InvalidTransition` | Transition not in the table. |
| `TransitionBlockedByGuard` | A guard returned falsy. |
| `RunBlockedByState` | `run_once()` called in a non-allowed state. |

All inherit from `StateError`.

Runnable examples:

```bash
python examples/agent_state_machine.py          # basic hooks + guards
python examples/agent_stateful_assistant.py      # multi-state LLM workflow
```

## Package layout

- `contextflow.core` — context data structures, composer, parser, pruning
- `contextflow.agents` — agent definitions + `AgentStateMachine`
- `contextflow.providers` — provider config + OpenAI-compatible/LiteLLM adapters
- `contextflow.streaming` — normalized streaming event model
- `contextflow.state` — session manager and async turn engine
- `contextflow.persistence` — storage interfaces and in-memory adapters
- `contextflow.runners` — async agent runner and interceptors
- `contextflow.lens` — transparent request/response tracing
- `contextflow.api` — FastAPI bootstrap

## Status

Initial scaffold implemented from `PROJECT.md` roadmap with extensible interfaces.
