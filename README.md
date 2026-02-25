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
	model="qwen-flash",
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

## Package layout

- `contextflow.core` — context data structures, composer, parser, pruning
- `contextflow.agents` — simple reusable agent definitions
- `contextflow.providers` — provider config + OpenAI-compatible/LiteLLM adapters
- `contextflow.streaming` — normalized streaming event model
- `contextflow.state` — session manager and async turn engine
- `contextflow.persistence` — storage interfaces and in-memory adapters
- `contextflow.runners` — async agent runner and interceptors
- `contextflow.lens` — transparent request/response tracing
- `contextflow.api` — FastAPI bootstrap

## Status

Initial scaffold implemented from `PROJECT.md` roadmap with extensible interfaces.
