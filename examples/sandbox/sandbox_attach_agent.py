import asyncio

from contextflow import Agent, AgentSandbox


async def main() -> None:
    sandbox = AgentSandbox.create("examples/_sandbox_smoke")

    sandbox.write_text("notes/todo.txt", "buy milk")
    print([entry.name for entry in sandbox.list_files()])

    agent = Agent(
        model="qwen-flash",
        name="sandboxed_helper",
        description="Assistant with sandbox tools",
        instruction=(
            "You can inspect and modify files using sandbox tools. "
            "Always return tool-call-ready JSON arguments."
        ),
    )

    sandboxed_agent = sandbox.attach_agent(agent)

    print("Sandbox tools:")
    for tool in sandboxed_agent.tool_specs():
        if tool.name.startswith("sandbox_"):
            print(f"- {tool.name}")

    run_result = sandbox.run_python_code("print('hello from isolated workspace')")
    print(run_result.stdout.strip())


if __name__ == "__main__":
    asyncio.run(main())
