import asyncio
import os
import re


from agents import (
    Agent,
    ModelSettings,
    Runner,
    RunResult,
    set_tracing_disabled,
)
from agents.mcp import MCPServerSse
from agents.models.openai_provider import OpenAIProvider
from dotenv import load_dotenv

load_dotenv()

set_tracing_disabled(disabled=True)

# model_id = "qwen3-235b-a22b"
model_id = os.getenv("MODEL_ID", "qwen3-8b")

extra_body = {}

API_KEY = os.getenv("API_KEY", "ollama")
if model_id == "qwen3-8b":
    extra_body = {"enable_thinking": False}
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
else:
    API_KEY = "ollama"
    BASE_URL = "http://127.0.0.1:11434/v1"

provider = OpenAIProvider(
    api_key=API_KEY,
    base_url=BASE_URL,
    use_responses=False
    )
model = provider.get_model(model_id)


async def call_openmanus_server(message: str) -> str:
    global model, extra_body
    
    remote_server = MCPServerSse(
        name="OpenManus Server",
        params={
            "url": "http://localhost:8005/sse",
        },
        cache_tools_list=False,
        client_session_timeout_seconds=300
    )

    # system_message = "Call OpenManus MCP Server for a response. The response should be plain text, don't use JSON or Markdown format"
    system_message = "调用 OpenManus Server 以回答与旅游、机票、酒店相关的问题。你的回答内容请使用简单的中文文本，不要使用 JSON 或者 Markdown 格式。"
    if model_id.startswith("qwen3"):
        system_message = f"{system_message} /no_think"

    async with remote_server as server:
        # Create an agent that uses the MCP server
        agent = Agent(
            name="Assistant",
            instructions=system_message,
            model=model,
            model_settings=ModelSettings(extra_body=extra_body),
            mcp_servers=[server],
            # tool_use_behavior="stop_on_first_tool"
        )

        # Run the agent
        result: RunResult = await Runner.run(agent, message)
        answer = result.final_output
        answer = re.sub(r"<think>.*?</think>\n?", "", answer, flags=re.DOTALL)
        return answer


async def main_func():
    user_message = "请在携程网查一下四川省最热门的旅游景点是哪个。回答请使用中文，简短一些。不要超过200字。"
    if model_id.startswith("qwen3"):
        user_message = f"{user_message} /no_think"

    weather = await call_openmanus_server(user_message)
    print(weather)


if __name__ == "__main__":
    asyncio.run(main_func())
