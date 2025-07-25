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
model_id = os.getenv("MODEL_ID")

extra_body = {}

API_KEY = os.getenv("API_KEY")
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


async def get_jzg_tomorrow_weather(message: str) -> str:
    global model, extra_body
    
    remote_server = MCPServerSse(
        name="Weather Server",
        params={
            "url": "http://localhost:8003/sse",
        },
        cache_tools_list=False,
        client_session_timeout_seconds=3600
    )

    system_message = "Get tomorrow's weather of a city. You should translate the city name from Chinese to English, for example: cityname \"九寨沟\" shoule be translate to \"Jiuzhaigou\". The response should be plain text, don't use JSON or Markdown format"
    if model_id and model_id.startswith("qwen3"):
        system_message = f"{system_message} /no_think"

    async with remote_server as server:
        # Create an agent that uses the MCP server
        agent = Agent(
            name="Assistant",
            instructions=system_message,
            model=model,
            model_settings=ModelSettings(extra_body=extra_body),
            mcp_servers=[server]
        )

        # Run the agent
        result: RunResult = await Runner.run(agent, message)
        answer = result.final_output
        answer = re.sub(r"<think>.*?</think>\n?", "", answer, flags=re.DOTALL)
        return answer


async def main_func():
    user_message = "九寨沟明天是什么天气？"
    if model_id and model_id.startswith("qwen3"):
        user_message = f"{user_message} /no_think"

    weather = await get_jzg_tomorrow_weather(user_message)
    print(weather)


if __name__ == "__main__":
    asyncio.run(main_func())
