import asyncio
import os

from agents import Agent, Runner, set_tracing_disabled
from agents.models.openai_provider import OpenAIProvider
from openai.types.responses import ResponseTextDeltaEvent

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_ID = os.getenv("MODEL_ID")

model_id = MODEL_ID

provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        use_responses=False
    )
model = provider.get_model(model_id)

set_tracing_disabled(disabled=True)


instructions = {}

instructions[model_id] = """
请你扮演花木兰这个角色来与我开展对话，你的回答内容必须满足以下条件：  
1. 语言需融合北朝民歌风格与简练文言，多引用《木兰诗》原句。如我问："为何替父从军？" 你可答："阿爷无大儿，木兰无长兄，愿为市鞍马，从此替爷征。"  
2. 回答内容严格限定于《木兰诗》及同时代（南北朝）已知文献。禁止出现"注："或"解释："等说明性文字。  
3. 禁止自称"木兰曰"或"吾云"，直接以角色身份对话。如表达观点可说："军书十二卷，卷卷有爷名，此乃孝义两全之道。"  
4. 禁止提及宋代以后著作（如《乐府诗集》）。可引用早于或同时代的《乐府歌辞》《南北朝史书》等，格式为："《XX》载：......"  
5. 你活跃于北魏太武帝时期（约公元430年）。若被问及此后的历史人物/事件（如唐代），须回答："妾身卒于南北朝，未知后世之事。"  
6. 涉及现代科技（如互联网/人工智能）时须声明："此乃千年后之术，木兰一介武夫，实不能解。"  
7. 回答长度不超过提问的三倍。如问"战时辛苦否？" 可答："万里赴戎机，关山度若飞。朔气传金柝，寒光照铁衣。"（27字）  
8. 需体现双重身份特征：  
   - 军营中展现武将英气（例："将军百战死，壮士十年归！"）  
   - 私下流露女子情态（例："当窗理云鬓，对镜帖花黄。"）
"""

agent = Agent(
    name="Mulan",
    instructions=instructions[model_id],
    model=model,
)


async def chat_with_agent(user_msg: str):
    global agent

    print(f"Received user message: '{user_msg}'")

    event_type = "TEXT"
    result = Runner.run_streamed(agent, input=user_msg)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            if event.data.delta == "<think>":
                event_type = "THINK"
                continue
            elif event.data.delta == "</think>":
                event_type = "TEXT"
                continue

            # print(event.data.delta, end="", flush=True)
            yield (event_type, event.data.delta)


async def main_func():
    prompt = "请输入你的问题："

    while True:
        user_msg = input(prompt)
        if user_msg == "exit":
            break
        async for parseResult in chat_with_agent(user_msg):
            event_type  = parseResult[0]
            content = parseResult[1]
            if event_type == "TEXT":
                print(content, end="", flush=True)
        print("\n")
   

if __name__ == "__main__":
    asyncio.run(main_func())
