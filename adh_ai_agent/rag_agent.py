from .rag_with_vecdb import do_answer_question
from .weather_mcp_host import get_jzg_tomorrow_weather

async def chat_with_agent(user_msg: str):
    print(f"Received user message: '{user_msg}'")
    try:
        if "天气" in user_msg:
            agent_reply = await get_jzg_tomorrow_weather(user_msg)
        else:
            agent_reply = await do_answer_question(user_msg)
        yield ('TEXT', agent_reply)
        # return
    except Exception as e:
        print(f"ERROR: {e}")
