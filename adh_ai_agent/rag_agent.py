from .rag_wo_vecdb import do_answer_question

async def chat_with_agent(user_msg: str):
    print(f"Received user message: '{user_msg}'")
    try:
        agent_reply = await do_answer_question(user_msg)
        yield ('TEXT', agent_reply)
        # return
    except Exception as e:
        print(f"ERROR: {e}")
