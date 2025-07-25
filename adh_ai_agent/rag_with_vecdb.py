import argparse
import asyncio
import numpy as np
import re
import os
import pickle
import tiktoken
import time

from bs4 import BeautifulSoup
from faiss import IndexFlatL2
from multiprocessing import Process, Queue
from openai import OpenAI
from patchright.sync_api import sync_playwright

from agents import (
    Agent,
    ModelSettings,
    Runner,
    RunResult,
    function_tool,
    set_tracing_disabled,
)
from agents.models.openai_provider import OpenAIProvider
from dotenv import load_dotenv

load_dotenv()


set_tracing_disabled(disabled=True)

chunks_file = 'data/index_and_chunks.pkl'

model_id = os.getenv("MODEL_ID", "qwen3:8b")

embedding_model_id = os.getenv("EMBEDDING_MODEL_ID", "bge-large")


API_KEY = os.getenv("API_KEY", "ollama")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
if model_id.startswith("qwen3:"):
    API_KEY = "ollama"
    BASE_URL = "http://127.0.0.1:11434/v1"
    
if model_id == "gpt-4-turbo":
    provider = OpenAIProvider(
        api_key=API_KEY,
        use_responses=False
        )
else:
    provider = OpenAIProvider(
        api_key=API_KEY,
        base_url=BASE_URL,
        use_responses=False
        )
model = provider.get_model(model_id)

if model_id == "gpt-4-turbo":
    client = OpenAI(api_key=API_KEY)
else:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

extra_body = {"enable_thinking": False}
if model_id == "gpt-4-turbo" or model_id.startswith("qwen3:"):
    extra_body = {}


def get_page_content(url: str, q: Queue):
    with sync_playwright() as playwright:
        chromium = playwright.chromium
        browser = chromium.launch(
            headless=True,
            args=['--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36']
        )
        page = browser.new_page()
        page.goto(url, timeout=30000)
        page_content = page.content()
        browser.close()
        q.put(page_content)
    

@function_tool
def scrape_page(url: str) -> str:
    """Scrape all visible text from a webpage URL.

    Keyword arguments:
      url: The webpage to scrape
    """

    try:
        q = Queue()
        p = Process(target=get_page_content, args=(url,q))
        p.start()
        # p.join()
        page_content = q.get()
        
        soup = BeautifulSoup(page_content, "html.parser")

        # Remove scripts and styles
        for tag in soup(['script', 'style']):
            tag.decompose()

        text = soup.get_text(separator='\n')
        result = text.strip()
        return result
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


def create_agent1():
    global model, extra_body
    
    system_message = "You're a data extraction and research assistant."
    if model_id.startswith("qwen3:"):
        system_message = f"{system_message} /no_think"
    
    my_agent = Agent(
        name="AgenticRAG",
        instructions=system_message,
        model=model,
        model_settings=ModelSettings(extra_body=extra_body),
        tools=[scrape_page],
        tool_use_behavior="stop_on_first_tool"
    )
    return my_agent


def create_agent2():
    global model, extra_body
    
    system_message = "You are a little private secretary. Please answer my question based on the context information I give."
    if model_id.startswith("qwen3:"):
        system_message = f"{system_message} /no_think"
    
    my_agent = Agent(
        name="PrivateSecretary",
        instructions=system_message,
        model=model,
        model_settings=ModelSettings(extra_body=extra_body),
    )
    return my_agent
    

def get_chunks(text, chunk_size=300, overlap=30):
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = enc.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(enc.decode(chunk))
    return chunks


async def scrape_webpage_info(url: str):
    user_message = f"Scrape and extract all information from {url}"
    if model_id.startswith("qwen3:"):
        user_message = f"{user_message} /no_think"

    agent1 = create_agent1()
    result: RunResult = await Runner.run(
        agent1, 
        user_message,
    )

    scraped_text = result.final_output
    return scraped_text


def embedding_and_store(scraped_text):
    global client, embedding_model_id

    chunks = get_chunks(scraped_text)
    embeddings = [
        client.embeddings.create(model=embedding_model_id, input=[chunk]).data[0].embedding
        for chunk in chunks
    ]
    
    index = IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))  # type: ignore
    return index, chunks


async def query_rag(index: IndexFlatL2, chunks: list, question):
    global client, embedding_model_id
    
    query_embedding = client.embeddings.create(
        model=embedding_model_id, input=[question]).data[0].embedding
    D, I = index.search(np.array([query_embedding]).astype("float32"), k=3)  # type: ignore
    context = "\n\n".join([chunks[i] for i in I[0]])
    
    # print(context)

    user_message = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""
    if model_id.startswith("qwen3:"):
        user_message = f"{user_message} /no_think"

    agent2 = create_agent2()
    result: RunResult = await Runner.run(
        agent2, 
        user_message
    )

    answer = result.final_output
    if model_id.startswith("qwen3:"):
        answer = re.sub(r"<think>.*?</think>\n?", "", answer, flags=re.DOTALL)

    return answer.strip()


async def do_preprocess(url: str):
    scraped_text = await scrape_webpage_info(url)
    index, chunks = embedding_and_store(scraped_text)
    with open(chunks_file, 'wb') as f:
        pickle.dump((index, chunks), f)


async def do_answer_question(question: str):
    global chunks_file
    
    with open(chunks_file, 'rb') as f:
        index, chunks = pickle.load(f)

    # Answer user question by navigation result
    answer = await query_rag(index, chunks, question)
    return answer


async def main_func():
    url = "https://www.jiuzhai.com/intelligent-service/way-of-play"

    parser = argparse.ArgumentParser(description="rag with vecdb")
    parser.add_argument('--preprocess', required=False, action='store_true')
    args = parser.parse_args()
    preprocess = args.preprocess

    if preprocess:
        await do_preprocess(url)
        return

    prompt = "你的问题："
    question = input(prompt)

    answer = await do_answer_question(question)
    print(f"问题回答：{answer}")


if __name__ == "__main__":
    asyncio.run(main_func())
