#!/usr/bin/env python3
"""
RAG without Vector Database

A question-answering system that uses AI agents to navigate and 
analyze documents without traditional vector databases.

For detailed configuration and usage instructions, see README.md
"""

import asyncio
import os
import pickle
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agents import (
    Agent,
    ModelSettings,
    Runner,
    set_tracing_disabled,
)
from agents.models.openai_provider import OpenAIProvider

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
set_tracing_disabled(disabled=True)

# Constants
DATA_DIR = Path('data')
CHUNKS_FILE = DATA_DIR / 'document_chunks.pkl'

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    chunk_id: int
    text: str
    display_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backwards compatibility."""
        return {
            "id": self.chunk_id,
            "text": self.text,
            "display_id": self.display_id
        }

class OutputType(TypedDict):
    """Type definition for agent output."""
    chunk_ids: List[int]

class CustomContext:
    """Custom context for agent communication."""
    def __init__(self, context_variables: Dict[str, Any]):
        self.context_variables = context_variables

class SimplifiedRAG:
    """Simplified RAG system that works reliably."""
    
    def __init__(self):
        self.model_id = self._get_model_id()
        self.provider = self._create_provider()
        self.model = self.provider.get_model(self.model_id)
        self.extra_body = self._get_extra_body()
    
    def _get_model_id(self) -> str:
        """Get model ID from environment or use default."""
        return os.getenv("MODEL_ID", "qwen3-8b")
    
    def _get_extra_body(self) -> Dict[str, Any]:
        """Get extra body configuration."""
        if self.model_id == "gpt-4.1-mini" or self.model_id.startswith("qwen3:"):
            return {}
        else:
            enable_thinking = os.getenv("ENABLE_THINKING", "false").lower() == "true"
            return {"enable_thinking": enable_thinking}
    
    def _create_provider(self) -> OpenAIProvider:
        """Create OpenAI provider based on model type."""
        if self.model_id.startswith("qwen3:"):
            api_key = "ollama"
            base_url = "http://127.0.0.1:11434/v1"
        elif self.model_id == "gpt-4.1-mini":
            api_key = os.getenv("OPENAI_API_KEY", os.getenv("API_KEY", ""))
            base_url = None
        else:
            api_key = os.getenv("DASHSCOPE_API_KEY", os.getenv("API_KEY", ""))
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        if not api_key:
            raise ValueError("API key not found. Please set the appropriate environment variable.")
        
        if base_url:
            return OpenAIProvider(
                api_key=api_key,
                base_url=base_url,
                use_responses=False
            )
        else:
            return OpenAIProvider(
                api_key=api_key,
                use_responses=False
            )
    
    def _get_system_message(self) -> str:
        """Get system message for navigation agents."""
        base_message = """You are an expert document navigator. Your task is to:
1. Identify which text chunks might contain information to answer the user's question
2. Choose chunks that are most likely relevant. Be selective, but thorough.

Analyze each chunk and select the ones that contain relevant information to answer the question."""
        
        if self.model_id.startswith("qwen3:"):
            return f"{base_message} /no_think"
        return base_message
    
    def _get_answering_system_message(self) -> str:
        """Get system message for answering agent."""
        base_message = """You are a helpful assistant. Please answer the question based on the context information provided in the <context>...</context> tags. If you can't answer the question based on the context information, you should say: "Sorry, I can't answer this question." in Chinese. The response should be plain text, don't use JSON or Markdown format."""
        
        if self.model_id.startswith("qwen3:"):
            return f"{base_message} /no_think"
        return base_message
    
    def _create_selector_agent(self) -> Agent[CustomContext]:
        """Create chunk selector agent."""
        return Agent[CustomContext](
            name="ChunkSelector",
            instructions=self._get_system_message(),
            output_type=OutputType,
            model=self.model,
            model_settings=ModelSettings(extra_body=self.extra_body),
        )
    
    def _create_answering_agent(self) -> Agent:
        """Create answering agent."""
        return Agent(
            name="AnsweringAgent",
            instructions=self._get_answering_system_message(),
            model=self.model,
            model_settings=ModelSettings(extra_body=self.extra_body),
        )
    
    async def _select_relevant_chunks(self, question: str, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Select relevant chunks using the AI agent."""
        # Build message for chunk selection
        user_message = f"QUESTION: {question}\n\nTEXT CHUNKS:\n\n"
        
        # Add chunks (limit to first 20 for performance)
        chunks_to_evaluate = chunks[:20]
        for chunk in chunks_to_evaluate:
            chunk_text = chunk.text[:2000] + "..." if len(chunk.text) > 2000 else chunk.text
            user_message += f"CHUNK {chunk.chunk_id}:\n{chunk_text}\n\n"
        
        user_message += "Please select the chunk IDs that are most relevant to answering the question. Return a JSON object with 'chunk_ids' containing an array of selected chunk IDs."
        
        if self.model_id.startswith("qwen3:"):
            user_message += " /no_think"
        
        try:
            agent = self._create_selector_agent()
            context = CustomContext({"question": question})
            
            result = await Runner.run(agent, user_message, context=context)
            
            if result.final_output and "chunk_ids" in result.final_output:
                selected_ids = result.final_output["chunk_ids"]
                selected_chunks = [chunk for chunk in chunks_to_evaluate if chunk.chunk_id in selected_ids]
                logger.info(f"Selected {len(selected_chunks)} relevant chunks")
                return selected_chunks
            
        except Exception as e:
            logger.warning(f"Chunk selection failed: {e}")
        
        # Fallback: return first few chunks
        logger.info("Using fallback chunk selection")
        return chunks[:3]
    
    async def _answer_with_context(self, question: str, relevant_chunks: List[DocumentChunk]) -> str:
        """Generate answer using relevant chunks as context."""
        context_content = "\n"
        
        for i, chunk in enumerate(relevant_chunks):
            display_id = chunk.display_id or str(chunk.chunk_id)
            para_text = f"PARAGRAPH {i+1} (ID: {display_id}):\n{chunk.text}"
            context_content += f"{para_text}\n\n"
        
        user_message = f"<context>{context_content}</context>\n\nquestion: {question}"
        
        if self.model_id.startswith("qwen3:"):
            user_message += " /no_think"
        
        try:
            agent = self._create_answering_agent()
            result = await Runner.run(agent, user_message)
            
            answer = result.final_output or "抱歉，无法回答您的问题。"
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "抱歉，处理问题时出现错误。"
    
    async def answer_question(self, question: str) -> str:
        """Main method to answer a question."""
        start_time = time.time()
        
        try:
            # Load document chunks
            if not CHUNKS_FILE.exists():
                return "错误：请先运行预处理步骤生成文档块。"
            
            with open(CHUNKS_FILE, 'rb') as f:
                chunks_data = pickle.load(f)
            
            # Convert to DocumentChunk objects if needed
            if chunks_data and isinstance(chunks_data[0], dict):
                document_chunks = [
                    DocumentChunk(
                        chunk_id=chunk["id"],
                        text=chunk["text"],
                        display_id=chunk.get("display_id")
                    )
                    for chunk in chunks_data
                ]
            else:
                document_chunks = chunks_data
            
            logger.info(f"Loaded {len(document_chunks)} document chunks")
            
            # Select relevant chunks
            relevant_chunks = await self._select_relevant_chunks(question, document_chunks)
            
            if not relevant_chunks:
                return "抱歉，数据源中未找到相关信息。"
            
            # Generate answer
            answer = await self._answer_with_context(question, relevant_chunks)
            
            processing_time = time.time() - start_time
            logger.info(f"Question answered in {processing_time:.2f} seconds")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "抱歉，处理问题时出现错误。"

# Global instance for backward compatibility
_rag_instance = None

async def do_answer_question(question: str) -> str:
    """Answer a question using the simplified RAG system."""
    global _rag_instance
    
    if _rag_instance is None:
        try:
            _rag_instance = SimplifiedRAG()
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return "配置错误：无法初始化RAG系统。请检查环境变量配置。"
    
    return await _rag_instance.answer_question(question)

if __name__ == "__main__":
    async def main():
        question = input("请输入你的问题：")
        answer = await do_answer_question(question)
        print(f"回答：{answer}")
    
    asyncio.run(main())
