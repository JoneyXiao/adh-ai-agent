#!/usr/bin/env python3
"""
PDF文档预处理器

用于下载PDF文档、提取文本、分割成块并保存预处理结果。
支持缓存机制，避免重复下载。
"""

import argparse
import asyncio
import hashlib
import logging
import nltk
import os
import pickle
import re
import requests
import tiktoken
import urllib.parse
from dataclasses import dataclass
from io import BytesIO
from nltk.tokenize import sent_tokenize
from pathlib import Path
from pypdf import PdfReader
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed. Environment variables will be loaded from system environment only.")

# 配置日志
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量
TOKENIZER_NAME = "o200k_base"
DATA_DIR = Path('data')
CHUNKS_FILE = DATA_DIR / 'document_chunks.pkl'
PDF_CACHE_DIR = DATA_DIR / 'pdf_cache'
MAX_PAGES = 920  # 最大处理页数
MIN_TOKENS_PER_CHUNK = 500  # 每块最小token数
DEFAULT_TARGET_CHUNKS = 20  # 默认目标块数

# HTTP请求头
DOCUMENT_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
}

@dataclass
class DocumentChunk:
    """文档块数据结构"""
    chunk_id: int
    text: str
    display_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        return {
            "id": self.chunk_id,
            "text": self.text,
            "display_id": self.display_id
        }

class PDFProcessor:
    """PDF文档处理器"""
    
    def __init__(self, tokenizer_name: str = TOKENIZER_NAME):
        self.tokenizer_name = tokenizer_name
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """确保必要的目录存在"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_filename(self, url: str) -> Path:
        """根据URL生成缓存文件名"""
        # 创建URL的哈希值作为文件名
        url_hash = hashlib.md5(url.encode()).hexdigest()
        
        # 尝试从URL中提取有意义的文件名
        parsed_url = urllib.parse.urlparse(url)
        original_filename = Path(parsed_url.path).name
        
        # 如果无法获取文件名，使用默认名称
        if not original_filename or not original_filename.endswith('.pdf'):
            original_filename = "document.pdf"
        
        # 组合哈希值和原始文件名，确保唯一性和可读性
        cache_filename = f"{url_hash}_{original_filename}"
        return PDF_CACHE_DIR / cache_filename
    
    def _download_pdf(self, url: str, cache_path: Path) -> None:
        """下载PDF并保存到缓存"""
        logger.info(f"正在从 {url} 下载PDF...")
        
        try:
            response = requests.get(url, headers=DOCUMENT_REQUEST_HEADERS, timeout=30)
            response.raise_for_status()
            
            # 保存到缓存
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"PDF已缓存到 {cache_path} ({file_size_mb:.1f} MB)")
            
        except requests.RequestException as e:
            logger.error(f"下载PDF失败: {e}")
            raise
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """从PDF文件中提取文本"""
        logger.info(f"正在处理PDF文件: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                
                full_text = ""
                pages_processed = min(len(pdf_reader.pages), MAX_PAGES)
                
                for i in range(pages_processed):
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():  # 只添加非空页面
                        full_text += page_text + "\n"
                
                # 计算统计信息
                word_count = len(re.findall(r'\b\w+\b', full_text))
                token_count = len(self.tokenizer.encode(full_text))
                
                logger.info(
                    f"文档处理完成: {len(pdf_reader.pages)} 页, "
                    f"处理了 {pages_processed} 页, "
                    f"{word_count} 词, {token_count} tokens"
                )
                
                return full_text
                
        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            raise
    
    def load_document(self, url: str, force_download: bool = False) -> str:
        """从URL加载文档，支持缓存"""
        cache_path = self._get_cache_filename(url)
        
        # 检查缓存是否存在且不强制下载
        if cache_path.exists() and not force_download:
            logger.info(f"使用缓存文件: {cache_path}")
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"使用缓存的PDF ({cache_size_mb:.1f} MB) - 跳过下载")
            return self._extract_text_from_pdf(cache_path)
        
        # 下载并缓存PDF
        try:
            self._download_pdf(url, cache_path)
            return self._extract_text_from_pdf(cache_path)
            
        except Exception as e:
            logger.error(f"文档处理失败: {e}")
            raise
    
    def split_into_chunks(
        self, 
        text: str, 
        target_chunks: int = DEFAULT_TARGET_CHUNKS,
        min_tokens: int = MIN_TOKENS_PER_CHUNK,
        max_tokens: int = 2000
    ) -> List[DocumentChunk]:
        """将文本分割成优化大小的块，保持句子边界"""
        
        # 首先分割成句子
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            # 检查是否应该完成当前块
            would_exceed_max = current_chunk_tokens + sentence_tokens > max_tokens
            meets_minimum = current_chunk_tokens >= min_tokens
            
            if would_exceed_max and meets_minimum:
                # 完成当前块
                chunk_text = " ".join(current_chunk_sentences)
                chunks.append(DocumentChunk(
                    chunk_id=len(chunks),
                    text=chunk_text
                ))
                current_chunk_sentences = [sentence]
                current_chunk_tokens = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
        
        # 添加最后一块
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(DocumentChunk(
                chunk_id=len(chunks),
                text=chunk_text
            ))
        
        # 如果块数太多，进行合并
        if len(chunks) > target_chunks * 2:
            chunks = self._consolidate_chunks(chunks, target_chunks)
        
        self._log_chunk_statistics(chunks)
        return chunks
    
    def _consolidate_chunks(
        self, 
        chunks: List[DocumentChunk], 
        target_count: int
    ) -> List[DocumentChunk]:
        """合并块到目标数量"""
        logger.info(f"合并 {len(chunks)} 个块到约 {target_count} 个块")
        
        all_text = " ".join(chunk.text for chunk in chunks)
        sentences = sent_tokenize(all_text)
        sentences_per_chunk = max(1, len(sentences) // target_count)
        
        consolidated_chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences)
            consolidated_chunks.append(DocumentChunk(
                chunk_id=len(consolidated_chunks),
                text=chunk_text
            ))
        
        return consolidated_chunks
    
    def _log_chunk_statistics(self, chunks: List[DocumentChunk]) -> None:
        """记录块统计信息"""
        logger.info(f"文档分割为 {len(chunks)} 个块")
        
        total_tokens = 0
        min_tokens = float('inf')
        max_tokens = 0
        
        for chunk in chunks:
            token_count = len(self.tokenizer.encode(chunk.text))
            total_tokens += token_count
            min_tokens = min(min_tokens, token_count)
            max_tokens = max(max_tokens, token_count)
            logger.debug(f"块 {chunk.chunk_id}: {token_count} tokens")
        
        avg_tokens = total_tokens // len(chunks) if chunks else 0
        logger.info(f"总tokens: {total_tokens}")
        logger.info(f"平均每块tokens: {avg_tokens} (最小: {min_tokens}, 最大: {max_tokens})")

class DocumentPreprocessor:
    """文档预处理器主类"""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
    
    def preprocess_document(
        self, 
        document_url: str, 
        force_download: bool = False,
        target_chunks: int = DEFAULT_TARGET_CHUNKS,
        min_tokens: int = MIN_TOKENS_PER_CHUNK
    ) -> None:
        """预处理文档并保存块"""
        
        logger.info("=" * 60)
        logger.info("开始文档预处理")
        logger.info(f"文档URL: {document_url}")
        logger.info(f"目标块数: {target_chunks}")
        logger.info(f"强制下载: {force_download}")
        logger.info("=" * 60)
        
        try:
            # 设置NLTK数据
            self._setup_nltk()
            
            # 加载文档
            document_text = self.pdf_processor.load_document(
                document_url, 
                force_download=force_download
            )
            
            # 分割成块
            document_chunks = self.pdf_processor.split_into_chunks(
                document_text,
                target_chunks=target_chunks,
                min_tokens=min_tokens
            )
            
            # 转换为字典格式以保持向后兼容性
            chunks_data = [chunk.to_dict() for chunk in document_chunks]
            
            # 保存预处理结果
            with open(CHUNKS_FILE, 'wb') as f:
                pickle.dump(chunks_data, f)
            
            logger.info("=" * 60)
            logger.info(f"预处理完成！结果已保存到: {CHUNKS_FILE}")
            logger.info(f"生成了 {len(chunks_data)} 个文档块")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"预处理过程中出现错误: {e}")
            raise
    
    def _setup_nltk(self) -> None:
        """设置NLTK数据"""
        try:
            logger.info("下载NLTK数据...")
            nltk.download('punkt_tab', quiet=True)
            logger.info("NLTK数据下载完成")
        except Exception as e:
            logger.error(f"NLTK数据下载失败: {e}")
            raise
    
    def get_cache_info(self) -> None:
        """显示缓存信息"""
        if not PDF_CACHE_DIR.exists():
            print("未找到PDF缓存目录。")
            return
        
        cached_files = list(PDF_CACHE_DIR.glob("*.pdf"))
        if not cached_files:
            print("未找到缓存的PDF文件。")
            return
        
        print(f"找到 {len(cached_files)} 个缓存的PDF文件:")
        total_size = 0
        for file_path in cached_files:
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            total_size += file_size
            print(f"  - {file_path.name} ({file_size_mb:.1f} MB)")
        
        total_size_mb = total_size / (1024 * 1024)
        print(f"总缓存大小: {total_size_mb:.1f} MB")
    
    def clear_cache(self) -> None:
        """清理缓存"""
        if not PDF_CACHE_DIR.exists():
            print("未找到PDF缓存目录。")
            return
        
        cached_files = list(PDF_CACHE_DIR.glob("*.pdf"))
        if not cached_files:
            print("没有需要清理的缓存文件。")
            return
        
        for file_path in cached_files:
            file_path.unlink()
        
        print(f"已清理 {len(cached_files)} 个缓存文件。")

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="PDF文档预处理器")
    
    parser.add_argument(
        'document_url', 
        nargs='?',
        help='要处理的PDF文档URL'
    )
    
    parser.add_argument(
        '--force-download', 
        action='store_true',
        help='强制下载，即使缓存文件存在'
    )
    
    parser.add_argument(
        '--target-chunks', 
        type=int, 
        default=DEFAULT_TARGET_CHUNKS,
        help=f'目标块数 (默认: {DEFAULT_TARGET_CHUNKS})'
    )
    
    parser.add_argument(
        '--min-tokens', 
        type=int, 
        default=MIN_TOKENS_PER_CHUNK,
        help=f'每块最小token数 (默认: {MIN_TOKENS_PER_CHUNK})'
    )
    
    parser.add_argument(
        '--cache-info', 
        action='store_true',
        help='显示缓存信息'
    )
    
    parser.add_argument(
        '--clear-cache', 
        action='store_true',
        help='清理所有缓存文件'
    )
    
    args = parser.parse_args()
    
    preprocessor = DocumentPreprocessor()
    
    # 处理特殊命令
    if args.cache_info:
        preprocessor.get_cache_info()
        return
    
    if args.clear_cache:
        preprocessor.clear_cache()
        return
    
    # 处理文档URL
    if not args.document_url:
        # 使用默认URL或从环境变量获取
        default_url = os.getenv(
            "DOCUMENT_URL", 
            "https://www.sc.gov.cn/10462/zfwjts/2023/7/10/5612ee444646428b9ff14260bfc0142f/files/d1be4d3234c849e6b537fc880ede2151.pdf"
        )
        print(f"使用默认文档URL: {default_url}")
        args.document_url = default_url
    
    try:
        preprocessor.preprocess_document(
            document_url=args.document_url,
            force_download=args.force_download,
            target_chunks=args.target_chunks,
            min_tokens=args.min_tokens
        )
        print("\n✅ 预处理成功完成！")
        print("现在可以使用 RAG 系统进行问答了。")
        
    except KeyboardInterrupt:
        print("\n❌ 用户中断操作")
    except Exception as e:
        print(f"\n❌ 预处理失败: {e}")
        logger.error(f"预处理失败: {e}", exc_info=True)

if __name__ == "__main__":
    main()
