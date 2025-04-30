# RAG Pipeline implementation

# 移除导入 retrieve_docs
# from retrieval import retrieve_docs 
# 导入 Retriever (假设它位于 app.retriever)
from app.retriever import Retriever
# 使用重构后的 llm_service
from app.llm_service import generate_answer 
# 移除 OpenAIGenerator 导入
# from app.openai_generator import OpenAIGenerator 
import logging
from typing import Union, AsyncGenerator, Dict, List, Any, Optional
# 移除 Generator 导入 (如果不再使用本地生成器)
# from app.generator import Generator
import time
import asyncio


# 获取已配置的logger
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline.
    Combines document retrieval with generative AI selected via environment configuration.
    """
    
    def __init__(self, retriever: Retriever):
        """
        Initializes the RAG pipeline.
        
        Args:
            retriever: An instance of the retriever.
        """
        if not isinstance(retriever, Retriever):
             raise TypeError("retriever must be an instance of Retriever")
        self.retriever = retriever
        # 不再需要管理 LLM 生成器实例，llm_service 会处理
        # self.generator = None # 移除
        logger.info(f"RAG Pipeline initialized with retriever: {type(retriever).__name__}")
        
    def add_knowledge(self, documents: List[str], doc_ids: List[str] = None) -> bool:
        """
        Adds knowledge documents to the retriever.
        
        Args:
            documents: List of document contents.
            doc_ids: Optional list of document IDs.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        start_time = time.time()
        logger.info(f"Starting to add knowledge - Document count: {len(documents)}")
        
        try:
            # 确保 retriever 实例存在
            if self.retriever is None:
                 logger.error("Cannot add knowledge: Retriever is not initialized.")
                 return False
            result = self.retriever.add_documents(documents, doc_ids)
            elapsed = time.time() - start_time
            logger.info(f"Knowledge addition completed - Time taken: {elapsed:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}", exc_info=True)
            return False
            
    async def process_query(self, query, top_k=3) -> dict:
        """
        Process a user query through the RAG pipeline (non-streaming response).
        
        Args:
            query (str): User's query.
            top_k (int, optional): Number of top documents to retrieve. Defaults to 3.
            
        Returns:
            dict: Response containing the full answer, sources, and status.
        """
        try:
            logger.info(f"Processing query: '{query}' with top_k={top_k} (non-streaming)")
            
            if self.retriever is None:
                 logger.error("Retriever is not initialized.")
                 # 返回更详细的错误信息
                 return {
                     "query": query,
                     "answer": "System error: Retriever component is missing.",
                     "sources": [],
                     "success": False,
                     "error": "Retriever not initialized"
                 }

            # 执行检索
            retrieval_start = time.time()
            retrieved_docs_content, _ = self.retriever.retrieve(query, top_k=top_k)
            retrieval_time = time.time() - retrieval_start
            logger.info(f"Retrieved {len(retrieved_docs_content)} document contents in {retrieval_time:.2f}s")

            # 调用重构后的 generate_answer
            generation_start = time.time()
            answer = await generate_answer(
                query=query, 
                retrieved_doc_contents=retrieved_docs_content,
                stream=False
            )
            generation_time = time.time() - generation_start
            logger.info(f"Generated non-streaming answer in {generation_time:.2f}s")
            
            # 准备响应
            response = {
                "query": query,
                "answer": answer,
                # 返回实际检索到的文档内容作为 sources
                "sources": retrieved_docs_content, 
                "success": True,
                "error": None
            }
            
            logger.info(f"Query processed successfully (non-streaming)")
            return response
            
        except Exception as e:
            logger.error(f"Error processing non-streaming query: {e}", exc_info=True)
            # 返回详细错误信息
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "success": False,
                "error": str(e)
            }
    
    async def answer_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Answers a question using retrieval and generation (non-streaming).
        This method now acts as a wrapper around process_query for consistency.
        
        Args:
            question: The user's question.
            top_k: The number of documents to retrieve.
            
        Returns:
            Dict: A dictionary containing the answer and related information.
        """
        logger.info(f"Answering question (non-streaming wrapper): '{question[:50]}...'")
        # 直接调用 process_query 处理
        return await self.process_query(query=question, top_k=top_k)

    async def stream_answer_question(self, question: str, top_k: int = 3) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Answers a question using retrieval and generation (streaming).
        
        Args:
            question: The user's question.
            top_k: The number of documents to retrieve.
            
        Yields:
            Intermediate processing steps (like 'retrieving', 'generating') 
            and then the streamed answer chunks (strings). Finally yields sources.
        """
        logger.info(f"Processing query (streaming): '{question[:50]}...', top_k={top_k}")
        start_time = time.time()

        try:
            if self.retriever is None:
                 logger.error("Retriever is not initialized for streaming.")
                 yield {"type": "error", "content": "System error: Retriever component is missing."}
                 return

            # 1. 检索步骤
            yield {"type": "status", "content": "检索相关文档..."}
            retrieval_start = time.time()
            retrieved_docs_content, _ = self.retriever.retrieve(question, top_k=top_k)
            retrieval_time = time.time() - retrieval_start
            logger.info(f"Streaming retrieval completed - Found {len(retrieved_docs_content)} docs in {retrieval_time:.2f}s")
            yield {"type": "status", "content": f"找到 {len(retrieved_docs_content)} 篇相关文档"}

            # 2. 生成步骤 (流式)
            yield {"type": "status", "content": "生成回答..."}
            generation_start = time.time()
            async for chunk_dict in generate_answer(
                query=question, 
                retrieved_doc_contents=retrieved_docs_content,
                stream=True 
            ):
                # 直接将 generate_answer 返回的字典 yield 出去
                # api.py 中的 message_stream_generator 已经知道如何处理这个字典
                yield chunk_dict 
                
            generation_time = time.time() - generation_start
            logger.info(f"Streaming generation completed in {generation_time:.2f}s")

            # 3. 返回源文档 (作为最后一步信息)
            yield {"type": "sources", "content": retrieved_docs_content}

            total_time = time.time() - start_time
            logger.info(f"Streaming query processed successfully - Total time: {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Error processing streaming query: {e}", exc_info=True)
            yield {"type": "error", "content": f"处理请求时发生错误: {str(e)}"}