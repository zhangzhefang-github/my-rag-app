# RAG Pipeline implementation

# 移除导入 retrieve_docs
# from retrieval import retrieve_docs 
# 导入 Retriever (假设它位于 app.retriever)
from app.retriever import Retriever
from llm_service import generate_answer
# 导入 OpenAIGenerator
from app.openai_generator import OpenAIGenerator 
import logging
from typing import Union, AsyncGenerator

class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline.
    Combines document retrieval with generative AI.
    """
    
    # 修改 __init__ 签名，接收 retriever 实例
    def __init__(self, retriever: Retriever, low_memory_mode=False, use_openai=False, openai_config=None):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever (Retriever): Retriever instance for document retrieval and embedding.
            low_memory_mode (bool): Whether to operate in low memory mode
            use_openai (bool): Whether to use OpenAI for generation
            openai_config (dict): Configuration for OpenAI API
        """
        # 存储 retriever 实例
        self.retriever = retriever 
        # 移除 vector_store 和 retriever_config 的存储 (如果不再需要)
        # self.vector_store = vector_store 
        self.low_memory_mode = low_memory_mode
        self.use_openai = use_openai
        self.openai_config = openai_config
        self.llm_generator = None

        # 添加调试日志
        logging.debug(f"RAGPipeline init: use_openai={self.use_openai}, openai_config provided: {self.openai_config is not None}")
        if self.openai_config:
             logging.debug(f"RAGPipeline init: openai_config details: {self.openai_config}")

        if self.use_openai:
            if self.openai_config:
                try:
                    self.llm_generator = OpenAIGenerator(
                        model=self.openai_config.get('model'),
                        api_key=self.openai_config.get('api_key'),
                        base_url=self.openai_config.get('base_url')
                    )
                    logging.info("OpenAI Generator initialized successfully.") # 更明确的成功日志
                    # 添加日志确认 llm_generator 实例状态
                    logging.debug(f"RAGPipeline init: llm_generator initialized: {self.llm_generator is not None}")
                except Exception as e:
                    logging.error(f"Failed to initialize OpenAIGenerator: {e}")
                    self.llm_generator = None # 确保初始化失败时为 None
                    self.use_openai = False # 初始化失败则禁用 OpenAI
                    logging.warning("Disabling OpenAI due to generator initialization failure.")
            else:
                logging.warning("use_openai is True, but openai_config is missing. OpenAI Generator not initialized.")
                self.use_openai = False
        else:
             logging.info("Running RAG without OpenAI LLM generation.")

        # 确认最终的 use_openai 状态
        logging.debug(f"RAGPipeline init finished. Final use_openai state: {self.use_openai}")
        # 移除这条重复的日志
        # logging.info("RAG Pipeline initialized")
    
    # 添加 add_knowledge 方法，用于调用 retriever 添加文档
    def add_knowledge(self, documents: list[str], doc_ids: list[str]):
         """Adds documents to the retriever's index."""
         if not self.retriever:
             logging.error("Retriever not initialized, cannot add knowledge.")
             return
         try:
             added_count = self.retriever.add_documents(documents, doc_ids)
             logging.info(f"Added {added_count} documents to the retriever index.")
         except Exception as e:
             logging.error(f"Error adding documents to retriever: {str(e)}")

    # Make process_query async as it now calls async generate_answer
    async def process_query(self, query, top_k=3) -> dict:
        """
        Process a user query through the RAG pipeline (non-streaming response).
        
        Args:
            query (str): User's query
            top_k (int, optional): Number of top documents to retrieve. Defaults to 3.
            
        Returns:
            dict: Response containing the full answer, sources, and status.
        """
        try:
            logging.info(f"Processing query: '{query}' with top_k={top_k} (non-streaming)")
            
            if self.retriever is None:
                 logging.error("Retriever is not initialized.")
                 return {
                     "query": query,
                     "answer": "System error: Retriever not initialized.",
                     "sources": [],
                     "success": False,
                     "error": "Retriever not initialized"
                 }

            retrieved_doc_contents = self.retriever.retrieve(query, top_k=top_k)
            logging.info(f"Retrieved {len(retrieved_doc_contents)} document contents")

            logging.debug(f"process_query: About to call generate_answer. self.use_openai is {self.use_openai}, self.llm_generator is None is {self.llm_generator is None}")
            
            # Call generate_answer with stream=False
            answer = await generate_answer(
                query=query, 
                retrieved_doc_contents=retrieved_doc_contents, 
                use_openai=self.use_openai, 
                llm_generator=self.llm_generator,
                stream=False # Explicitly non-streaming
            )
            
            response = {
                "query": query,
                "answer": answer,
                "sources": retrieved_doc_contents, 
                "success": True,
                "error": None
            }
            
            logging.info(f"Query processed successfully (non-streaming)")
            return response
            
        except Exception as e:
            logging.error(f"Error processing non-streaming query: {e}", exc_info=True)
            return {
                "query": query,
                "answer": "Error processing query.",
                "sources": [],
                "success": False,
                "error": str(e)
            }
    
    # Keep original answer_question as an alias for non-streaming
    async def answer_question(self, query, top_k=3) -> dict:
        return await self.process_query(query, top_k=top_k)

    # Add a new method for streaming
    async def stream_answer_question(self, query: str, top_k: int = 3) -> AsyncGenerator[str, None]:
        """
        Process a user query and streams the answer back token by token.

        Args:
            query: The user query.
            top_k: Number of documents to retrieve.

        Yields:
            String chunks of the generated answer.
        """
        try:
            logging.info(f"Processing query: '{query}' with top_k={top_k} (streaming)")
            
            if self.retriever is None:
                logging.error("Retriever is not initialized for streaming.")
                yield "Error: Retriever not initialized."
                return

            # Step 1: Retrieve documents (sync for now, can be async if retriever supports it)
            retrieved_doc_contents = self.retriever.retrieve(query, top_k=top_k)
            logging.info(f"Retrieved {len(retrieved_doc_contents)} documents for streaming context.")
            # Optionally, yield sources info first if desired
            # yield json.dumps({"type": "sources", "data": retrieved_doc_contents}) + "\n\n"

            # Step 2: Generate answer using streaming
            logging.debug(f"stream_answer_question: Calling generate_answer (stream=True). use_openai={self.use_openai}")
            answer_stream = await generate_answer(
                query=query,
                retrieved_doc_contents=retrieved_doc_contents,
                use_openai=self.use_openai,
                llm_generator=self.llm_generator,
                stream=True
            )

            # Step 3: Yield chunks from the answer stream
            async for chunk in answer_stream:
                yield chunk
            
            logging.info(f"Finished streaming answer for query: '{query}'")

        except Exception as e:
            logging.error(f"Error during streaming query processing: {e}", exc_info=True)
            yield f"Error processing stream: {e}" # Yield error message in stream