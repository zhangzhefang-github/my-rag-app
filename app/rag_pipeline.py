# RAG Pipeline implementation

# 移除导入 retrieve_docs
# from retrieval import retrieve_docs 
# 导入 Retriever (假设它位于 app.retriever)
from app.retriever import Retriever
# 使用重构后的 llm_service
# from app.llm_service import generate_answer # This seems outdated
# Import the LLM strategy interface and factory
from app.llm_strategies.base import LLMStrategy # Corrected import path
from app.llm_strategies import get_llm_strategy # Corrected factory import
# 移除 OpenAIGenerator 导入
# from app.openai_generator import OpenAIGenerator 
import logging
from typing import Union, AsyncGenerator, Dict, List, Any, Optional
# 移除 Generator 导入 (如果不再使用本地生成器)
# from app.generator import Generator
import time
import asyncio
import json
import re # Add re for parsing
import uuid # Added for chunk_id generation

# Import the Chunk and Citation models for type hinting
from app.models.document import Chunk, Citation, CitationSourceDetail

# 获取已配置的logger
logger = logging.getLogger(__name__)

# --- Define common greetings and self-identification queries --- 
COMMON_GREETINGS = ["你好", "您好", "hello", "hi", "hey", "喂"]
SELF_IDENTIFICATION_QUERIES = ["你是谁", "你是谁？", "你叫什么名字", "你叫什么名字？", "who are you", "who are you?", "what is your name", "what is your name?"]

COMMON_GREETINGS_NORMALIZED = {g.lower().strip("?？!") for g in COMMON_GREETINGS}
SELF_IDENTIFICATION_QUERIES_NORMALIZED = {q.lower().strip("?？!") for q in SELF_IDENTIFICATION_QUERIES}

class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline.
    Combines document retrieval with generative AI selected via environment configuration.
    """
    
    def __init__(self, retriever: Optional[Retriever] = None, llm: Optional[LLMStrategy] = None):
        """
        Initializes the RAG pipeline.
        
        Args:
            retriever: An instance of the retriever.
            llm: An instance of the LLM service.
        """
        self.retriever = retriever if retriever else Retriever() # Use default init if none provided
        self.llm = llm if llm else get_llm_strategy() # Use factory if none provided
        logger.info(f"RAG Pipeline initialized with Retriever: {type(self.retriever).__name__} and LLM: {type(self.llm).__name__}")
        
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
            # Note: Retriever.add_documents might need updates if its signature changed.
            # Assuming it still works or needs separate adjustments.
            # For now, focusing on retrieve call adaptation.
            result = self.retriever.add_documents(documents=documents, doc_ids=doc_ids)
            elapsed = time.time() - start_time
            logger.info(f"Knowledge addition completed - Time taken: {elapsed:.2f} seconds")
            # Ensure result reflects the success/failure based on retriever implementation
            return True if result > 0 else False # Assuming result is number added
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}", exc_info=True)
            return False
            
    def process_query(self, query: str, custom_top_k: Optional[int] = None, stream: bool = False, **kwargs) -> Dict[str, Any]:
        """Processes a query, retrieves context, and generates an answer (non-streaming)."""
        logger.info(f"Processing query: '{query[:50]}...', stream={stream}, custom_top_k={custom_top_k}")
        start_time = time.time()

        # 1. Retrieve relevant chunks (now returns List[Chunk])
        try:
            retrieved_chunks: List[Chunk] = self.retriever.retrieve(query, custom_top_k=custom_top_k)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")
            if not retrieved_chunks:
                 logger.warning(f"No relevant documents found for query: '{query}'")
                 # Return a specific message if no context is found
                 return {
                     "answer": "Sorry, I couldn't find any relevant information to answer your question.",
                     "sources": [], # Keep sources empty
                     "debug_info": {"retrieval_time": time.time() - start_time, "llm_time": 0}
                 }
        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            # Consider returning an error message or raising
            return {
                 "answer": "An error occurred during information retrieval.",
                 "sources": [],
                 "debug_info": {"retrieval_time": time.time() - start_time, "llm_time": 0}
             }

        retrieval_time = time.time() - start_time
        logger.debug(f"Retrieval took {retrieval_time:.4f} seconds.")

        # **SPRINT 1 TEMPORARY CHANGE**: Extract only text from chunks for LLM context
        context_texts: List[str] = [chunk.text for chunk in retrieved_chunks]
        logger.debug(f"Extracted {len(context_texts)} text contexts for LLM.")

        # 2. Generate answer using LLM with the extracted context texts
        llm_start_time = time.time()
        try:
            # Pass only the text list to the LLM service
            # The LLM service's answer_question method needs to accept List[str]
            answer = self.llm.generate(context_texts)
            llm_time = time.time() - llm_start_time
            logger.info(f"LLM generation completed in {llm_time:.4f} seconds.")

            # **SPRINT 1 TEMPORARY CHANGE**: Return raw chunks in a temporary 'sources' field
            # This is NOT the final citation structure.
            raw_sources = [chunk.model_dump(mode='json') if hasattr(chunk, 'model_dump') else chunk.dict() for chunk in retrieved_chunks]

            return {
                "answer": answer,
                "sources": raw_sources, # Temporary field with raw chunk data
                "debug_info": {"retrieval_time": retrieval_time, "llm_time": llm_time}
            }
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=True)
            llm_time = time.time() - llm_start_time
            return {
                 "answer": "An error occurred while generating the answer.",
                 "sources": [], # Return empty sources on generation error
                 "debug_info": {"retrieval_time": retrieval_time, "llm_time": llm_time}
            }

    # --- SPRINT 2: Helper to format context with indices --- 
    def _format_context_with_indices(self, chunks: List[Chunk]) -> str:
        """Formats list of chunks into a numbered string for the LLM prompt."""
        context_blocks = []
        for i, chunk in enumerate(chunks):
            # Truncate long chunk text if necessary for the prompt
            # truncated_text = chunk.text[:1500] + ("..." if len(chunk.text) > 1500 else "")
            # Use full text for now, maybe truncate later if needed
            context_blocks.append(f"[{i}] Source: {chunk.metadata.source}\n{chunk.text}")
        return "\n\n---\n\n".join(context_blocks)

    # --- SPRINT 2: Helper to parse LLM output for answer and citations --- 
    def _parse_llm_output_with_citations(self, raw_output: str) -> tuple[str, List[Dict]]:
        """Parses raw LLM output containing answer text and citation JSON.
        
        Expected format:
        Answer text potentially containing [index] markers...
        <<<CITATIONS>>>
        [{"text_quote": "...", "source_chunk_indices": [0, 1]}, ...]
        
        Returns:
            tuple: (answer_text, parsed_citation_data_list)
        """
        citations_marker = "<<<CITATIONS>>>"
        answer_text = raw_output
        parsed_citations = []

        marker_pos = raw_output.find(citations_marker)
        if marker_pos != -1:
            answer_text = raw_output[:marker_pos].strip()
            citation_json_str = raw_output[marker_pos + len(citations_marker):].strip()
            try:
                parsed_citations = json.loads(citation_json_str)
                if not isinstance(parsed_citations, list):
                    logger.warning(f"Parsed citation data is not a list: {parsed_citations}")
                    parsed_citations = [] # Fallback to empty list
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse citation JSON: {e}\nJSON string was: {citation_json_str}")
                parsed_citations = [] # Fallback to empty list
        else:
            logger.warning("Citation marker '<<<CITATIONS>>>' not found in LLM output.")
            # Keep the full raw_output as answer_text if marker is missing

        return answer_text, parsed_citations

    # --- SPRINT 2: Helper to build Citation objects --- 
    def _build_citation_objects(self, parsed_citations: List[Dict], retrieved_chunks: List[Chunk]) -> List[Citation]:
        """Builds Citation Pydantic objects from parsed data and original chunks."""
        citation_objects = []
        for parsed_cite in parsed_citations:
            text_quote = parsed_cite.get("text_quote")
            source_indices = parsed_cite.get("source_chunk_indices")

            if not text_quote or not isinstance(source_indices, list):
                logger.warning(f"Skipping invalid parsed citation data: {parsed_cite}")
                continue

            # Find the corresponding source chunks based on index
            source_chunks = []
            valid_indices = True
            for index in source_indices:
                if isinstance(index, int) and 0 <= index < len(retrieved_chunks):
                    source_chunks.append(retrieved_chunks[index])
                else:
                    logger.warning(f"Invalid source chunk index {index} found in citation: {parsed_cite}")
                    valid_indices = False
                    break 
            
            if not valid_indices or not source_chunks: # Skip if indices were bad or no chunks found
                continue

            # --- Revised Logic: Create ONE citation object per quote, linking to relevant source chunks --- 
            if source_chunks: # Only create if we have valid source chunks
                # Prepare the list of source details dictionaries/models
                details_list = [
                    CitationSourceDetail(
                        chunk_id=chunk.chunk_id,
                        doc_source_name=chunk.metadata.source,
                        chunk_text=chunk.text,
                        doc_source_id=chunk.doc_id
                        # Add other fields from chunk or metadata if defined in CitationSourceDetail
                    )
                    for chunk in source_chunks 
                ]
                
                # Create the Citation object using the new structure
                citation_obj = Citation(
                    text_quote=text_quote,
                    source_details=details_list
                )
                citation_objects.append(citation_obj)

        return citation_objects
        
    # --- SPRINT 2: Updated stream_answer_question --- 
    async def stream_answer_question(self, query: str, custom_top_k: Optional[int] = None, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Processes a query, retrieves context, streams the answer, and yields structured citations at the end.
           Includes special handling for common greetings and self-identification.
        """
        logger.info(f"Processing query for streaming: '{query[:50]}...', custom_top_k={custom_top_k}")
        start_time = time.time()

        # --- Special Query Handling (Greetings & Self-ID) --- 
        normalized_query = query.strip().lower().rstrip("?!.？") # Normalize and remove trailing punctuation

        bypass_rag = False
        direct_response_prompt = None

        if normalized_query in COMMON_GREETINGS_NORMALIZED:
            logger.info(f"Detected greeting: '{query}'. Bypassing RAG.")
            bypass_rag = True
            direct_response_prompt = f"你是一个友好的人工智能助手。请用自然、友好的方式简单回复用户的问候：'{query}'"
        elif normalized_query in SELF_IDENTIFICATION_QUERIES_NORMALIZED:
            logger.info(f"Detected self-identification query: '{query}'. Bypassing RAG.")
            bypass_rag = True
            # Customize the self-introduction as needed
            direct_response_prompt = "你是一个基于 RAG 架构的 AI 对话助手，名为智源对话。请向用户做个简单的自我介绍。"

        if bypass_rag:
            try:
                logger.info(f"Generating direct response using prompt: '{direct_response_prompt}'")
                messages = [("human", direct_response_prompt)]
                if hasattr(self.llm, 'astream') and callable(self.llm.astream):
                    async for chunk in self.llm.astream(messages):
                        yield {"type": "chunk", "data": chunk}
                    yield {"type": "citations", "data": []} 
                    logger.info("Direct response streamed successfully.")
                    return 
                else:
                    logger.warning("LLM strategy does not have 'astream' for direct response. Falling back to RAG.")
            except Exception as e:
                logger.error(f"Error generating direct response: {e}", exc_info=True)
                yield {"type": "error", "data": "生成直接回复时出错。"}
                yield {"type": "citations", "data": []}
                return
        # --- End Special Query Handling ---
        
        # --- Standard RAG Pipeline (if not handled above) ---
        logger.info("Query requires RAG pipeline.")
        
        # 1. Retrieve relevant chunks
        retrieved_chunks: List[Chunk] = []
        try:
            retrieved_chunks = self.retriever.retrieve(query, custom_top_k=custom_top_k)
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks for streaming.")
            if not retrieved_chunks:
                 logger.warning(f"No relevant documents found for query: '{query}'. Streaming empty answer.")
                 # Use the specific RAG-based "can't answer" message
                 yield {"type": "chunk", "data": "根据提供的上下文，我无法回答该问题。"} 
                 yield {"type": "citations", "data": []} 
                 return
        except Exception as e:
            logger.error(f"Error during retrieval for streaming: {e}", exc_info=True)
            yield {"type": "error", "data": "检索信息时出错。"}
            yield {"type": "citations", "data": []} 
            return

        retrieval_time = time.time() - start_time
        logger.debug(f"Streaming retrieval took {retrieval_time:.4f} seconds.")
        yield {"type": "debug", "data": {"retrieval_time": retrieval_time}}

        # 2. Format context with indices for the prompt
        context_str = self._format_context_with_indices(retrieved_chunks)
        logger.debug(f"Formatted context for LLM streaming: {context_str[:200]}...")

        # 4. Stream LLM response using stream_generate (which has the complex RAG prompt)
        llm_start_time = time.time()
        full_response = ""
        try:
            # Use stream_generate for the RAG-specific prompt and parsing
            llm_stream = self.llm.stream_generate(context_str=context_str, query=query) 

            if hasattr(llm_stream, '__aiter__'):
                async for token in llm_stream:
                    if token:
                        full_response += token
                        # Don't yield here yet, parse after full response
            elif hasattr(llm_stream, '__iter__'):
                 for token in llm_stream:
                    if token:
                        full_response += token
            else:
                logger.error("LLM stream_generate did not return a valid generator.")
                raise TypeError("LLM strategy stream_generate must return a generator or async generator.")

            llm_time = time.time() - llm_start_time
            logger.info(f"LLM generation completed in {llm_time:.4f} seconds. Full raw output length: {len(full_response)}")
            yield {"type": "debug", "data": {"llm_time": llm_time}} 

        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=True)
            yield {"type": "error", "data": f"生成答案时出错: {e}"}
            yield {"type": "citations", "data": []} 
            return

        # 5. Parse the full LLM output
        answer_text, parsed_citations_data = self._parse_llm_output_with_citations(full_response)
        logger.debug(f"Parsed Answer Text: {answer_text[:100]}...")
        logger.debug(f"Parsed Citations Data: {parsed_citations_data}")

        # 6. Yield the final answer text 
        yield {"type": "chunk", "data": answer_text}

        # 7. Build structured Citation objects
        citation_objects = self._build_citation_objects(parsed_citations_data, retrieved_chunks)
        logger.info(f"Built {len(citation_objects)} structured citation objects.")

        # 8. Yield the final structured citations
        serialized_citations = [c.model_dump() for c in citation_objects]
        yield {"type": "citations", "data": serialized_citations}