# LLM service module for the RAG application

import logging
import asyncio
from typing import Optional, AsyncGenerator, Union
from app.openai_generator import OpenAIGenerator

# Make the function async
async def generate_answer(query, retrieved_doc_contents, use_openai=False, llm_generator: Optional[OpenAIGenerator]=None, stream=False) -> Union[str, AsyncGenerator[str, None]]:
    """
    Generate an answer based on the query and retrieved documents, potentially using an LLM.
    Supports both regular string response and streaming async generator response.
    
    Args:
        query (str): User's query
        retrieved_doc_contents (list[str]): List of retrieved document contents
        use_openai (bool): Flag indicating whether to use OpenAI.
        llm_generator (OpenAIGenerator, optional): Initialized OpenAI generator instance.
        stream (bool): If True, attempt to stream the response.
        
    Returns:
        If stream is False: The generated answer as a string.
        If stream is True: An async generator yielding answer chunks.
    """
    
    logging.debug(f"generate_answer called with: use_openai={use_openai}, llm_generator is None: {llm_generator is None}, stream={stream}")
    if llm_generator:
         logging.debug(f"LLM generator details: model={llm_generator.model}")

    context = "\n---\n".join(retrieved_doc_contents)
    
    if use_openai and llm_generator:
        prompt = f"Based on the following context:\n{context}\n\nAnswer the question: {query}"
        
        logging.info(f"Generating answer using OpenAI LLM... (Stream={stream})")
        try:
            # Call the async generate method, passing the stream flag
            response = await llm_generator.generate(prompt, stream=stream)
            
            if stream:
                # If streaming, return the async generator directly
                logging.info("Returning LLM stream.")
                return response 
            else:
                # If not streaming, return the complete string
                logging.info("LLM generation successful (non-streaming).")
                return response
                
        except Exception as e:
            logging.error(f"Error during LLM generation: {e}", exc_info=True)
            error_message = "抱歉，调用语言模型生成答案时出错。"
            if stream:
                async def error_stream():
                    yield error_message
                    await asyncio.sleep(0)
                return error_stream()
            else:
                return error_message
    else:
        # Fallback for non-OpenAI or missing generator
        logging.warning("LLM generator not used. Returning placeholder answer.")
        fallback_message = f"根据检索到的信息：\n{context}\n\n(注意：未使用大型语言模型进行最终回答生成)"
        if not context:
             fallback_message = "抱歉，未能检索到相关信息来回答您的问题。"

        if stream:
            async def fallback_stream():
                yield fallback_message
                await asyncio.sleep(0) # Ensure it's an async generator
            return fallback_stream()
        else:
            return fallback_message