import logging
import os
from typing import List, Union, AsyncGenerator, Dict, Any
import traceback
# import re # No longer needed here
# 移除 DocumentRetriever，因为它在这里没有使用
# from .retriever import DocumentRetriever 
# 移除 OpenAIGenerator 导入
# from .openai_generator import OpenAIGenerator
from langchain_core.messages import SystemMessage, HumanMessage

# 导入新的策略工厂函数
from .llm_strategies import get_llm_strategy

# 获取 LLM 策略实例 (在模块加载时获取并缓存)
try:
    llm_strategy = get_llm_strategy()
except ValueError as e:
    logging.error(f"Failed to initialize LLM strategy: {e}")
    llm_strategy = None # 或者提供一个默认的错误处理策略

# Remove the helper function
# def remove_think_block(text: str) -> str:
#     ...

async def generate_answer(
    query: str,
    retrieved_doc_contents: List[str] = None,
    stream: bool = False
) -> AsyncGenerator[Dict[str, Any], None]: # Always returns AsyncGenerator
    """
    Generates an answer using the configured LLM strategy.
    Always returns an async generator. If stream=False, the generator 
    yields a single dictionary with type 'final_answer' or 'error'.
    If stream=True, yields multiple dictionaries with type 'chunk' or 'error'.
    """
    logging.info(f"Generating answer for query: {query[:50]}... Streaming: {stream}")

    # --- Get LLM Strategy --- 
    try:
        llm_strategy = get_llm_strategy()
    except ValueError as e:
        logging.error(f"Failed to initialize LLM strategy: {e}")
        yield {"type": "error", "content": f"LLM strategy initialization failed: {e}"}
        return

    if llm_strategy is None:
        yield {"type": "error", "content": "LLM strategy is not available."}
        return
            
    if not query:
        yield {"type": "error", "content": "Invalid query: Query content is empty."}
        return

    # --- Prepare messages --- 
    try:
        context = ""
        # Define a more explicit base instruction
        base_instruction = "\n请仔细阅读以下提供的文档，并严格根据文档内容回答问题。注意文档中的细节，例如时间、日期、数字和名称。如果需要，请进行简单的逻辑推理。请像一个乐于助人的朋友一样用中文回答。" 

        if retrieved_doc_contents and any(retrieved_doc_contents):
            logging.info(f"Using {len(retrieved_doc_contents)} retrieved documents.")
            # Modify the lead-in text for documents
            context += "请根据以下文档回答问题:\n\n" 
            for i, doc_content in enumerate(retrieved_doc_contents):
                context += f"文档 {i+1}:\n{doc_content}\n\n"
            # Append base instruction after the documents
            system_message_content = context + base_instruction 
        else:
            logging.warning("No relevant documents provided or found. Answering based on general knowledge.")
            # Simplify the no-document prompt but include the base instruction
            system_message_content = "请尽力回答以下问题。" + base_instruction

        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=query)
        ]
        logging.debug(f"Messages prepared for LLM: {messages}")

    except Exception as e:
        logging.error(f"Error preparing LLM messages: {e}", exc_info=True)
        yield {"type": "error", "content": f"Error preparing request: {e}"}
        return
        
    # --- Execute LLM Call (Streaming or Non-Streaming) --- 
    try:
        if stream:
            logging.info("Initiating streaming response.")
            async for chunk_content in llm_strategy.astream(messages):
                yield {"type": "chunk", "content": chunk_content}
            # Streaming ends implicitly
            
        else: # Non-streaming case
            logging.info("Generating standard (non-streaming) response.")
            raw_answer = await llm_strategy.ainvoke(messages) # Use ainvoke for consistency?
                                                         # Or keep invoke if strategies don't have ainvoke
            # Let's assume strategies have invoke (like current Ollama/CustomAPI)
            # If ainvoke is needed, OllamaStrategy/CustomAPIStrategy need ainvoke impl.
            raw_answer = llm_strategy.invoke(messages) 
            
            # Clean the response (if needed, or do it in API layer)
            # For simplicity, let's keep the cleaning logic here for now
            # --- Removing the call to remove_think_block --- 
            # answer = remove_think_block(raw_answer) 
            answer = raw_answer # Yield the raw answer now
            # --- End removal ---
            
            logging.info("Successfully generated non-streaming answer.")
            # Yield the single final answer
            yield {"type": "final_answer", "content": answer}

    except Exception as e:
        error_msg = f"Error during LLM call: {str(e)}"
        logging.error(error_msg, exc_info=True)
        yield {"type": "error", "content": error_msg}
        return # End generation after error 