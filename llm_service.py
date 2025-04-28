# LLM service module for the RAG application

import logging
# 导入 OpenAIGenerator 用于类型提示 (可选)
from typing import Optional
from app.openai_generator import OpenAIGenerator

def generate_answer(query, retrieved_doc_contents, use_openai=False, llm_generator: Optional[OpenAIGenerator]=None):
    """
    Generate an answer based on the query and retrieved documents, potentially using an LLM.
    
    Args:
        query (str): User's query
        retrieved_doc_contents (list[str]): List of retrieved document contents
        use_openai (bool): Flag indicating whether to use OpenAI.
        llm_generator (OpenAIGenerator, optional): Initialized OpenAI generator instance.
        
    Returns:
        str: Generated answer (either from LLM or a fallback).
    """
    
    # 添加调试日志
    logging.debug(f"generate_answer called with: use_openai={use_openai}, llm_generator is None: {llm_generator is None}")
    if llm_generator:
         logging.debug(f"LLM generator details: model={llm_generator.model}")

    # 构建上下文
    context = "\n---\n".join(retrieved_doc_contents)
    
    if use_openai and llm_generator:
        # 构建 Prompt
        prompt = f"Based on the following context:\n{context}\n\nAnswer the question: {query}"
        
        logging.info("Generating answer using OpenAI LLM...")
        try:
            # 调用 LLM 生成答案
            llm_answer = llm_generator.generate(prompt)
            logging.info("LLM generation successful.")
            return llm_answer
        except Exception as e:
            logging.error(f"Error during LLM generation: {e}")
            return "抱歉，调用语言模型生成答案时出错。"
    else:
        # 如果不使用 OpenAI 或生成器无效，返回基于上下文的简单回答或提示
        logging.warning("LLM generator not used. Returning placeholder answer.")
        # 可以选择返回之前的模板答案，或更明确的提示
        # return f"Based on the context below:\n{context}\n\nAnswer for '{query}': This is a sample answer (LLM not used)."
        if not context:
            return "抱歉，未能检索到相关信息来回答您的问题。"
        else:
             return f"根据检索到的信息：\n{context}\n\n(注意：未使用大型语言模型进行最终回答生成)"