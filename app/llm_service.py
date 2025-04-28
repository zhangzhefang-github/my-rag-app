import logging
import os
from typing import List, Union, AsyncGenerator
import traceback
from .retriever import DocumentRetriever
from .openai_generator import OpenAIGenerator

async def generate_answer(
    query: str, 
    retrieved_doc_contents: List[str] = None,
    use_openai: bool = True,
    stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]:
    """
    根据用户查询生成回答，支持标准和流式响应。
    
    Args:
        query: 用户查询
        retrieved_doc_contents: 检索到的文档内容列表
        use_openai: 是否使用OpenAI API
        stream: 是否使用流式响应

    Returns:
        根据stream参数返回字符串或异步生成器
    """
    logging.info(f"生成回答，查询: {query[:50]}...")
    logging.info(f"流式模式: {stream}, 使用OpenAI: {use_openai}")
    
    if not query:
        error_msg = "无效查询: 查询内容为空"
        logging.error(error_msg)
        if stream:
            async def error_generator():
                yield error_msg
            return error_generator()
        else:
            return error_msg
    
    # 检查是否有检索内容
    if not retrieved_doc_contents or not any(retrieved_doc_contents):
        logging.warning("没有提供检索文档内容，将使用纯LLM生成")
    else:
        logging.info(f"提供了 {len(retrieved_doc_contents)} 个检索文档")
        
    try:
        if use_openai:
            logging.info("使用OpenAI生成回答")
            
            # 构建提示词
            prompt = f"用户问题: {query}\n\n"
            
            if retrieved_doc_contents and any(retrieved_doc_contents):
                prompt += "以下是相关文档内容，请基于这些内容回答问题:\n\n"
                for i, doc_content in enumerate(retrieved_doc_contents):
                    prompt += f"文档 {i+1}:\n{doc_content}\n\n"
            else:
                prompt += "没有找到相关文档，请尽力回答问题。\n\n"
                
            prompt += "请以中文回答上述问题，保持礼貌专业。如果无法从提供的文档中找到答案，请明确说明。"
            
            logging.debug(f"完整提示词: {prompt}")
            
            # 初始化OpenAI生成器
            generator = OpenAIGenerator()
            
            # 生成回答
            if stream:
                logging.info("使用流式生成模式")
                # 不要使用await来获取异步生成器，直接返回它
                return generator.generate(prompt=prompt, stream=True)
            else:
                logging.info("使用标准生成模式")
                # 标准模式需要await
                return await generator.generate(prompt=prompt, stream=False)
        else:
            # 使用默认回复
            error_msg = "仅支持OpenAI生成，请设置use_openai=True"
            logging.warning(error_msg)
            if stream:
                async def error_generator():
                    yield error_msg
                return error_generator()
            else:
                return error_msg
                
    except Exception as e:
        error_msg = f"生成过程出错: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        
        if stream:
            async def error_generator():
                yield error_msg
            return error_generator()
        else:
            return error_msg 