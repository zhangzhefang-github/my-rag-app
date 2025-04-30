from transformers import pipeline
import torch
import asyncio
from typing import AsyncGenerator, Dict, Any, Union, List
import logging

# 获取已配置的logger
logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, model_name='gpt2', use_gpu=True):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称，默认为gpt2
            use_gpu: 是否使用GPU
        """
        # 自动检测是否有可用的GPU
        device = -1  # 默认使用CPU
        
        if use_gpu and torch.cuda.is_available():
            # 获取GPU显存信息
            free_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB单位
            logger.info(f"GPU总显存: {free_mem:.2f}GB")
            
            # 如果显存较小，使用更小的模型
            if free_mem < 3 and model_name == 'gpt2':
                logger.warning("检测到低显存GPU，建议使用distilgpt2模型")
            
            device = 0  # 使用GPU
        
        self.generator = pipeline('text-generation', model=model_name, device=device)
        self.model_name = model_name
        
        # 如果使用GPU，打印信息
        if device == 0:
            logger.info(f"Generator使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Generator使用CPU")

    def generate(self, question: str, context: str = None, system_prompt: str = None, max_length: int = 1000) -> str:
        """
        根据问题和上下文生成回答
        
        Args:
            question: 用户问题
            context: 检索到的文档上下文
            system_prompt: 系统提示
            max_length: 最大生成长度
            
        Returns:
            生成的回答
        """
        # 构建完整提示
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        if context:
            full_prompt += f"参考信息:\n{context}\n\n"
            
        full_prompt += f"问题: {question}\n\n回答:"
        
        # 生成回答
        logger.debug(f"生成提示: {full_prompt[:100]}...")
        
        try:
            result = self.generator(full_prompt, max_length=max_length, num_return_sequences=1)
            generated_text = result[0]['generated_text']
            
            # 简单移除提示部分，只保留回答
            answer = generated_text[len(full_prompt):].strip()
            if not answer:
                answer = "抱歉，我无法生成有效回答。请尝试重新表述您的问题。"
                
            return answer
        except Exception as e:
            logger.error(f"生成过程出错: {e}", exc_info=True)
            return f"生成过程中出现错误: {str(e)}"
    
    async def stream_generate(self, question: str, context: str = None, system_prompt: str = None) -> AsyncGenerator[str, None]:
        """
        流式生成回答
        
        Args:
            question: 用户问题
            context: 检索到的文档上下文
            system_prompt: 系统提示
            
        Yields:
            生成的文本片段
        """
        # 构建完整提示
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        if context:
            full_prompt += f"参考信息:\n{context}\n\n"
            
        full_prompt += f"问题: {question}\n\n回答:"
        
        logger.debug(f"流式生成提示: {full_prompt[:100]}...")
        
        # 由于本地模型不支持真正的流式生成，这里我们模拟流式效果
        try:
            # 生成完整回答
            result = self.generator(full_prompt, max_length=1000, num_return_sequences=1)
            generated_text = result[0]['generated_text']
            
            # 提取回答部分
            answer = generated_text[len(full_prompt):].strip()
            if not answer:
                answer = "抱歉，我无法生成有效回答。请尝试重新表述您的问题。"
            
            # 模拟流式输出，按字符或词汇分割
            tokens = []
            for word in answer.split(' '):
                tokens.append(word + ' ')
            
            # 每次yield一个token，并添加短暂延迟模拟流式效果
            for token in tokens:
                yield token
                await asyncio.sleep(0.05)  # 模拟打字效果的延迟
                
        except Exception as e:
            logger.error(f"流式生成过程出错: {e}", exc_info=True)
            yield f"生成过程中出现错误: {str(e)}" 