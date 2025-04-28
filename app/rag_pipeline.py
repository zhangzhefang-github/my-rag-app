"""
RAG管道 - 整合检索和生成功能
"""

from app.retriever import Retriever
from app.generator import Generator
from app.openai_generator import OpenAIGenerator
from utils.gpu_manager import GPUManager
from utils.logger import log_function_call
import logging
import os

class RAGPipeline:
    def __init__(self, low_memory_mode=False, use_openai=True, openai_config=None, retriever_config=None):
        """
        初始化RAG管道
        
        Args:
            low_memory_mode (bool): 是否启用低内存模式
            use_openai (bool): 是否使用OpenAI API生成
            openai_config (dict): OpenAI配置，包含model, api_key, base_url
            retriever_config (dict): 检索器配置，包含model_name等
        """
        # 获取GPU管理器
        self.gpu_manager = GPUManager()
        
        # 检查低内存模式
        if self.gpu_manager.gpu_available:
            device_id = 0
            total_memory = self.gpu_manager.gpu_info['devices'][device_id]['total_memory_gb']
            logging.info(f"初始化RAG管道 - GPU总显存: {total_memory:.2f}GB")
            
            # 对于低显存设备，自动启用低内存模式
            if total_memory < 1.0 and not low_memory_mode:
                logging.info("检测到GPU显存小于1GB，自动启用低内存模式")
                low_memory_mode = True
        
        # 初始化检索器配置
        if retriever_config is None:
            retriever_config = {}
        
        # 从环境变量获取检索器模型名称（如果未在参数中指定）
        if "model_name" not in retriever_config:
            retriever_config["model_name"] = os.environ.get("RETRIEVER_MODEL", "moka-ai/m3e-base")
        
        # 设置本地模型目录
        if "local_model_dir" not in retriever_config:
            retriever_config["local_model_dir"] = os.environ.get("LOCAL_MODEL_DIR", "models")
            
        # 确保use_gpu配置存在
        if "use_gpu" not in retriever_config:
            retriever_config["use_gpu"] = True
        
        # 初始化检索器
        self.retriever = Retriever(
            model_name=retriever_config["model_name"],
            use_gpu=retriever_config["use_gpu"],
            local_model_dir=retriever_config["local_model_dir"]
        )
        
        # 初始化生成器
        if use_openai:
            if openai_config is None:
                openai_config = {
                    "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    "api_key": os.environ.get("OPENAI_API_KEY", None),
                    "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.fe8.cn/v1")
                }
            self.generator = OpenAIGenerator(**openai_config)
            logging.info(f"使用OpenAI API进行生成，模型: {openai_config['model']}")
        else:
            # 使用本地模型作为备用
            if low_memory_mode:
                logging.info("使用低内存模式：嵌入模型使用GPU，生成模型使用CPU")
                self.generator = Generator(model_name='distilgpt2', use_gpu=False)
            else:
                self.generator = Generator(use_gpu=True)

    @log_function_call
    def add_knowledge(self, documents, doc_ids=None):
        """
        添加知识到检索器
        
        Args:
            documents: 文档内容列表
            doc_ids: 文档ID列表，可选
        """
        self.retriever.add_documents(documents, doc_ids)

    @log_function_call
    def answer_question(self, query, top_k=3):
        """
        回答问题
        
        Args:
            query: 用户查询
            top_k: 检索结果数量
            
        Returns:
            生成的回答
        """
        logging.info(f"处理查询: {query}")
        contexts = self.retriever.retrieve(query, top_k=top_k)
        
        # 格式化增强查询，增加系统提示
        context_text = "\n".join(contexts)
        augmented_query = f"""基于以下背景信息回答问题。如果背景信息中没有相关内容，请诚实地说不知道。

背景信息：
{context_text}

问题：{query}"""
        
        logging.debug(f"增强查询: {augmented_query[:100]}...")
        answer = self.generator.generate(augmented_query)
        return answer 