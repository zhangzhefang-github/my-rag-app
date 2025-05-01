"""
模型工具函数 - 处理模型路径、加载和保存
"""

import os
import logging
from sentence_transformers import SentenceTransformer
from utils.gpu_manager import GPUManager

def get_safe_model_name(model_name):
    """
    将模型名称转换为安全的文件名
    """
    return model_name.replace('/', '_')

def get_model_path(model_name, base_dir='models'):
    """
    获取模型的本地路径
    """
    return os.path.join(base_dir, get_safe_model_name(model_name))

def ensure_dir_exists(directory):
    """
    确保目录存在，如不存在则创建
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"创建目录: {directory}")
    return directory

''' # Start comment block
class ModelManager:
    """
    模型管理器 - 单例模式处理模型加载和缓存
    """
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.gpu_manager = GPUManager()
        
    def load_sentence_transformer(self, model_name, use_gpu=True, local_model_dir='models', force_reload=False):
        """
        加载句子转换器模型，优先使用本地缓存
        
        Args:
            model_name: 模型名称
            use_gpu: 是否使用GPU
            local_model_dir: 本地模型目录
            force_reload: 是否强制重新加载
            
        Returns:
            加载的模型
        """
        model_key = f"st_{model_name}_{use_gpu}"
        
        # 如果已加载且不要求重新加载，直接返回
        if model_key in self._models and not force_reload:
            logging.info(f"使用已加载的模型: {model_name}")
            return self._models[model_key]
            
        # 确保模型目录存在
        ensure_dir_exists(local_model_dir)
        
        # 确定设备
        device = self.gpu_manager.get_device(
            use_gpu=use_gpu, 
            min_memory_gb=1.0,
            task_name=f"加载模型 {model_name}"
        )
        
        # 创建本地模型路径
        local_model_path = get_model_path(model_name, local_model_dir)
        
        # 加载模型
        if os.path.exists(local_model_path):
            logging.info(f"从本地加载模型: {local_model_path}")
            model = SentenceTransformer(local_model_path, device=device)
        else:
            logging.info(f"本地模型不存在，从网络加载: {model_name}")
            model = SentenceTransformer(model_name, device=device)
            
            # 保存到本地，以便下次使用
            logging.info(f"保存模型到本地: {local_model_path}")
            model.save(local_model_path)
        
        # 缓存模型
        self._models[model_key] = model
        
        # 获取模型嵌入维度
        embedding_dim = model.get_sentence_embedding_dimension()
        logging.info(f"模型 {model_name} 嵌入向量维度: {embedding_dim}")
        
        # 检查并输出设备信息
        model_device = "GPU" if next(model.parameters()).is_cuda else "CPU"
        logging.info(f"模型 {model_name} 使用{model_device}进行计算")
        
        return model
        
    def clear_cache(self, model_key=None):
        """
        清除模型缓存
        
        Args:
            model_key: 特定模型的key，如为None则清除所有
        """
        if model_key:
            if model_key in self._models:
                del self._models[model_key]
                logging.info(f"清除模型缓存: {model_key}")
        else:
            self._models.clear()
            logging.info("清除所有模型缓存") 
''' # End comment block 