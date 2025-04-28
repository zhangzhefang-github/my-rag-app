"""
文档检索器 - 负责文档向量化和相似性检索
"""

import faiss
import numpy as np
import torch
import os
import json
import logging
from utils.gpu_manager import GPUManager
from utils.model_utils import ModelManager, ensure_dir_exists
from utils.document_manager import DocumentManager
from utils.logger import log_function_call

class Retriever:
    """向量检索器，负责文档向量化和相似度检索"""
    
    def __init__(self, model_name='moka-ai/m3e-base', use_gpu=True, 
                 local_model_dir='models', index_dir='data/indexes',
                 docs_dir='data/documents'):
        """
        初始化检索器
        
        Args:
            model_name: 使用的嵌入模型名称
            use_gpu: 是否使用GPU
            local_model_dir: 本地模型存储目录
            index_dir: 索引存储目录
            docs_dir: 文档目录
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.local_model_dir = local_model_dir
        self.index_dir = index_dir
        self.docs_dir = docs_dir
        
        # 获取管理器实例
        self.gpu_manager = GPUManager()
        self.model_manager = ModelManager()
        self.document_manager = DocumentManager(docs_dir=docs_dir)
        
        # 确保索引目录存在
        ensure_dir_exists(index_dir)
        
        # 生成安全的文件名
        self.model_name_safe = model_name.replace('/', '_')
        self.index_file = os.path.join(index_dir, f"{self.model_name_safe}.index")
        self.docs_file = os.path.join(index_dir, f"{self.model_name_safe}.docs.json")
        
        # 加载模型
        self.model = self.model_manager.load_sentence_transformer(
            model_name=model_name,
            use_gpu=use_gpu,
            local_model_dir=local_model_dir
        )
        
        # 获取嵌入维度
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # 加载或创建索引
        self._load_index()
        
    @log_function_call
    def _load_index(self):
        """加载或创建索引和文档映射"""
        self.docs = []
        self.doc_ids = []
        
        if os.path.exists(self.index_file) and os.path.exists(self.docs_file):
            try:
                logging.info(f"加载索引: {self.index_file}")
                self.index = faiss.read_index(self.index_file)
                
                logging.info(f"加载文档ID: {self.docs_file}")
                with open(self.docs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.doc_ids = data.get('doc_ids', [])
                    
                # 加载文档内容
                for doc_id in self.doc_ids:
                    content = self.document_manager.get_document(doc_id)
                    if content:
                        self.docs.append(content)
                    else:
                        logging.warning(f"无法加载文档: {doc_id}")
                
                if len(self.docs) != len(self.doc_ids):
                    logging.warning(f"文档数量不匹配: {len(self.docs)} != {len(self.doc_ids)}")
                    # 重置索引和文档，稍后会重建
                    self.docs = []
                    self.doc_ids = []
                    raise ValueError("文档数量不匹配")
                    
                logging.info(f"成功加载索引和文档，包含 {len(self.docs)} 个文档")
            except Exception as e:
                logging.error(f"加载索引或文档失败: {str(e)}")
                # 创建新索引
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.docs = []
                self.doc_ids = []
        else:
            # 创建新索引
            logging.info("创建新的FAISS索引")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # 如果有GPU，且显存足够，使用GPU索引
        self._move_index_to_gpu_if_available()
        
    @log_function_call 
    def _move_index_to_gpu_if_available(self):
        """如果可能的话，将索引移动到GPU"""
        self.on_gpu = False
        
        if not self.use_gpu:
            logging.info("按配置不使用GPU")
            return
            
        if not torch.cuda.is_available():
            logging.info("GPU不可用，使用CPU")
            return
            
        if not hasattr(faiss, 'StandardGpuResources'):
            logging.warning("FAISS不支持GPU，请安装faiss-gpu")
            return
            
        try:
            # 获取GPU信息
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # 显存足够才使用GPU
            if total_memory < 1.0:
                logging.info(f"GPU显存不足，使用CPU: {total_memory:.2f}GB < 1.0GB")
                return
                
            # 使用GPU资源
            self.gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, self.index)
            self.on_gpu = True
            logging.info("FAISS使用GPU进行向量检索")
        except Exception as e:
            logging.error(f"将索引移至GPU失败: {str(e)}")
            
    @log_function_call
    def save_index(self):
        """保存索引和文档ID到磁盘"""
        try:
            # 如果索引在GPU上，需要先将其移回CPU
            index_to_save = self.index
            if self.on_gpu:
                index_to_save = faiss.index_gpu_to_cpu(self.index)
                
            # 保存FAISS索引
            logging.info(f"保存索引到: {self.index_file}")
            faiss.write_index(index_to_save, self.index_file)
            
            # 保存文档ID
            logging.info(f"保存文档ID到: {self.docs_file}")
            with open(self.docs_file, 'w', encoding='utf-8') as f:
                json.dump({'doc_ids': self.doc_ids}, f, ensure_ascii=False, indent=2)
                
            logging.info(f"索引和文档ID保存成功，共 {len(self.docs)} 个文档")
            return True
        except Exception as e:
            logging.error(f"保存索引和文档ID失败: {str(e)}")
            return False
    
    @log_function_call
    def add_documents(self, documents=None, doc_ids=None):
        """
        添加文档到索引
        
        Args:
            documents: 文档内容列表，如不提供则从文档管理器加载
            doc_ids: 文档ID列表，需与documents长度一致
            
        Returns:
            添加的文档数量
        """
        if documents is None:
            # 从文档管理器加载
            documents, doc_ids = self.document_manager.load_documents(incremental=True)
            
        if not documents:
            logging.info("没有新文档需要添加")
            return 0
            
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
            
        if len(documents) != len(doc_ids):
            raise ValueError(f"文档数量与ID数量不匹配: {len(documents)} != {len(doc_ids)}")
            
        logging.info(f"为 {len(documents)} 个文档创建向量嵌入")
        embeddings = self.model.encode(documents)
        
        # 添加到索引
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.docs.extend(documents)
        self.doc_ids.extend(doc_ids)
        
        # 保存索引
        self.save_index()
        
        return len(documents)
    
    @log_function_call
    def retrieve(self, query, top_k=3):
        """
        检索与查询最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            最相关的文档列表
        """
        if not self.docs:
            logging.warning("检索时没有可用文档")
            return []
            
        query_embedding = self.model.encode([query])
        k = min(top_k, len(self.docs))
        
        # 执行检索
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        # 获取结果
        results = []
        result_ids = []
        for idx in indices[0]:
            if 0 <= idx < len(self.docs):
                results.append(self.docs[idx])
                result_ids.append(self.doc_ids[idx])
                
        logging.info(f"查询 '{query}' 找到 {len(results)} 个相关文档")
        return results
    
    @log_function_call
    def retrieve_with_metadata(self, query, top_k=3):
        """
        检索与查询最相关的文档，并返回文档ID和距离
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            (文档内容, 文档ID, 距离得分)元组的列表
        """
        if not self.docs:
            logging.warning("检索时没有可用文档")
            return []
            
        query_embedding = self.model.encode([query])
        k = min(top_k, len(self.docs))
        
        # 执行检索
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        # 获取结果
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.docs):
                results.append((
                    self.docs[idx],         # 文档内容
                    self.doc_ids[idx],      # 文档ID
                    float(distances[0][i])  # 距离得分
                ))
                
        logging.info(f"查询 '{query}' 找到 {len(results)} 个相关文档")
        return results

    @log_function_call
    def clear_index(self):
        """清空索引和文档"""
        # 重新创建索引
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # 如果在GPU上，需要将新索引移到GPU
        if self.on_gpu and hasattr(self, 'gpu_res'):
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, self.index)
            
        # 清空文档列表
        self.docs = []
        self.doc_ids = []
        
        # 删除索引文件和文档文件
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
            logging.info(f"已删除索引文件: {self.index_file}")
            
        if os.path.exists(self.docs_file):
            os.remove(self.docs_file)
            logging.info(f"已删除文档ID文件: {self.docs_file}")
            
        logging.info("索引和文档已清空")
        return True 