"""
文档管理器 - 处理文档加载、增量更新和元数据管理
"""

import os
import json
import hashlib
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
from utils.model_utils import ensure_dir_exists

class DocumentManager:
    """文档管理器，处理文档的加载、增量更新和元数据管理"""
    
    def __init__(self, docs_dir="data/documents", metadata_dir="data/metadata"):
        """
        初始化文档管理器
        
        Args:
            docs_dir: 文档目录
            metadata_dir: 元数据目录
        """
        self.docs_dir = docs_dir
        self.metadata_dir = metadata_dir
        self.metadata_file = os.path.join(metadata_dir, "document_metadata.json")
        self.document_cache = {}
        self.metadata = self._load_metadata()
        
        # 确保目录存在
        ensure_dir_exists(docs_dir)
        ensure_dir_exists(metadata_dir)
        
    def _load_metadata(self) -> Dict[str, Any]:
        """加载文档元数据"""
        if not os.path.exists(self.metadata_file):
            return {"files": {}, "last_update": 0}
            
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"加载元数据失败: {str(e)}")
            return {"files": {}, "last_update": 0}
    
    def _save_metadata(self):
        """保存文档元数据"""
        self.metadata["last_update"] = time.time()
        
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存元数据失败: {str(e)}")
    
    def _calculate_hash(self, content: str) -> str:
        """计算文档内容的哈希值"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """获取文件信息"""
        stats = os.stat(file_path)
        return {
            "size": stats.st_size,
            "mtime": stats.st_mtime,
            "path": file_path
        }
    
    def load_documents(self, incremental=True) -> Tuple[List[str], List[str]]:
        """
        加载文档，支持增量更新
        
        Args:
            incremental: 是否增量加载（只加载新增或修改的文档）
            
        Returns:
            loaded_docs: 加载的文档内容
            doc_ids: 文档ID列表（文件名）
        """
        files = {}
        loaded_docs = []
        doc_ids = []
        
        # 扫描文档目录中的所有.txt文件
        for file in os.listdir(self.docs_dir):
            if not file.endswith('.txt'):
                continue
                
            file_path = os.path.join(self.docs_dir, file)
            file_info = self._get_file_info(file_path)
            files[file] = file_info
            
            # 检查文件是否需要处理（新增或修改）
            need_process = True
            if incremental and file in self.metadata["files"]:
                old_info = self.metadata["files"][file]
                # 如果文件大小和修改时间都没变，则跳过
                if old_info["size"] == file_info["size"] and old_info["mtime"] == file_info["mtime"]:
                    need_process = False
            
            if need_process:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 计算内容哈希值
                    content_hash = self._calculate_hash(content)
                    file_info["hash"] = content_hash
                    
                    # 缓存文档内容
                    self.document_cache[file] = content
                    
                    # 添加到加载列表
                    loaded_docs.append(content)
                    doc_ids.append(file)
                    
                    logging.info(f"加载文档: {file}")
                except Exception as e:
                    logging.error(f"加载文档 {file} 失败: {str(e)}")
            else:
                # 沿用原有哈希值
                file_info["hash"] = self.metadata["files"][file]["hash"]
                logging.debug(f"跳过未修改文档: {file}")
        
        # 更新元数据
        self.metadata["files"] = files
        self._save_metadata()
        
        logging.info(f"共加载 {len(loaded_docs)}/{len(files)} 个文档 (增量: {incremental})")
        return loaded_docs, doc_ids
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """
        获取指定ID的文档内容
        
        Args:
            doc_id: 文档ID（文件名）
            
        Returns:
            文档内容，如不存在则返回None
        """
        # 如果在缓存中，直接返回
        if doc_id in self.document_cache:
            return self.document_cache[doc_id]
            
        # 否则尝试从文件加载
        file_path = os.path.join(self.docs_dir, doc_id)
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 更新缓存
            self.document_cache[doc_id] = content
            return content
        except Exception as e:
            logging.error(f"获取文档 {doc_id} 失败: {str(e)}")
            return None
    
    def get_all_documents(self) -> List[Tuple[str, str]]:
        """
        获取所有文档的内容和ID
        
        Returns:
            文档内容和ID的元组列表 [(内容, ID), ...]
        """
        docs = []
        for doc_id in self.metadata["files"]:
            content = self.get_document(doc_id)
            if content:
                docs.append((content, doc_id))
        return docs
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        获取文档元数据
        
        Args:
            doc_id: 文档ID（文件名）
            
        Returns:
            文档元数据，如不存在则返回None
        """
        return self.metadata["files"].get(doc_id)
    
    def add_document(self, content: str, doc_id: str = None) -> str:
        """
        添加新文档
        
        Args:
            content: 文档内容
            doc_id: 文档ID（文件名），如不指定则自动生成
            
        Returns:
            文档ID
        """
        if doc_id is None:
            # 自动生成文件名
            timestamp = int(time.time())
            hash_suffix = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
            doc_id = f"doc_{timestamp}_{hash_suffix}.txt"
        
        # 确保有.txt后缀
        if not doc_id.endswith('.txt'):
            doc_id += '.txt'
            
        file_path = os.path.join(self.docs_dir, doc_id)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # 更新元数据
            file_info = self._get_file_info(file_path)
            file_info["hash"] = self._calculate_hash(content)
            self.metadata["files"][doc_id] = file_info
            self._save_metadata()
            
            # 更新缓存
            self.document_cache[doc_id] = content
            
            logging.info(f"添加文档: {doc_id}")
            return doc_id
        except Exception as e:
            logging.error(f"添加文档失败: {str(e)}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档ID（文件名）
            
        Returns:
            是否成功删除
        """
        file_path = os.path.join(self.docs_dir, doc_id)
        if not os.path.exists(file_path):
            logging.warning(f"删除文档失败: {doc_id} 不存在")
            return False
            
        try:
            os.remove(file_path)
            
            # 更新元数据
            if doc_id in self.metadata["files"]:
                del self.metadata["files"][doc_id]
            
            # 更新缓存
            if doc_id in self.document_cache:
                del self.document_cache[doc_id]
                
            self._save_metadata()
            logging.info(f"删除文档: {doc_id}")
            return True
        except Exception as e:
            logging.error(f"删除文档 {doc_id} 失败: {str(e)}")
            return False
    
    def clear_cache(self):
        """清除文档缓存"""
        self.document_cache.clear()
        logging.info("清除文档缓存") 