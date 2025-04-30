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
    """Vector retriever responsible for document vectorization and similarity search"""
    
    def __init__(self, model_name: str, use_gpu: bool, 
                 local_model_dir: str, index_dir: str,
                 docs_dir: str):
        """
        Initialize the Retriever.
        
        Args:
            model_name: Name of the embedding model to use.
            use_gpu: Whether to use GPU if available.
            local_model_dir: Directory for storing/caching local models.
            index_dir: Directory to store Faiss index and document mapping files.
            docs_dir: Directory containing the original source documents.
        """
        logging.info(f"Initializing Retriever with config: model_name='{model_name}', use_gpu={use_gpu}, local_model_dir='{local_model_dir}', index_dir='{index_dir}', docs_dir='{docs_dir}'")
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.local_model_dir = local_model_dir
        self.index_dir = index_dir
        self.docs_dir = docs_dir
        
        # Get manager instances
        self.gpu_manager = GPUManager()
        self.model_manager = ModelManager()
        # Initialize DocumentManager specific to this retriever's docs_dir
        self.document_manager = DocumentManager(docs_dir=self.docs_dir)
        
        # Ensure index directory exists
        logging.debug(f"Ensuring index directory exists: {index_dir}")
        ensure_dir_exists(index_dir)
        
        # Generate safe filename for the model
        self.model_name_safe = model_name.replace('/', '_')
        self.index_file = os.path.join(index_dir, f"{self.model_name_safe}.index")
        self.docs_file = os.path.join(index_dir, f"{self.model_name_safe}.docs.json")
        logging.debug(f"Index file path: {self.index_file}")
        logging.debug(f"Docs mapping file path: {self.docs_file}")
        
        # Load the embedding model
        logging.info(f"Loading embedding model: '{model_name}'")
        self.model = self.model_manager.load_sentence_transformer(
            model_name=self.model_name,
            use_gpu=self.use_gpu,
            local_model_dir=self.local_model_dir
        )
        logging.info(f"Embedding model '{model_name}' loaded successfully.")
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logging.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Load or create the index
        logging.info("Loading or creating FAISS index...")
        self._load_index()
        logging.info("FAISS index ready.")
        
    @log_function_call
    def _load_index(self):
        """Load or create the index and document mapping."""
        self.docs = []
        self.doc_ids = []
        
        if os.path.exists(self.index_file) and os.path.exists(self.docs_file):
            try:
                logging.info(f"Attempting to load existing index from: {self.index_file}")
                self.index = faiss.read_index(self.index_file)
                logging.info(f"Successfully loaded index.")
                
                logging.info(f"Attempting to load document mapping from: {self.docs_file}")
                with open(self.docs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.doc_ids = data.get('doc_ids', [])
                logging.info(f"Successfully loaded {len(self.doc_ids)} document IDs.")
                    
                # Load document content using DocumentManager
                logging.info(f"Loading content for {len(self.doc_ids)} documents...")
                loaded_count = 0
                for doc_id in self.doc_ids:
                    content = self.document_manager.get_document(doc_id)
                    if content:
                        self.docs.append(content)
                        loaded_count += 1
                    else:
                        logging.warning(f"Could not load document content for ID: {doc_id}")
                logging.info(f"Successfully loaded content for {loaded_count}/{len(self.doc_ids)} documents.")
                
                if len(self.docs) != len(self.doc_ids):
                    logging.warning(f"Document count mismatch after loading: {len(self.docs)} content vs {len(self.doc_ids)} IDs. Index may need rebuild.")
                    # Reset and force creation of a new index
                    raise ValueError("Document count mismatch")
                    
                logging.info(f"Successfully loaded existing index and documents. Total documents: {len(self.docs)}")
            except Exception as e:
                logging.error(f"Failed to load existing index or documents: {e}. Creating a new index.")
                # Create new index
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.docs = []
                self.doc_ids = []
        else:
            logging.info(f"Index file '{self.index_file}' or docs file '{self.docs_file}' not found. Creating a new FAISS index.")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Move index to GPU if possible
        logging.debug("Attempting to move index to GPU if available...")
        self._move_index_to_gpu_if_available()
        
    @log_function_call 
    def _move_index_to_gpu_if_available(self):
        """Move the FAISS index to GPU if configured and possible."""
        self.on_gpu = False
        
        if not self.use_gpu:
            logging.info("GPU usage is disabled by configuration.")
            return
            
        if not torch.cuda.is_available():
            logging.info("CUDA (GPU) not available, using CPU for index.")
            return
            
        if not hasattr(faiss, 'StandardGpuResources'):
            logging.warning("FAISS GPU support not available. Please install faiss-gpu. Using CPU for index.")
            return
            
        try:
            # Check available GPU memory (simple check for >1GB)
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory_gb = gpu_props.total_memory / (1024**3)
            logging.info(f"GPU detected: {gpu_props.name}, Total Memory: {total_memory_gb:.2f} GB")
            
            # Example threshold: require at least 1GB VRAM for the index on GPU
            if total_memory_gb < 1.0:
                logging.info(f"GPU VRAM ({total_memory_gb:.2f}GB) is below threshold (1.0GB). Using CPU for FAISS index.")
                return
                
            logging.info("Attempting to move FAISS index to GPU...")
            self.gpu_res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, self.index)
            self.on_gpu = True
            logging.info("FAISS index successfully moved to GPU.")
        except Exception as e:
            logging.error(f"Failed to move FAISS index to GPU: {e}. Using CPU for index.")
            # Fallback to CPU index if GPU transfer fails
            if self.on_gpu:
                try:
                    self.index = faiss.index_gpu_to_cpu(self.index) # Ensure it's back on CPU
                except:
                    pass # Ignore errors during fallback attempt
            self.on_gpu = False
            
    @log_function_call
    def save_index(self):
        """Save the index and document IDs to disk."""
        try:
            index_to_save = self.index
            if self.on_gpu:
                logging.info("Moving index from GPU to CPU before saving...")
                index_to_save = faiss.index_gpu_to_cpu(self.index)
                logging.info("Index moved to CPU.")
                
            logging.info(f"Saving FAISS index to: {self.index_file}")
            faiss.write_index(index_to_save, self.index_file)
            
            logging.info(f"Saving document IDs to: {self.docs_file}")
            with open(self.docs_file, 'w', encoding='utf-8') as f:
                json.dump({'doc_ids': self.doc_ids}, f, ensure_ascii=False, indent=2)
                
            logging.info(f"Index and document IDs saved successfully. Total documents: {len(self.docs)}")
            return True
        except Exception as e:
            logging.error(f"Failed to save index and document IDs: {e}")
            return False
    
    @log_function_call
    def add_documents(self, documents=None, doc_ids=None):
        """
        Add documents to the index.
        
        Args:
            documents: List of document contents. If None, loads from DocumentManager.
            doc_ids: List of document IDs, must match documents length.
            
        Returns:
            Number of documents added.
        """
        if documents is None:
            logging.info("Loading documents from DocumentManager...")
            # Load incrementally by default (can be changed in DocumentManager logic if needed)
            documents, doc_ids = self.document_manager.load_documents(incremental=True)
            logging.info(f"Loaded {len(documents)} new/modified documents from manager.")
            
        if not documents:
            logging.info("No new documents to add to the index.")
            return 0
            
        if doc_ids is None:
            # Generate generic doc_ids if not provided
            start_index = len(self.docs) # Base new IDs on current count
            doc_ids = [f"doc_{start_index + i}" for i in range(len(documents))]
            logging.warning(f"Document IDs not provided, generated generic IDs: {doc_ids}")
            
        if len(documents) != len(doc_ids):
            logging.error(f"Mismatch between document count ({len(documents)}) and ID count ({len(doc_ids)}). Aborting add.")
            raise ValueError(f"Document count and ID count mismatch: {len(documents)} != {len(doc_ids)}")
            
        logging.info(f"Creating embeddings for {len(documents)} documents...")
        # Consider adding batching for large numbers of documents if encode supports it well
        embeddings = self.model.encode(documents, normalize_embeddings=True) # Normalize for L2 index
        logging.info("Embeddings created.")
        
        logging.info(f"Adding {len(embeddings)} embeddings to the FAISS index...")
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.docs.extend(documents)
        self.doc_ids.extend(doc_ids)
        logging.info(f"{len(embeddings)} embeddings added. Index now contains {self.index.ntotal} vectors.")
        
        # Save the updated index
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
            Tuple[List[str], List[float]]: 包含文档内容和相应分数的元组
        """
        if not self.index or self.index.ntotal == 0:
            logging.warning("Retrieval attempted but index is empty.")
            return [], []
            
        logging.debug(f"Encoding query for retrieval: '{query[:100]}...'")
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Ensure k is not greater than the number of documents in the index
        k = min(top_k, self.index.ntotal)
        logging.debug(f"Searching index for top {k} documents.")
        
        # Perform the search
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        # Process results
        docs = []
        doc_scores = []
        retrieved_ids = []
        
        if k > 0 and len(indices) > 0:
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.docs):
                    docs.append(self.docs[idx])
                    # 将L2距离转换为相似度分数（越小越好 -> 越大越好）
                    similarity = 1.0 / (1.0 + float(distances[0][i]))
                    doc_scores.append(similarity)
                    retrieved_ids.append(self.doc_ids[idx])
                else:
                    logging.warning(f"Index search returned invalid index: {idx}")
                
        logging.info(f"Query '{query[:50]}...' retrieved {len(docs)} documents with IDs: {retrieved_ids}")
        return docs, doc_scores
    
    @log_function_call
    def retrieve_with_metadata(self, query, top_k=3):
        """
        Retrieve documents with their IDs and similarity scores.
        
        Args:
            query: The query text.
            top_k: Number of results to return.
            
        Returns:
            List of tuples: (document_content, document_id, score)
        """
        if not self.index or self.index.ntotal == 0:
            logging.warning("Retrieval attempted but index is empty.")
            return []
            
        logging.debug(f"Encoding query for retrieval with metadata: '{query[:100]}...'")
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        k = min(top_k, self.index.ntotal)
        logging.debug(f"Searching index for top {k} documents with metadata.")
        
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        
        results = []
        if k > 0 and len(indices) > 0:
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.docs):
                    results.append((
                        self.docs[idx],         # Document content
                        self.doc_ids[idx],      # Document ID
                        float(distances[0][i])  # Distance score (lower is better for L2)
                    ))
                else:
                     logging.warning(f"Index search returned invalid index: {idx}")
                
        logging.info(f"Query '{query[:50]}...' retrieved {len(results)} documents with metadata.")
        return results

    @log_function_call
    def clear_index(self):
        """Clear the index and associated documents."""
        logging.info("Clearing FAISS index and document store...")
        # Recreate the index object
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        logging.debug("New empty FAISS index created.")
        
        # Move to GPU if applicable
        if self.on_gpu and hasattr(self, 'gpu_res'):
            try:
                self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, self.index)
                logging.info("New empty index moved to GPU.")
            except Exception as e:
                 logging.error(f"Failed to move new empty index to GPU: {e}")
                 self.on_gpu = False # Ensure state reflects reality
            
        # Clear document lists
        self.docs = []
        self.doc_ids = []
        logging.debug("In-memory document lists cleared.")
        
        # Delete index and docs files from disk
        deleted_files = []
        try:
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
                logging.info(f"Deleted index file: {self.index_file}")
                deleted_files.append(self.index_file)
            if os.path.exists(self.docs_file):
                os.remove(self.docs_file)
                logging.info(f"Deleted document mapping file: {self.docs_file}")
                deleted_files.append(self.docs_file)
        except OSError as e:
            logging.error(f"Error deleting index/docs files: {e}")
            
        logging.info(f"Index clear operation complete. Deleted files: {deleted_files}")
        return True 