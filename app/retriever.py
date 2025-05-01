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
from utils.model_utils import ensure_dir_exists
from utils.document_manager import DocumentManager
from utils.logger import log_function_call
from app.embedding_strategies.config import EmbeddingConfig
from app.embedding_strategies.factory import get_embedder
from app.embedding_strategies.base import EmbeddingStrategy

class Retriever:
    """Vector retriever responsible for document vectorization and similarity search"""
    
    def __init__(self,
                 embedding_config: EmbeddingConfig,
                 index_dir: str,
                 docs_dir: str):
        """
        Initialize the Retriever.
        
        Args:
            embedding_config: Configuration object for the embedding strategy.
            index_dir: Directory to store Faiss index and document mapping files.
            docs_dir: Directory containing the original source documents.
        """
        # Log the embedding config safely (excluding sensitive info if possible)
        safe_config_dump = embedding_config.config.model_dump(exclude={'api_key'})
        logging.info(f"Initializing Retriever with config: embedding_provider={embedding_config.config.provider}, index_dir='{index_dir}', docs_dir='{docs_dir}', embedding_config={safe_config_dump}")

        self.embedding_config = embedding_config
        self.index_dir = index_dir
        self.docs_dir = docs_dir
        
        # Get manager instances
        self.gpu_manager = GPUManager()
        self.document_manager = DocumentManager(docs_dir=self.docs_dir)
        
        # Ensure index directory exists
        logging.debug(f"Ensuring index directory exists: {index_dir}")
        ensure_dir_exists(index_dir)
        
        # --- Initialize Embedder using the factory ---
        logging.info("Initializing embedding strategy...")
        try:
            self.embedder: EmbeddingStrategy = get_embedder(embedding_config)
            # Extract model name or path for file naming (best effort)
            if hasattr(embedding_config.config, 'model_path'):
                 # Primarily for HF models
                 model_identifier = embedding_config.config.model_path
            elif hasattr(embedding_config.config, 'model'):
                 # For OpenAI, Ollama models
                 model_identifier = embedding_config.config.model
            else:
                 model_identifier = embedding_config.config.provider # Fallback
            self.model_name_safe = model_identifier.replace('/', '_').replace(':','_') # Make it safe for filename
            logging.info(f"Embedding strategy initialized for provider: {embedding_config.config.provider}")
        except Exception as e:
            logging.error(f"Failed to initialize embedding strategy: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize embedder: {e}") from e
        # --- End Embedder Initialization ---
        
        # Generate safe filename for the index based on model identifier
        self.index_file = os.path.join(index_dir, f"{self.model_name_safe}.index")
        self.docs_file = os.path.join(index_dir, f"{self.model_name_safe}.docs.json")
        logging.debug(f"Index file path: {self.index_file}")
        logging.debug(f"Docs mapping file path: {self.docs_file}")
        
        # --- Determine Embedding Dimension ---
        # This is now tricky as EmbeddingStrategy doesn't have a get_dimension method yet.
        # Option 1: Try embedding a dummy text to get the dimension.
        # Option 2: Add a get_dimension method to the EmbeddingStrategy interface and implementations.
        # Option 3: Make dimension configurable or assume a default and resize Faiss index later.
        # Let's try Option 1 for now, but Option 2 is cleaner long-term.
        try:
            logging.info("Determining embedding dimension...")
            dummy_embedding = self.embedder.embed(["dimension_test"])
            self.embedding_dim = len(dummy_embedding[0])
            logging.info(f"Determined embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logging.error(f"Could not determine embedding dimension automatically: {e}. You might need to configure Faiss index dimension manually or enhance EmbeddingStrategy.", exc_info=True)
            # Fallback or raise error? Let's raise for now.
            raise RuntimeError(f"Failed to determine embedding dimension: {e}") from e
        # --- End Dimension Determination ---
        
        # Load or create the index (using the determined dimension)
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
                logging.info(f"Successfully loaded index. Dimension: {self.index.d}, Is Trained: {self.index.is_trained}, Total vectors: {self.index.ntotal}")
                
                # Check if loaded index dimension matches the current model's dimension
                if self.index.d != self.embedding_dim:
                    logging.warning(f"Loaded index dimension ({self.index.d}) does not match current embedding dimension ({self.embedding_dim}). Rebuilding index.")
                    raise ValueError("Index dimension mismatch")
                    
                logging.info(f"Attempting to load document mapping from: {self.docs_file}")
                with open(self.docs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.doc_ids = data.get('doc_ids', [])
                logging.info(f"Successfully loaded {len(self.doc_ids)} document IDs.")
                    
                # Load document content using DocumentManager
                logging.info(f"Loading content for {len(self.doc_ids)} documents...")
                loaded_count = 0
                missing_ids = []
                for doc_id in self.doc_ids:
                    content = self.document_manager.get_document(doc_id)
                    if content:
                        self.docs.append(content)
                        loaded_count += 1
                    else:
                        logging.warning(f"Could not load document content for ID: {doc_id}")
                        missing_ids.append(doc_id) # Keep track of missing ones

                if missing_ids:
                     logging.warning(f"Failed to load content for {len(missing_ids)} document IDs. Index may be stale.")
                     # Decide on recovery strategy: rebuild, remove missing, or just warn?
                     # For now, let's continue but log the warning prominently.

                logging.info(f"Successfully loaded content for {loaded_count}/{len(self.doc_ids)} documents.")
                
                # Check consistency between loaded docs and index size
                if self.index.ntotal != loaded_count:
                    logging.warning(f"Loaded document count ({loaded_count}) does not match Faiss index size ({self.index.ntotal}). Index might be corrupted or partially loaded. Rebuilding index.")
                    raise ValueError("Document count vs Index size mismatch")

                logging.info(f"Successfully loaded existing index and documents. Total documents: {len(self.docs)}")
            except Exception as e:
                logging.error(f"Failed to load existing index or documents: {e}. Creating a new index.", exc_info=True)
                # Create new index
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                self.docs = []
                self.doc_ids = []
        else:
            logging.info(f"Index file '{self.index_file}' or docs file '{self.docs_file}' not found. Creating a new FAISS index.")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Move index to GPU if possible
        logging.debug("Attempting to move index to GPU if available...")
        # Check GPU availability based on embedding config, not a separate flag
        self.use_gpu = self.embedding_config.config.use_gpu if hasattr(self.embedding_config.config, 'use_gpu') else False
        self._move_index_to_gpu_if_available()
        
    @log_function_call 
    def _move_index_to_gpu_if_available(self):
        """Move the FAISS index to GPU if configured and possible."""
        self.on_gpu = False
        
        # Check if embedder is configured to use GPU (best effort check)
        if not self.use_gpu:
            logging.info("GPU usage is disabled by embedding configuration (or default).")
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
            
        logging.info(f"Creating embeddings for {len(documents)} documents using {self.embedding_config.config.provider} strategy...")
        # --- Use the embedder strategy ---
        try:
            # Consider adding batching logic here if embedder handles it poorly
            # Note: Faiss IndexFlatL2 uses L2 distance, prefers normalized embeddings.
            # SentenceTransformer usually normalizes by default. Check other strategies.
            # Assuming embedder handles normalization or strategy includes it if needed.
            embeddings = self.embedder.embed(documents)
            # Convert to numpy float32 for Faiss
            embeddings_np = np.array(embeddings, dtype=np.float32)
            # Normalize embeddings here if the strategy doesn't guarantee it
            faiss.normalize_L2(embeddings_np)
            logging.info("Embeddings created and normalized.")
        except Exception as e:
             logging.error(f"Failed to create embeddings: {e}", exc_info=True)
             raise RuntimeError(f"Embedding creation failed: {e}") from e
        # --- End using embedder ---
        
        logging.info(f"Adding {len(embeddings_np)} embeddings to the FAISS index...")
        self.index.add(embeddings_np) # Add the numpy array
        self.docs.extend(documents)
        self.doc_ids.extend(doc_ids)
        logging.info(f"{len(embeddings_np)} embeddings added. Index now contains {self.index.ntotal} vectors.")
        
        # Save the updated index
        self.save_index()
        
        return len(documents)
    
    @log_function_call
    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve the most relevant documents for a given query.
        
        Args:
            query: The query text.
            top_k: The number of results to return.
            
        Returns:
            Tuple[List[str], List[float]]: Tuple containing document contents and corresponding scores.
        """
        if not self.index or self.index.ntotal == 0:
            logging.warning("Attempted to retrieve from an empty or non-existent index.")
            return [], [] # Return empty lists
        
        logging.info(f"Creating embedding for query: '{query}'")
        # --- Use the embedder strategy for query ---
        try:
            query_embedding = self.embedder.embed([query])
            query_embedding_np = np.array(query_embedding, dtype=np.float32)
             # Normalize query vector to match index normalization
            faiss.normalize_L2(query_embedding_np)
            logging.info("Query embedding created and normalized.")
        except Exception as e:
             logging.error(f"Failed to create query embedding: {e}", exc_info=True)
             return [], [] # Return empty on error
        # --- End using embedder ---
        
        logging.info(f"Searching index for top {top_k} results...")
        # D: distances (L2 squared), I: indices
        distances, indices = self.index.search(query_embedding_np, top_k)
        
        results_docs = []
        results_scores = []
        
        if indices.size > 0:
            for i, idx in enumerate(indices[0]): # indices is 2D array [[idx1, idx2, ...]]
                if idx != -1: # Faiss uses -1 for invalid index
                    if 0 <= idx < len(self.docs):
                        results_docs.append(self.docs[idx])
                        # Convert L2 squared distance to a similarity score (e.g., 1 / (1 + D))
                        # Or simply return the distance. Let's return distance for now.
                        results_scores.append(float(distances[0][i]))
                    else:
                         logging.warning(f"Retrieved index {idx} is out of bounds for loaded documents (count: {len(self.docs)}). Skipping.")
                else:
                    logging.debug(f"Index search returned invalid index -1 at position {i}.")

        logging.info(f"Retrieved {len(results_docs)} documents.")
        return results_docs, results_scores # Return docs and scores
    
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
        query_embedding = self.embedder.embed([query])
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
        """Clears the in-memory index and deletes the files from disk."""
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dim) # Recreate empty index
            self.docs = []
            self.doc_ids = []
            self._move_index_to_gpu_if_available() # Reset GPU state if applicable
            logging.info("In-memory index cleared.")

            if os.path.exists(self.index_file):
                os.remove(self.index_file)
                logging.info(f"Deleted index file: {self.index_file}")
            if os.path.exists(self.docs_file):
                os.remove(self.docs_file)
                logging.info(f"Deleted docs mapping file: {self.docs_file}")
            return True
        except Exception as e:
             logging.error(f"Error clearing index: {e}", exc_info=True)
             return False 