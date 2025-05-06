"""
文档检索器 - 负责文档向量化和相似性检索
"""

import faiss
import numpy as np
import torch
import os
import json
import logging
import time # Added for timing build index
from utils.gpu_manager import GPUManager
from utils.model_utils import ensure_dir_exists
from utils.document_manager import DocumentManager
from utils.logger import log_function_call
from app.embedding_strategies.config import EmbeddingConfig
from app.embedding_strategies.factory import get_embedder
from app.embedding_strategies.base import EmbeddingStrategy
from app.reranker_strategies.base import BaseRerankerStrategy
from app.reranker_strategies import get_reranker # Import the factory function
# Import chunking strategy and factory
from app.chunking_strategies.base import BaseChunkingStrategy
from app.chunking_strategies.factory import get_chunker
# Import core data models
from app.models.document import ParsedDocument, Chunk, ChunkMetadata, Citation # Added Citation for type hint consistency if needed later

from typing import List, Dict, Any, Optional, Tuple
from app.utils.env_helpers import _get_int_from_env, _get_bool_from_env

logger = logging.getLogger(__name__)

class Retriever:
    """Vector retriever responsible for document chunking, vectorization, and similarity search"""
    
    def __init__(self,
                 embedding_config: EmbeddingConfig,
                 docs_dir: str = "data/documents",
                 index_dir: str = "data/indexes",
                 top_k: Optional[int] = None,
                 rerank_k: Optional[int] = None,
                 rerank_top_n: Optional[int] = None,
                 use_reranker: Optional[bool] = None,
                 reranker_model: Optional[str] = None,
                 reranker_device: Optional[str] = None):
        """
        Initialize the Retriever.
        
        Args:
            embedding_config: Configuration object for the embedding strategy.
            docs_dir: Directory containing the original source documents.
            index_dir: Directory to store Faiss index and chunk mapping files.
            top_k: Default number of chunks to retrieve. Reads from TOP_K env var if None.
            rerank_k: Number of chunks to pass to the reranker. Reads from RERANK_K env var if None.
            rerank_top_n: Number of chunks to return after reranking. Reads from RERANK_TOP_N env var if None.
            use_reranker: Whether to use reranking. Reads from USE_RERANKER env var if None.
            reranker_model: Model name/path for the reranker. Reads from RERANKER_MODEL env var if None.
            reranker_device: Device for the reranker. Reads from RERANKER_DEVICE env var if None.
        """
        # --- Initialize core attributes FIRST --- 
        self.embedding_config = embedding_config
        self.docs_dir = docs_dir
        self.index_dir = index_dir
        # doc_mapping now stores Chunk objects, not just dicts
        self.chunk_mapping: List[Chunk] = [] 
        self.index: Optional[faiss.Index] = None
        self.on_gpu = False # Initialize GPU flag early
        self.reranker: Optional[BaseRerankerStrategy] = None # Initialize reranker early
        self.chunker: BaseChunkingStrategy = None # Initialize chunker
        self.embedding_dim: Optional[int] = None # Initialize embedding dim
        # --- End core attribute init ---

        # Log the embedding config safely
        safe_config_dump = embedding_config.config.model_dump(exclude={'api_key'}) if hasattr(embedding_config.config, 'model_dump') else embedding_config.config.__dict__
        logger.info(f"Initializing Retriever with config: embedding_provider={embedding_config.config.provider}, docs_dir='{docs_dir}', index_dir='{index_dir}', embedding_config={safe_config_dump}")

        # --- Get configuration from environment variables (or args) ---
        self.top_k = top_k if top_k is not None else _get_int_from_env("TOP_K", 5)
        # Read reranker config from args or env vars
        self.use_reranker = use_reranker if use_reranker is not None else _get_bool_from_env("USE_RERANKER", True)
        _reranker_model_name = reranker_model if reranker_model is not None else os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
        _reranker_device_name = reranker_device if reranker_device is not None else os.environ.get("RERANKER_DEVICE", "auto")

        self.rerank_top_n = rerank_top_n if rerank_top_n is not None else _get_int_from_env("RERANK_TOP_N", self.top_k)
        self.rerank_k = rerank_k if rerank_k is not None else _get_int_from_env("RERANK_K", max(self.rerank_top_n * 2, 10))

        logger.info(f"Retriever config: top_k={self.top_k}, use_reranker={self.use_reranker}, "
                    f"rerank_k={self.rerank_k}, rerank_top_n={self.rerank_top_n}, "
                    f"reranker_model='{_reranker_model_name}', reranker_device='{_reranker_device_name}'")
        # --- End environment variable configuration ---
        
        # Get manager instances
        self.gpu_manager = GPUManager()
        # Document Manager now primarily used inside build_index
        self.document_manager = DocumentManager(docs_dir=self.docs_dir)
        
        # Ensure index directory exists
        logger.debug(f"Ensuring index directory exists: {index_dir}")
        ensure_dir_exists(index_dir)
        
        # --- Initialize Embedder --- 
        logger.info("Initializing embedding strategy...")
        try:
            self.embedder: EmbeddingStrategy = get_embedder(embedding_config)
            # Determine safe model name for filenames
            model_identifier = getattr(embedding_config.config, 'model_path', getattr(embedding_config.config, 'model', embedding_config.config.provider))
            self.model_name_safe = model_identifier.replace('/', '_').replace(':', '_') # Make it safe for filename
            logger.info(f"Embedding strategy initialized for provider: {embedding_config.config.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding strategy: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize embedder: {e}") from e
        # --- End Embedder Initialization ---

        # --- Initialize Chunker --- 
        logger.info("Initializing chunking strategy...")
        try:
            self.chunker = get_chunker() # Use the factory
            logger.info(f"Chunking strategy initialized: {type(self.chunker).__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize chunking strategy: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize chunker: {e}") from e
        # --- End Chunker Initialization ---
        
        # Generate safe filename for the index based on model identifier
        self.index_file = os.path.join(index_dir, f"{self.model_name_safe}.index")
        # Docs file now stores chunk mapping
        self.chunks_file = os.path.join(index_dir, f"{self.model_name_safe}.chunks.json") 
        logger.debug(f"Index file path: {self.index_file}")
        logger.debug(f"Chunks mapping file path: {self.chunks_file}")
        
        # --- Determine Embedding Dimension --- 
        # Embedding dimension needs to be determined before loading/building index
        try:
            logger.info("Determining embedding dimension...")
            dummy_embedding = self.embedder.embed(["dimension_test"])
            self.embedding_dim = len(dummy_embedding[0])
            logger.info(f"Determined embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Could not determine embedding dimension automatically: {e}. Faiss index might fail.", exc_info=True)
            # Allow proceeding but log error - index building might fail later
            self.embedding_dim = None # Indicate dimension is unknown
        # --- End Dimension Determination ---
        
        # Load or create the index (requires dimension)
        logger.info("Loading or creating FAISS index and chunk mapping...")
        self._load_or_build_index()
        logger.info("FAISS index and chunk mapping ready.")
        
        # --- Reranker Initialization --- 
        if self.use_reranker:
             if not _reranker_model:
                 logger.warning("Reranking is enabled (USE_RERANKER=true) but RERANKER_MODEL is not set. Reranking will be disabled.")
                 self.use_reranker = False
             else:
                 logger.info(f"Reranker is enabled. Initializing reranker '{_reranker_model}'...")
                 try:
                     # Use reranker factory
                     self.reranker = get_reranker(model_name=_reranker_model)
                     logger.info("Reranker initialized successfully.")
                 except Exception as e:
                     logger.error(f"Failed to initialize Reranker: {e}. Reranking will be disabled.", exc_info=True)
                     self.reranker = None
                     self.use_reranker = False
        else:
             logger.info("Reranker is disabled.")
        # --- End Reranker Initialization ---
        
        # Document manager instance might still be useful for adding/deleting docs later
        # self.doc_manager = DocumentManager(docs_dir=self.docs_dir) 
        # Already initialized earlier
        
    def _load_or_build_index(self):
        """Loads the Faiss index and chunk mapping if they exist, otherwise builds them."""
        os.makedirs(self.index_dir, exist_ok=True)
        load_successful = False

        # Check if index AND chunk mapping files exist
        if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
            logger.info(f"Found existing index file ({self.index_file}) and chunk mapping file ({self.chunks_file}). Attempting to load...")
            try:
                logger.info(f"Loading existing Faiss index from {self.index_file}")
                self.index = faiss.read_index(self.index_file)
                logger.info(f"Loaded Faiss index with {self.index.ntotal} vectors.")

                # Verify dimension if known
                if self.embedding_dim is not None and self.index.d != self.embedding_dim:
                     logger.warning(f"Loaded index dimension ({self.index.d}) != expected embedding dimension ({self.embedding_dim}). Index needs rebuild.")
                     raise ValueError("Index dimension mismatch")
                elif self.embedding_dim is None:
                    # If dimension wasn't determined earlier, set it from the loaded index
                    self.embedding_dim = self.index.d
                    logger.info(f"Embedding dimension set from loaded index: {self.embedding_dim}")

                logger.info(f"Loading chunk mapping from {self.chunks_file}")
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    # Load JSON data which is a list of dicts
                    chunks_data = json.load(f)
                
                # --- Validate and reconstruct Chunk objects --- 
                if not isinstance(chunks_data, list):
                    logger.warning(f"Loaded chunk mapping from {self.chunks_file} is not a list (type: {type(chunks_data)}). File might be corrupted. Index needs rebuild.")
                    raise TypeError("Invalid chunk_mapping format")

                self.chunk_mapping = []
                for i, chunk_dict in enumerate(chunks_data):
                     try:
                         # Reconstruct Chunk objects using Pydantic validation
                         # Assumes chunk_dict structure matches Chunk.model_validate (Pydantic v2)
                         # or Chunk.parse_obj (Pydantic v1)
                         if hasattr(Chunk, 'model_validate'):
                             chunk = Chunk.model_validate(chunk_dict)
                         else:
                             chunk = Chunk.parse_obj(chunk_dict)
                         self.chunk_mapping.append(chunk)
                     except Exception as pydantic_e:
                         logger.error(f"Failed to parse chunk data at index {i} from {self.chunks_file}: {pydantic_e}. Chunk data: {chunk_dict}. Rebuilding index.", exc_info=True)
                         raise TypeError(f"Invalid chunk data format: {pydantic_e}") from pydantic_e
                # --- End Validation --- 

                # Verify consistency: number of vectors in index vs number of chunks loaded
                if self.index.ntotal != len(self.chunk_mapping):
                     logger.warning(f"Loaded chunk mapping count ({len(self.chunk_mapping)}) != Faiss index size ({self.index.ntotal}). Index needs rebuild.")
                     raise ValueError("Index vs chunk mapping count mismatch")

                logger.info(f"Index and chunk mapping loaded successfully. Contains {len(self.chunk_mapping)} chunks.")
                self._move_index_to_gpu_if_available() # Attempt moving loaded index to GPU
                load_successful = True # Mark as successful

            except Exception as e:
                 logger.error(f"Failed to load existing index/mapping: {e}. Rebuilding index.", exc_info=True)
                 # Clean up potentially problematic files before rebuild
                 self._cleanup_index_files()
                 # Ensure variables are reset for rebuild
                 self.index = None
                 self.chunk_mapping = []
        else:
            logger.info(f"Index file ({self.index_file}) or chunk mapping file ({self.chunks_file}) not found. Index needs to be built.")
        
        if not load_successful:
            logger.info("Building new index from documents...")
            start_time = time.time()
            self.build_index()
            end_time = time.time()
            logger.info(f"Index building completed in {end_time - start_time:.2f} seconds.")

    def _cleanup_index_files(self):
        """Safely deletes existing index and chunk mapping files."""
        try:
            if os.path.exists(self.index_file):
                os.remove(self.index_file)
                logger.info(f"Deleted potentially problematic index file: {self.index_file}")
            if os.path.exists(self.chunks_file):
                os.remove(self.chunks_file)
                logger.info(f"Deleted potentially problematic chunk mapping file: {self.chunks_file}")
        except OSError as del_e:
            logger.error(f"Error deleting old index/mapping files: {del_e}", exc_info=True)
            
    def build_index(self):
        """Builds the Faiss index from documents by chunking, embedding, and indexing."""
        # 1. Load ALL documents directly for a full rebuild
        logger.info(f"Loading documents from {self.docs_dir} for full index rebuild...")
        # Call load_documents directly to get the list for processing
        # Pass incremental=False to ensure all documents are loaded
        # This returns List[str] content and List[str] doc_ids
        try:
            loaded_contents, loaded_doc_ids = self.document_manager.load_documents(incremental=False)
        except Exception as load_e:
             logger.error(f"Error loading documents via DocumentManager: {load_e}", exc_info=True)
             # If loading fails, we cannot proceed.
             self._create_empty_index() # Create empty index as fallback
             return

        if not loaded_contents:
            logger.warning(f"No documents successfully loaded from {self.docs_dir}. Index will be empty.")
            self._create_empty_index()
            return

        # Convert loaded data to ParsedDocument objects
        parsed_documents: List[ParsedDocument] = []
        for content, doc_id in zip(loaded_contents, loaded_doc_ids):
            # Get metadata from document manager (which should now be populated after load_documents)
            doc_meta = self.document_manager.get_document_metadata(doc_id) or {}
            # Ensure 'source' is in metadata, defaulting to doc_id
            if 'source' not in doc_meta:
                doc_meta['source'] = doc_id
            parsed_doc = ParsedDocument(doc_id=doc_id, text=content, metadata=doc_meta)
            parsed_documents.append(parsed_doc)
        logger.info(f"Loaded and converted {len(parsed_documents)} documents to be processed.")

        # 2. Chunk documents
        logger.info(f"Chunking documents using {type(self.chunker).__name__}...")
        all_chunks: List[Chunk] = []
        for doc in parsed_documents:
            try:
                doc_chunks = self.chunker.chunk(doc)
                all_chunks.extend(doc_chunks)
            except Exception as chunk_e:
                logger.error(f"Failed to chunk document {doc.doc_id}: {chunk_e}", exc_info=True)
                # Optionally skip problematic docs or raise error

        if not all_chunks:
            logger.warning("No chunks were generated from the documents. Index will be empty.")
            self._create_empty_index()
            return

        logger.info(f"Generated a total of {len(all_chunks)} chunks.")
        self.chunk_mapping = all_chunks # Store the generated chunks

        # 3. Embed chunks
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        chunk_texts = [chunk.text for chunk in all_chunks]
        try:
            batch_size = _get_int_from_env("EMBED_BATCH_SIZE", 32)
            all_embeddings = []
            num_batches = (len(chunk_texts) + batch_size - 1) // batch_size
            for i in range(0, len(chunk_texts), batch_size):
                 batch = chunk_texts[i:i+batch_size]
                 logger.debug(f"Embedding batch {i//batch_size + 1}/{num_batches} (size {len(batch)})..." )
                 batch_embeddings = self.embedder.embed(batch)
                 all_embeddings.extend(batch_embeddings)
                 # Optional: Add a small sleep or yield for very large datasets
                 # await asyncio.sleep(0.01)

            if not all_embeddings:
                 logger.error("Embedding process resulted in an empty list. Cannot build index.")
                 self._create_empty_index()
                 return

            embeddings_np = np.array(all_embeddings).astype('float32')

            # Verify and potentially set embedding dimension
            generated_dim = embeddings_np.shape[1]
            if self.embedding_dim is None:
                 logger.info(f"Setting embedding dimension based on generated embeddings: {generated_dim}")
                 self.embedding_dim = generated_dim
            elif generated_dim != self.embedding_dim:
                 logger.error(f"Generated embedding dimension ({generated_dim}) does not match expected ({self.embedding_dim})! Index building failed.")
                 raise ValueError("Embedding dimension mismatch during index build.")

            faiss.normalize_L2(embeddings_np) # Normalize for L2 distance (cosine similarity)
            logger.info(f"Embeddings generated and normalized. Shape: {embeddings_np.shape}")

        except Exception as e:
             logger.error(f"Failed to generate embeddings during index build: {e}", exc_info=True)
             self._cleanup_index_files()
             raise RuntimeError(f"Embedding generation failed: {e}") from e

        # 4. Build Faiss index
        if self.embedding_dim is None:
             logger.error("Cannot build Faiss index: embedding dimension is unknown.")
             raise RuntimeError("Failed to determine embedding dimension before index creation.")

        logger.info(f"Building Faiss index (IndexFlatL2) with dimension {self.embedding_dim}...")
        try:
            self.index = faiss.IndexFlatL2(self.embedding_dim) # Using L2 distance for normalized vectors = cosine similarity
            self.index.add(embeddings_np)
            logger.info(f"Faiss index built. Total vectors: {self.index.ntotal}")
        except Exception as faiss_e:
             logger.error(f"Failed to build Faiss index: {faiss_e}", exc_info=True)
             self._cleanup_index_files()
             raise RuntimeError(f"Faiss index building failed: {faiss_e}") from faiss_e

        # 5. Save index and chunk mapping
        self._save_index_and_mapping()
        # 6. Move to GPU if applicable
        self._move_index_to_gpu_if_available()

    def _create_empty_index(self):
        """Creates an empty Faiss index and saves corresponding empty mapping."""
        if self.embedding_dim is None:
             logger.error("Cannot create empty Faiss index: embedding dimension is unknown.")
             # Optionally, try to determine dimension again or use a default? Risky.
             # For now, we probably can't proceed without a dimension.
             # Let's assume dimension determination succeeded earlier if we reach here.
             # If self.embedding_dim is STILL None, an error should have been raised before.
             # However, defensive check:
             if self.embedding_dim is None:
                 raise RuntimeError("Cannot create empty index without a known embedding dimension.")
                 
        logger.info(f"Creating empty Faiss index with dimension {self.embedding_dim}.")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.chunk_mapping = []
        self._save_index_and_mapping() # Save the empty state
        self._move_index_to_gpu_if_available() # Handle GPU for empty index

    def _save_index_and_mapping(self):
        """Saves the Faiss index and the chunk mapping to disk."""
        if self.index is None:
            logger.error("Cannot save index: Index object is None.")
            return
            
        logger.info(f"Saving Faiss index to {self.index_file} ({self.index.ntotal} vectors)")
        try:
            # If index is on GPU, move it back to CPU before saving
            if self.on_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                logger.debug("Moving index from GPU to CPU for saving...")
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, self.index_file)
                logger.debug("Index saved from GPU to CPU temporary copy.")
            else:
                faiss.write_index(self.index, self.index_file)
            logger.info("Faiss index saved successfully.")
        except Exception as e:
             logger.error(f"Failed to save Faiss index to {self.index_file}: {e}", exc_info=True)
             # Consider if we should raise an error here

        logger.info(f"Saving chunk mapping to {self.chunks_file} ({len(self.chunk_mapping)} chunks)")
        try:
            # Convert Chunk objects to dictionaries for JSON serialization
            chunks_data = []
            for chunk in self.chunk_mapping:
                 # Use Pydantic's serialization method
                 if hasattr(chunk, 'model_dump'): # Pydantic v2
                     chunks_data.append(chunk.model_dump(mode='json'))
                 else: # Pydantic v1
                     chunks_data.append(chunk.dict())
                     
            with open(self.chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
            logger.info("Chunk mapping saved successfully.")
        except Exception as e:
             logger.error(f"Failed to save chunk mapping to {self.chunks_file}: {e}", exc_info=True)

    @log_function_call
    def _move_index_to_gpu_if_available(self):
        """Moves the loaded/built Faiss index to GPU if available and configured."""
        # Check if embedder is configured to use GPU
        should_use_gpu = getattr(self.embedding_config.config, 'use_gpu', False)

        if not should_use_gpu:
            logger.info("GPU usage not configured for embedder. Keeping index on CPU.")
            self.on_gpu = False
            return

        # Check if GPUManager reports availability
        if not self.gpu_manager.gpu_available:
            logger.warning("GPU usage configured, but no compatible GPU detected by GPUManager. Keeping index on CPU.")
            self.on_gpu = False
            return

        if self.index is None:
            logger.warning("Attempted to move index to GPU, but index is not loaded.")
            return

        if self.on_gpu:
            logger.info("Index is already on GPU.")
            return

        logger.info("Attempting to move Faiss index to GPU...")
        try:
            # Get the first available GPU ID from the manager
            gpu_ids = self.gpu_manager.get_available_gpu_ids() # Renamed variable for clarity
            if not gpu_ids:
                 logger.warning("GPUManager reported no available GPU IDs. Cannot move index to GPU.")
                 return # Should not happen if gpu_available was True, but safety check
            gpu_id = gpu_ids[0] # Use the first one

            # Check for StandardGpuResources before using it
            if hasattr(faiss, 'StandardGpuResources'):
                res = faiss.StandardGpuResources() # Use standard GPU resources
                # Important: Ensure index is not already a GpuIndex variant before wrapping
                if not isinstance(self.index, faiss.GpuIndex):
                     self.index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)
                     self.on_gpu = True
                     logger.info(f"Successfully moved Faiss index to GPU {gpu_id}.")
                else:
                     logger.info(f"Index is already a GpuIndex type on GPU {gpu_id}. No move needed.")
                     self.on_gpu = True # Ensure flag is set correctly
            else:
                 logger.error("Loaded 'faiss' module does not have 'StandardGpuResources'. Cannot move index to GPU automatically. Ensure 'faiss-gpu' is correctly installed and compatible.")
                 # Keep self.on_gpu as False
                 self.on_gpu = False

        except AttributeError as ae:
            # Catch cases where faiss might lack GPU functions altogether
            logger.error(f"Failed to move Faiss index to GPU: AttributeError ('{ae}'). Likely using faiss-cpu or incompatible faiss-gpu. Falling back to CPU.")
            self.on_gpu = False
        except Exception as e:
            # Catch other potential errors during GPU transfer
            logger.error(f"Failed to move Faiss index to GPU: {e}. Falling back to CPU.", exc_info=True)
            # Attempt to ensure index is usable on CPU if transfer fails mid-way?
            # This might require reloading from disk or keeping a CPU copy.
            # For now, just mark as not on GPU.
            self.on_gpu = False
    
    @log_function_call
    def retrieve(self, query: str, custom_top_k: Optional[int] = None) -> List[Chunk]:
        """Retrieves relevant chunks for a given query."""
        if self.index is None or self.embedding_dim is None:
            logger.error("Cannot retrieve: Index not initialized or embedding dimension unknown.")
            return []
        if not self.chunk_mapping:
            logger.warning("Retrieval attempted but chunk mapping is empty. Returning no results.")
            return []

        # Determine K based on whether reranking is used
        k_initial_retrieval = self.rerank_k if self.use_reranker and self.reranker else self.top_k
        # Ensure k is not larger than the number of items in the index
        k_initial_retrieval = min(k_initial_retrieval, self.index.ntotal)
        
        # Override with custom_top_k if provided AND reranker is NOT used
        # If reranker IS used, custom_top_k should ideally control rerank_top_n, not k_initial_retrieval
        if not self.use_reranker and custom_top_k is not None:
             k_to_use = min(custom_top_k, self.index.ntotal)
        else:
             k_to_use = k_initial_retrieval

        if k_to_use <= 0:
            logger.warning(f"Retrieval k ({k_to_use}) is zero or negative. Returning no results.")
            return []

        try:
            logger.debug(f"Generating query embedding for: '{query[:50]}...'")
            query_embedding = self.embedder.embed([query])
            query_vector = np.array(query_embedding).astype('float32')
            faiss.normalize_L2(query_vector) # Normalize query vector
            logger.debug(f"Query vector generated, shape: {query_vector.shape}")

            # Perform Faiss search
            logger.debug(f"Performing Faiss search for top {k_to_use} chunks...")
            # D: distances (lower is better for L2), I: indices
            distances, indices = self.index.search(query_vector, k_to_use)
            logger.debug(f"Faiss search completed. Found indices: {indices[0]}")
            
            # Map indices to actual Chunk objects
            retrieved_chunks: List[Chunk] = []
            retrieved_scores: List[float] = []
            # indices[0] contains the list of indices for the first (and only) query vector
            for i, index in enumerate(indices[0]):
                if index != -1 and index < len(self.chunk_mapping): # Check for valid index
                    chunk = self.chunk_mapping[index]
                    # Calculate similarity score (e.g., 1 - L2_distance/2 for normalized vectors, or use dot product)
                    # For normalized L2 distance, score = 1 - D^2 / 2. 
                    # Simpler: Higher dot product = higher cosine similarity. Let's use distance for now.
                    # Score interpretation: lower distance = more similar. Can convert later.
                    score = float(distances[0][i]) # Store L2 distance for now
                    retrieved_chunks.append(chunk)
                    retrieved_scores.append(score)
                else:
                     logger.warning(f"Faiss returned invalid index: {index}. Skipping.")

            logger.info(f"Initial retrieval found {len(retrieved_chunks)} chunks.")

            # --- Reranking Step --- 
            if self.use_reranker and self.reranker and retrieved_chunks:
                 logger.info(f"Applying reranker to {len(retrieved_chunks)} initially retrieved chunks (target top {self.rerank_top_n})...")
                 try:
                     # Reranker expects List[Tuple[str, str]] or List[Dict[str, str]] (query, passage_text)
                     # Prepare passages for reranker
                     passages = [chunk.text for chunk in retrieved_chunks]
                     
                     # Get reranking scores (higher is better)
                     rerank_scores = self.reranker.rerank(query, passages)
                     
                     # Combine chunks with their rerank scores and sort
                     # Ensure rerank_scores list matches retrieved_chunks list length
                     if len(rerank_scores) != len(retrieved_chunks):
                         logger.error(f"Mismatch between rerank scores ({len(rerank_scores)}) and retrieved chunks ({len(retrieved_chunks)}). Skipping reranking.")
                     else:
                         # Add rerank_score to chunk metadata (or a temporary structure)
                         chunks_with_scores = sorted(
                             zip(retrieved_chunks, rerank_scores),
                             key=lambda item: item[1], # Sort by rerank score
                             reverse=True # Higher score is better
                         )
                         
                         # Select top N after reranking
                         reranked_chunks = [chunk for chunk, score in chunks_with_scores[:self.rerank_top_n]]
                         reranked_final_scores = [score for chunk, score in chunks_with_scores[:self.rerank_top_n]]
                         
                         logger.info(f"Reranking complete. Selected top {len(reranked_chunks)} chunks.")
                         # Log scores if needed
                         # for chunk, score in zip(reranked_chunks, reranked_final_scores):
                         #    logger.debug(f"  Reranked: score={score:.4f}, chunk_idx={chunk.metadata.chunk_index}, doc={chunk.metadata.source}")
                         return reranked_chunks # Return the reranked list
                 
                 except Exception as rerank_e:
                      logger.error(f"Reranker failed: {rerank_e}. Returning initially retrieved chunks.", exc_info=True)
                      # Fallback to returning non-reranked chunks (up to top_k)
                      return retrieved_chunks[:self.top_k] # Use original top_k if reranker fails

            # If no reranking, return the initially retrieved chunks (up to original top_k)
            # We already retrieved k_to_use based on reranker status, so just return them.
            # If reranker wasn't used, k_to_use = self.top_k (or custom_top_k)
            logger.debug(f"Returning {len(retrieved_chunks)} chunks (no reranking or reranker failed).")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Error during retrieval for query '{query}': {e}", exc_info=True)
            return []

    @log_function_call
    def clear_index(self):
        """Clears the loaded index and chunk mapping from memory and deletes files."""
        logger.info("Clearing index and chunk mapping...")
        self.index = None
        self.chunk_mapping = []
        self.on_gpu = False # Reset GPU flag
        self._cleanup_index_files()
        logger.info("Index and mapping cleared.")

    # Deprecated retrieve_with_metadata
    # @log_function_call
    # def retrieve_with_metadata(self, query, top_k=3):
    #     logger.warning("retrieve_with_metadata is deprecated. Use retrieve() which returns Chunk objects with metadata.")
    #     chunks = self.retrieve(query, custom_top_k=top_k)
    #     # Convert List[Chunk] back to List[Dict] for old compatibility (if needed)
    #     # This might not be necessary if caller adapts to Chunk objects
    #     return [chunk.model_dump(mode='json') if hasattr(chunk, 'model_dump') else chunk.dict() for chunk in chunks]

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Deprecate old save_index method, use _save_index_and_mapping --- 
    # @log_function_call
    # def save_index(self):
    #     """DEPRECATED: Use internal _save_index_and_mapping during build/load."""
    #     logger.warning("save_index() is deprecated. Saving happens internally.")
    #     self._save_index_and_mapping()

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # Placeholder return
    #     return 0

    # --- Add Documents (Needs Rework for Chunking) --- 
    # @log_function_call
    # def add_documents(self, documents=None, doc_ids=None):
    #     """Adds new documents, chunks them, embeds, and adds to the index.
    #        NOTE: This needs significant rework for handling chunking and efficient updates.
    #        Current implementation might be inefficient or incorrect for incremental adds.
    #        Consider using a vector DB with better CRUD support for frequent updates.
    #     """
    #     logger.warning("add_documents method needs rework for chunk-based indexing and updates.")
    #     if not documents or not self.index:
    #         logger.error("Cannot add documents: No documents provided or index not initialized.")
    #         return 0 # Return 0 added
        
    #     # 1. Convert to ParsedDocument (assuming documents are List[str])
    #     # This needs a more robust way to handle doc_ids and metadata
    #     # ... implementation needed ...
        
    #     # 2. Chunk the new documents
    #     # ... implementation needed ...
        
    #     # 3. Embed the new chunks
    #     # ... implementation needed ...
        
    #     # 4. Add to Faiss index (consider using add_with_ids for potential deletion later)
    #     # ... implementation needed ...
        
    #     # 5. Update chunk_mapping
    #     # ... implementation needed ...
        
    #     # 6. Resave index and mapping? (Might be slow)
    #     # ... implementation needed ...
        
    #     # This needs a more robust way