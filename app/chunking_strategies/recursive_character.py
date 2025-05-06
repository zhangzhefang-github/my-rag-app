# app/chunking_strategies/recursive_character.py

import logging
from typing import List, Optional
import uuid # Import the uuid library

try:
    # Try importing from langchain_text_splitters first (newer structure)
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback for older langchain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from .base import BaseChunkingStrategy
from app.models.document import ParsedDocument, Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

class RecursiveCharacterChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks text using Langchain's RecursiveCharacterTextSplitter.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        # Ensure overlap is not larger than chunk size
        if chunk_overlap >= chunk_size:
            logger.warning(f"Chunk overlap ({chunk_overlap}) >= chunk size ({chunk_size}). Setting overlap to {chunk_size // 5}")
            chunk_overlap = chunk_size // 5 # Adjust overlap to a fraction of chunk size

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False, # Use default separators
            # separators=["\n\n", "\n", " ", ""] # Default separators
        )
        logger.info(f"Initialized RecursiveCharacterChunkingStrategy with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        """Chunks the document text and creates Chunk objects with metadata."""
        if not document.text:
            logger.warning(f"Document {document.doc_id} has empty text. Returning no chunks.")
            return []

        try:
            # Use the splitter to get text chunks
            text_chunks = self._splitter.split_text(document.text)
            logger.debug(f"Split document {document.doc_id} into {len(text_chunks)} text chunks.")

            chunks: List[Chunk] = []
            current_offset = 0
            for i, text_chunk in enumerate(text_chunks):
                start_offset = document.text.find(text_chunk, current_offset)
                if start_offset == -1:
                     # This might happen due to splitting/overlap complexities or identical chunks
                     # Try finding from the beginning as a fallback, but log a warning
                     start_offset = document.text.find(text_chunk) 
                     logger.warning(f"Could not find exact start offset for chunk {i} of doc {document.doc_id} sequentially. Found at {start_offset}. Offset accuracy might be reduced.")
                     if start_offset == -1:
                         logger.error(f"CRITICAL: Could not find text of chunk {i} '{text_chunk[:50]}...' in original document {document.doc_id}. Skipping chunk.")
                         continue # Skip this chunk if text can't be found

                # --- Prepare ChunkMetadata --- #
                # Start with the base metadata required by ChunkMetadata
                base_meta = {
                    "source": document.metadata.get("source", document.doc_id), # Default to doc_id if source missing
                    "chunk_index": i,
                    "offset": start_offset
                }
                # Merge the original document metadata into the chunk metadata
                # This assumes ChunkMetadata.Config has extra='allow'
                chunk_meta_dict = {**document.metadata, **base_meta}
                chunk_metadata = ChunkMetadata(**chunk_meta_dict)
                # ---------------------------- #

                # Create the Chunk object, NOW ADDING chunk_id
                chunk_id = str(uuid.uuid4())
                chunk = Chunk(
                    chunk_id=chunk_id, # Generate and assign a unique chunk ID
                    doc_id=document.doc_id,
                    text=text_chunk,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)

                # Update offset for next search to be slightly after the start of the current chunk
                # This helps find overlapping chunks correctly
                current_offset = start_offset + 1 # Move search start forward

            logger.info(f"Successfully created {len(chunks)} Chunk objects for document {document.doc_id}.")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking document {document.doc_id} with RecursiveCharacterTextSplitter: {e}", exc_info=True)
            return [] # Return empty list on error 