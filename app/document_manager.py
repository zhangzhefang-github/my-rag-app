import os
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime

# Assuming TextLoader is defined correctly
from utils.loaders import TextLoader

logger = logging.getLogger(__name__)

class DocumentManager:
    """Handles loading documents from a directory and tracking their state."""
    STATE_FILENAME = "document_state.json"

    def __init__(self, docs_dir: str = "data/documents"):
        self.docs_dir = Path(docs_dir)
        self.state_path = self.docs_dir / self.STATE_FILENAME
        self.loaded_documents_state: Dict[str, Dict[str, Any]] = {}
        # Store metadata separately for quick access
        self._document_metadata: Dict[str, Dict[str, Any]] = {}

        if not self.docs_dir.exists():
            logger.warning(f"Documents directory '{self.docs_dir}' not found. Creating it.")
            self.docs_dir.mkdir(parents=True, exist_ok=True)
        else:
             logger.info(f"DocumentManager initialized. Using document directory: '{self.docs_dir}'")

        self._load_state()

    def _load_state(self):
        """Loads the tracking state from the JSON file."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    self.loaded_documents_state = json.load(f)
                logger.info(f"Loaded document state for {len(self.loaded_documents_state)} files from '{self.state_path}'.")
                # Populate metadata cache from loaded state
                for doc_id, state_info in self.loaded_documents_state.items():
                     if "metadata" in state_info:
                         self._document_metadata[doc_id] = state_info["metadata"]
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from '{self.state_path}'. Starting with empty state.")
                self.loaded_documents_state = {}
            except Exception as e:
                logger.error(f"Error loading state file '{self.state_path}': {e}. Starting with empty state.")
                self.loaded_documents_state = {}
        else:
            logger.info("Document state file not found. Starting with empty state.")
            self.loaded_documents_state = {}

    def _save_state(self):
        """Saves the current tracking state to the JSON file."""
        try:
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(self.loaded_documents_state, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved document state for {len(self.loaded_documents_state)} files to '{self.state_path}'.")
        except Exception as e:
            logger.error(f"Error saving state file '{self.state_path}': {e}")

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculates the SHA256 checksum of a file."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as file:
                while chunk := file.read(4096):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except IOError as e:
            logger.error(f"Could not read file {file_path} for checksum: {e}")
            return ""

    def _get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extracts basic metadata from a file."""
        try:
            stat_info = file_path.stat()
            return {
                "source": file_path.name, # Use filename as source identifier
                "file_path": str(file_path.resolve()),
                "size_bytes": stat_info.st_size,
                "last_modified": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                # Add other potential metadata extraction here (e.g., file type)
            }
        except Exception as e:
            logger.warning(f"Could not get metadata for file {file_path}: {e}")
            return {"source": file_path.name, "file_path": str(file_path.resolve())}

    def load_documents(self, incremental: bool = True) -> Tuple[List[str], List[str]]:
        """Loads documents from the directory.

        Args:
            incremental: If True, only loads new or modified documents based on state.
                         If False, loads all documents regardless of state.

        Returns:
            A tuple containing: (list of document contents, list of document IDs)
        """
        logger.info(f"Starting document loading (Incremental: {incremental}). Searching in '{self.docs_dir}'...")
        documents_content = []
        documents_ids = []
        updated_state = {} # Track state for files processed in this run
        files_processed = 0
        files_skipped = 0
        files_added = 0
        files_updated = 0

        # Use TextLoader for loading .txt files (can be extended for other types)
        loader = TextLoader()

        for file_path in self.docs_dir.rglob('*'): # Recursively find all files
            if file_path.is_file() and file_path.suffix.lower() == '.txt' and file_path.name != self.STATE_FILENAME:
                files_processed += 1
                doc_id = file_path.name # Use filename as document ID for simplicity
                logger.debug(f"Processing file: {file_path}")

                current_checksum = self._calculate_checksum(file_path)
                if not current_checksum:
                    logger.warning(f"Skipping file {file_path} due to checksum error.")
                    files_skipped += 1
                    continue

                current_metadata = self._get_file_metadata(file_path)
                load_this_file = False
                reason = ""

                if not incremental:
                    load_this_file = True
                    reason = "Full load requested."
                else:
                    if doc_id not in self.loaded_documents_state:
                        load_this_file = True
                        reason = "New file detected."
                        files_added += 1
                    elif self.loaded_documents_state[doc_id].get('checksum') != current_checksum:
                        load_this_file = True
                        reason = "File modified (checksum changed)."
                        files_updated += 1
                    else:
                        # File exists and checksum matches, skip loading content but update state
                        reason = "File unchanged."
                        files_skipped += 1
                        # Keep existing state info, but update metadata in case file path changed slightly etc.
                        updated_state[doc_id] = self.loaded_documents_state[doc_id]
                        updated_state[doc_id]["metadata"] = current_metadata
                        self._document_metadata[doc_id] = current_metadata # Update metadata cache

                if load_this_file:
                    logger.info(f"Loading content from '{file_path}'. Reason: {reason}")
                    try:
                        content = loader.load(file_path)
                        if content:
                            documents_content.append(content)
                            documents_ids.append(doc_id)
                            # Update state for the newly loaded/updated document
                            updated_state[doc_id] = {
                                "checksum": current_checksum,
                                "loaded_at": datetime.now().isoformat(),
                                "metadata": current_metadata
                            }
                            self._document_metadata[doc_id] = current_metadata # Update metadata cache
                            logger.debug(f"Successfully loaded content for {doc_id}.")
                        else:
                            logger.warning(f"Loader returned empty content for {file_path}. Skipping.")
                            files_skipped += 1
                    except Exception as e:
                        logger.error(f"Error loading document {file_path}: {e}", exc_info=True)
                        files_skipped += 1
                else:
                     logger.debug(f"Skipping file {file_path}. Reason: {reason}")

            elif file_path.is_file() and file_path.name != self.STATE_FILENAME:
                 logger.debug(f"Skipping unsupported file type: {file_path}")
                 files_skipped += 1

        # Update the main state with the results of this run
        # If incremental, keep state for files not seen in this run
        if incremental:
             final_state = self.loaded_documents_state.copy()
             final_state.update(updated_state) # Overwrite with new/updated info
             self.loaded_documents_state = final_state
        else:
            # Full load: only keep state for files actually processed in this run
            self.loaded_documents_state = updated_state

        self._save_state()
        logger.info(f"Document loading finished. Processed: {files_processed}, Added: {files_added}, Updated: {files_updated}, Skipped/Unchanged: {files_skipped}. Returning {len(documents_content)} documents.")

        return documents_content, documents_ids

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Returns the cached metadata for a given document ID."""
        logger.debug(f"Requesting metadata for doc_id: '{doc_id}'")
        return self._document_metadata.get(doc_id) 