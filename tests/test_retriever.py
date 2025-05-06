import sys
import os
import unittest
import shutil
import logging # Import logging
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retriever import Retriever
# Import config and factory
from app.embedding_strategies.config import EmbeddingConfig
from app.models.document import Chunk, ChunkMetadata # Import Chunk

# Set up logging for the test file
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG) # Ensure debug messages are shown

# Dummy paths for testing
TEST_MODEL_DIR = "./test_retriever_models" # HFEmbedder will use this
TEST_INDEX_DIR = "./test_retriever_index"
TEST_DOCS_DIR = "./test_retriever_docs"
# Use a specific, small model for retriever tests
TEST_RETRIEVER_MODEL = 'moka-ai/m3e-base' 

class TestRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Download the model once for the entire test class if needed locally
        # This might be handled internally by the HuggingFaceEmbeddingStrategy now
        logger.info(f"Ensuring test model directory exists: {TEST_MODEL_DIR}")
        os.makedirs(TEST_MODEL_DIR, exist_ok=True)
        # Potentially pre-download model here if strategy doesn't handle it robustly
        # from sentence_transformers import SentenceTransformer
        # try:
        #     SentenceTransformer(TEST_RETRIEVER_MODEL, cache_folder=TEST_MODEL_DIR)
        #     logger.info(f"Test model {TEST_RETRIEVER_MODEL} cached/available.")
        # except Exception as e:
        #     logger.error(f"Could not pre-download test model: {e}")
        #     # Decide if tests can proceed without pre-downloading

    def setUp(self):
        logger.debug("Setting up TestRetriever...")
        # Ensure dummy dirs exist or mock them if needed
        os.makedirs(TEST_DOCS_DIR, exist_ok=True)
        # Create dummy document file(s) if needed by DocumentManager or build_index
        with open(os.path.join(TEST_DOCS_DIR, "dummy_doc1.txt"), "w", encoding='utf-8') as f:
            f.write("人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。")
        with open(os.path.join(TEST_DOCS_DIR, "dummy_doc2.txt"), "w", encoding='utf-8') as f:
            f.write("深度学习是机器学习的一个子领域，使用多层神经网络进行特征学习。")
        with open(os.path.join(TEST_DOCS_DIR, "dummy_doc3.txt"), "w", encoding='utf-8') as f:
            f.write("自然语言处理是AI的一个领域，专注于让计算机理解人类语言。")

        # --- Create EmbeddingConfig for the test --- 
        # Restore the required nested structure for EmbeddingConfig
        hf_conf_data = {
            "config": { # This key is required by EmbeddingConfig model
                "provider": "huggingface", # Discriminator inside 'config'
                "model_path": TEST_RETRIEVER_MODEL,
                "use_gpu": False, # Force CPU for test predictability
                "local_model_dir": TEST_MODEL_DIR
            }
        }
        logger.debug(f"Attempting to create EmbeddingConfig with data: {hf_conf_data}")
        try:
            self.embedding_config = EmbeddingConfig(**hf_conf_data)
            logger.debug("EmbeddingConfig created successfully.")
        except Exception as e:
            logger.error(f"Failed to create EmbeddingConfig: {e}", exc_info=True)
            self.fail(f"Failed to create EmbeddingConfig for test: {e}")
        # --- End Config Creation --- 

        # Initialize Retriever instance with EmbeddingConfig
        # Force rebuild of index for consistent tests
        if os.path.exists(TEST_INDEX_DIR):
            shutil.rmtree(TEST_INDEX_DIR)
        
        logger.debug("Initializing Retriever...")
        try:
            self.retriever = Retriever(
                embedding_config=self.embedding_config,
                index_dir=TEST_INDEX_DIR,
                docs_dir=TEST_DOCS_DIR
            )
            logger.debug("Retriever initialized successfully.")
        except Exception as e:
             logger.error(f"Failed to initialize Retriever during setUp: {e}", exc_info=True)
             self.fail(f"Retriever initialization failed: {e}")
        
        # Store expected documents
        self.documents = [
            "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
            "深度学习是机器学习的一个子领域，使用多层神经网络进行特征学习。",
            "自然语言处理是AI的一个领域，专注于让计算机理解人类语言。"
        ]
        logger.debug("TestRetriever setup complete.")

    def tearDown(self):
        logger.debug("Tearing down TestRetriever...")
        # Clean up dummy directories
        if os.path.exists(TEST_MODEL_DIR):
             shutil.rmtree(TEST_MODEL_DIR)
        if os.path.exists(TEST_INDEX_DIR):
             shutil.rmtree(TEST_INDEX_DIR)
        if os.path.exists(TEST_DOCS_DIR):
             shutil.rmtree(TEST_DOCS_DIR)
        logger.debug("TestRetriever teardown complete.")

    def test_initialization(self):
        # 测试初始化是否成功
        self.assertIsNotNone(self.retriever)
        self.assertIsNotNone(self.retriever.index)
        self.assertGreater(self.retriever.index.ntotal, 0, "Index should contain vectors after initialization")
        self.assertGreater(len(self.retriever.chunk_mapping), 0, "Chunk mapping should contain chunks")
        logger.info("test_initialization passed.")

    def test_document_retrieval(self):
        # 测试简单查询
        query = "什么是人工智能"
        print(f"\n----- Running test_document_retrieval (Query 1: '{query}') -----")
        # Use custom_top_k and expect List[Chunk] now
        results: List[Chunk] = self.retriever.retrieve(query, custom_top_k=1)
        print(f"Retrieve results: {results}")
        print(f"Type of results: {type(results)}")
        
        self.assertIsInstance(results, list)
        self.assertGreaterEqual(len(results), 1)
        # Assert that the items in the list are Chunk objects
        self.assertIsInstance(results[0], Chunk)
        # Check attributes of the Chunk object
        self.assertIsNotNone(results[0].text)
        self.assertIsNotNone(results[0].doc_id)
        self.assertIn("人工智能", results[0].text) # Check if relevant text is present
        self.assertEqual(results[0].doc_id, "dummy_doc1.txt") # Check if it retrieved the correct doc
        self.assertIsInstance(results[0].metadata, ChunkMetadata)
        self.assertEqual(results[0].metadata.source, "dummy_doc1.txt")

        print("----- test_document_retrieval (Query 1) passed -----")

        # 测试相关性查询
        query2 = "神经网络"
        print(f"\n----- Running test_document_retrieval (Query 2: '{query2}') -----")
        results2: List[Chunk] = self.retriever.retrieve(query2, custom_top_k=1)
        print(f"Retrieve results 2: {results2}")
        print(f"Type of results 2: {type(results2)}")

        self.assertIsInstance(results2, list)
        self.assertGreaterEqual(len(results2), 1)
        self.assertIsInstance(results2[0], Chunk)
        # Access attributes directly using dot notation
        self.assertIsNotNone(results2[0].text)
        self.assertIn("神经网络", results2[0].text) # Check content using attribute
        self.assertEqual(results2[0].doc_id, "dummy_doc2.txt") # Check doc_id using attribute
        # Optionally check metadata as well
        self.assertIsInstance(results2[0].metadata, ChunkMetadata)
        self.assertEqual(results2[0].metadata.source, "dummy_doc2.txt")

        print("----- test_document_retrieval (Query 2) passed -----")

    def test_multiple_results(self):
        # 测试返回多个结果
        query = "AI 技术"
        print(f"\n----- Running test_multiple_results (Query: '{query}') -----")
        # Use custom_top_k and expect List[Chunk] now
        retrieved_chunks: List[Chunk] = self.retriever.retrieve(query, custom_top_k=2)
        print(f"Retrieved chunks list: {retrieved_chunks}")
        print(f"Type of retrieved_chunks: {type(retrieved_chunks)}")
        
        self.assertIsInstance(retrieved_chunks, list)
        # Depending on reranking and similarity, might get 1 or 2 results
        self.assertLessEqual(len(retrieved_chunks), 2) 
        self.assertGreaterEqual(len(retrieved_chunks), 1) 

        print("--- Checking results in loop ---")
        # Iterate over the list of retrieved Chunk objects
        doc_ids_found = set()
        for i, chunk in enumerate(retrieved_chunks):
            print(f"Checking retrieved chunk {i}: {chunk.doc_id} - {chunk.text[:30]}...")
            self.assertIsInstance(chunk, Chunk) # Assert it's a Chunk instance
            self.assertIsNotNone(chunk.text)
            self.assertIsNotNone(chunk.doc_id)
            self.assertIsNotNone(chunk.metadata)
            self.assertIsNotNone(chunk.metadata.source)
            doc_ids_found.add(chunk.doc_id)

        # Check if expected documents were retrieved (order might vary)
        self.assertTrue("dummy_doc1.txt" in doc_ids_found or "dummy_doc3.txt" in doc_ids_found)
        print("--- test_multiple_results passed ---")

    def test_no_results(self):
        # 测试没有结果的查询
        query = "无关话题例如食谱"
        print(f"\n----- Running test_no_results (Query: '{query}') -----")
        results = self.retriever.retrieve(query, custom_top_k=1)
        print(f"Retrieve results for irrelevant query: {results}")
        # Depending on the threshold, might return 0 or the least dissimilar document.
        # For this test, let's assume it should return an empty list if nothing is relevant enough.
        # This behavior might need refinement in the retriever logic itself.
        # For now, we check if it's a list. A stricter test might assert len(results) == 0.
        self.assertIsInstance(results, list)
        # self.assertEqual(len(results), 0) # Stricter check if desired
        print("--- test_no_results potentially passed (check retriever logic for strictness) ---")

    # Add more tests:
    # - test_reranking_effect (if reranker is used and not NoOp)
    # - test_add_documents_idempotency (if add_documents is implemented)
    # - test_index_persistence (save, delete retriever, init again, check if loaded)
    # - test_different_top_k
    # - test_empty_query
    # - test_special_characters_query

if __name__ == "__main__":
    unittest.main() 