import sys
import os
import unittest
import shutil

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retriever import Retriever
# Import config and factory
from app.embedding_strategies.config import EmbeddingConfig

# Dummy paths for testing
TEST_MODEL_DIR = "./test_retriever_models" # HFEmbedder will use this
TEST_INDEX_DIR = "./test_retriever_index"
TEST_DOCS_DIR = "./test_retriever_docs"
# Use a specific, small model for retriever tests
TEST_RETRIEVER_MODEL = 'moka-ai/m3e-base' 

class TestRetriever(unittest.TestCase):

    def setUp(self):
        # Ensure dummy dirs exist or mock them if needed
        os.makedirs(TEST_DOCS_DIR, exist_ok=True)
        # Create dummy document file(s) if needed by DocumentManager or build_index
        with open(os.path.join(TEST_DOCS_DIR, "dummy_doc.txt"), "w") as f:
            f.write("This is a dummy document for testing.")

        # --- Create EmbeddingConfig for the test --- 
        hf_conf_data = {
            "config": {
                "provider": "huggingface",
                "model_path": TEST_RETRIEVER_MODEL,
                "use_gpu": False, # Force CPU for test predictability
                "local_model_dir": TEST_MODEL_DIR
            }
        }
        try:
            self.embedding_config = EmbeddingConfig(**hf_conf_data)
        except Exception as e:
            self.fail(f"Failed to create EmbeddingConfig for test: {e}")
        # --- End Config Creation --- 

        # Initialize Retriever instance with EmbeddingConfig
        self.retriever = Retriever(
            embedding_config=self.embedding_config,
            index_dir=TEST_INDEX_DIR,
            docs_dir=TEST_DOCS_DIR
        )
        
        # 测试文档
        self.documents = [
            "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
            "深度学习是机器学习的一个子领域，使用多层神经网络进行特征学习。",
            "自然语言处理是AI的一个领域，专注于让计算机理解人类语言。"
        ]

        # Add documents AFTER retriever initialization
        try:
             # This will now use the embedder initialized within the retriever
             self.retriever.add_documents(self.documents)
        except Exception as e:
             # If add_documents fails (e.g., model download issues), fail the setup
             self.fail(f"Failed to add documents during setUp: {e}")

    def tearDown(self):
        # Clean up dummy directories
        if os.path.exists(TEST_MODEL_DIR):
             shutil.rmtree(TEST_MODEL_DIR)
        if os.path.exists(TEST_INDEX_DIR):
             shutil.rmtree(TEST_INDEX_DIR)
        if os.path.exists(TEST_DOCS_DIR):
             shutil.rmtree(TEST_DOCS_DIR)

    def test_document_retrieval(self):
        # 测试简单查询
        query = "什么是人工智能"
        print(f"\n----- Running test_document_retrieval (Query 1: '{query}') -----")
        results = self.retriever.retrieve(query, top_k=1)
        print(f"Retrieve results: {results}")
        print(f"Type of results: {type(results)}")
        if results:
            print(f"Type of results[0]: {type(results[0])}")
            try:
                print(f"Type of results[0][0]: {type(results[0][0])}") 
                print(f"Value of results[0][0]: {repr(results[0][0])}") # Use repr to see hidden chars
            except IndexError:
                 print("results[0] does not have index 0")
        print(f"Expected document (self.documents[0]): {repr(self.documents[0])}")
        print(f"Type of self.documents[0]: {type(self.documents[0])}")
        
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0][0], self.documents[0]) 
        
        # 测试相关性查询
        query = "神经网络"
        print(f"\n----- Running test_document_retrieval (Query 2: '{query}') -----")
        results = self.retriever.retrieve(query, top_k=1)
        print(f"Retrieve results: {results}")
        print(f"Type of results: {type(results)}")
        if results:
            print(f"Type of results[0]: {type(results[0])}")
            try:
                print(f"Type of results[0][0]: {type(results[0][0])}") 
                print(f"Value of results[0][0]: {repr(results[0][0])}") # Use repr to see hidden chars
            except IndexError:
                 print("results[0] does not have index 0")
        print(f"Expected document (self.documents[1]): {repr(self.documents[1])}")
        print(f"Type of self.documents[1]: {type(self.documents[1])}")
        
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0][0], self.documents[1]) 
    
    def test_multiple_results(self):
        # 测试返回多个结果
        query = "AI 技术"
        print(f"\n----- Running test_multiple_results (Query: '{query}') -----")
        results_tuple = self.retriever.retrieve(query, top_k=2)
        print(f"Retrieve results tuple: {results_tuple}")
        print(f"Type of results_tuple: {type(results_tuple)}")
        
        # Extract the list of documents from the tuple
        if not isinstance(results_tuple, tuple) or len(results_tuple) < 1:
            self.fail(f"Retrieve did not return the expected tuple structure. Got: {results_tuple}")
        
        retrieved_docs = results_tuple[0]
        print(f"Retrieved documents list: {retrieved_docs}")
        print(f"Type of retrieved_docs: {type(retrieved_docs)}")
        print(f"Expected documents (self.documents): {self.documents}")
        
        self.assertLessEqual(len(retrieved_docs), 2)
        self.assertGreaterEqual(len(retrieved_docs), 1) 

        print("--- Checking results in loop ---")
        # Iterate only over the list of retrieved documents
        for i, doc in enumerate(retrieved_docs): 
            print(f"Checking retrieved doc {i}: {repr(doc)}")
            print(f"Type of retrieved doc {i}: {type(doc)}")
            # Now doc should be a string, directly check if it's in the original list
            self.assertIn(doc, self.documents)
            
        print("--- Loop finished ---")

if __name__ == "__main__":
    unittest.main() 