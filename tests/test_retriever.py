import sys
import os
import unittest

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retriever import Retriever

class TestRetriever(unittest.TestCase):
    
    def setUp(self):
        # 初始化Retriever实例，强制使用CPU以避免GPU依赖
        self.retriever = Retriever(model_name='moka-ai/m3e-base', use_gpu=False)
        
        # 测试文档
        self.documents = [
            "人工智能是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。",
            "深度学习是机器学习的一个子领域，使用多层神经网络进行特征学习。",
            "自然语言处理是AI的一个领域，专注于让计算机理解人类语言。"
        ]
        
        # 将文档添加到检索器
        self.retriever.add_documents(self.documents)
    
    def test_document_retrieval(self):
        # 测试简单查询
        query = "什么是人工智能"
        results = self.retriever.retrieve(query, top_k=1)
        
        # 应该返回第一个文档（关于人工智能的）
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.documents[0])
        
        # 测试相关性查询
        query = "神经网络"
        results = self.retriever.retrieve(query, top_k=1)
        
        # 应该返回第二个文档（关于深度学习的）
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], self.documents[1])
    
    def test_multiple_results(self):
        # 测试返回多个结果
        query = "AI 技术"
        results = self.retriever.retrieve(query, top_k=2)
        
        # 应该返回两个结果
        self.assertEqual(len(results), 2)
        # 结果应该是我们添加的文档
        for result in results:
            self.assertIn(result, self.documents)

if __name__ == "__main__":
    unittest.main() 