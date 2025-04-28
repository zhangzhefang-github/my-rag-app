import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestRAGPipeline(unittest.TestCase):
    
    @patch('app.openai_generator.OpenAIGenerator')
    def test_rag_pipeline(self, mock_generator):
        # 导入在这里进行，以便可以正确应用patch
        from app.rag_pipeline import RAGPipeline
        
        # 设置模拟的生成器
        mock_generator_instance = MagicMock()
        mock_generator_instance.generate.return_value = "这是一个关于人工智能历史的测试回答。"
        mock_generator.return_value = mock_generator_instance
        
        # 创建RAG管道，使用CPU避免GPU依赖
        retriever_config = {
            "model_name": "moka-ai/m3e-base",
            "local_model_dir": "models"
        }
        
        # 使用模拟的OpenAI配置
        openai_config = {
            "api_key": "test_key",
            "model": "test_model",
            "base_url": "test_url"
        }
        
        # 初始化RAG系统
        rag = RAGPipeline(
            use_openai=True,
            openai_config=openai_config,
            retriever_config=retriever_config
        )
        
        # 添加测试文档
        documents = [
            "人工智能起源于20世纪50年代，1956年的达特茅斯会议被认为是AI领域的开端。",
            "深度学习是机器学习的一个分支，在2012年取得了重大突破。",
            "自然语言处理是AI的关键应用领域之一。"
        ]
        rag.add_knowledge(documents)
        
        # 测试问答
        query = "人工智能是什么时候开始的？"
        answer = rag.answer_question(query)
        
        # 验证结果
        self.assertEqual(answer, "这是一个关于人工智能历史的测试回答。")
        
        # 验证是否用正确的增强查询调用了生成器
        called_args = mock_generator_instance.generate.call_args[0][0]
        self.assertIn("背景信息", called_args)
        self.assertIn("达特茅斯会议", called_args)  # 应该检索到包含这个关键词的文档
        self.assertIn(query, called_args)  # 查询应该包含在增强查询中
        
        print("\n测试RAG管道成功！")
        print(f"用户查询: {query}")
        print(f"回答: {answer}")
        print(f"增强查询包含了正确的背景信息")

if __name__ == "__main__":
    unittest.main() 