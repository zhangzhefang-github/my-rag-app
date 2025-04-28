"""
RAG系统使用示例

此脚本展示了如何使用RAG系统进行简单的问答交互。
不需要.env文件，可以直接运行。
"""

import os
from app.rag_pipeline import RAGPipeline

def main():
    print("=== RAG系统示例 ===")
    
    # 为了示例目的，检查是否有OpenAI API密钥
    api_key = os.environ.get("OPENAI_API_KEY")
    use_openai = api_key is not None
    
    if use_openai:
        print(f"检测到OpenAI API密钥，将使用OpenAI进行生成")
        openai_config = {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "api_key": api_key,
            "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.fe8.cn/v1")
        }
        rag = RAGPipeline(
            use_openai=True, 
            openai_config=openai_config
        )
    else:
        print("未检测到OpenAI API密钥，将使用本地模型进行生成")
        print("注意：本地生成效果可能不如OpenAI，如需更好效果请设置OPENAI_API_KEY环境变量")
        rag = RAGPipeline(use_openai=False)
    
    # 加载测试文档
    print("\n加载文档...")
    docs_path = 'data/documents'
    
    # 检查是否有测试文档
    if not os.path.exists(docs_path) or len(os.listdir(docs_path)) == 0:
        print(f"警告: {docs_path} 目录不存在或为空")
        print("创建示例文档...")
        
        # 确保目录存在
        os.makedirs(docs_path, exist_ok=True)
        
        # 创建一个简单的测试文档
        with open(os.path.join(docs_path, "example.txt"), "w", encoding="utf-8") as f:
            f.write("""人工智能（AI）的发展历程

人工智能（Artificial Intelligence，简称AI）的概念最早可以追溯到20世纪50年代。1956年，在达特茅斯会议上，约翰·麦卡锡首次提出了"人工智能"这一术语。自那时起，人工智能领域经历了几次起伏。

早期的AI研究（1956-1974）主要集中在问题求解和符号方法上。研究人员开发了能够解决代数问题、证明逻辑定理和理解简单英语的程序。这个时期被称为AI的"黄金年代"。

真正的突破发生在21世纪初，随着机器学习技术，特别是深度学习的兴起。2006年，Geoffrey Hinton提出了深度学习的有效训练方法。2012年，他的团队在ImageNet图像识别挑战中取得了重大突破，标志着深度学习时代的到来。
""")
    
    # 加载文档
    documents = []
    for file in os.listdir(docs_path):
        if file.endswith('.txt'):
            with open(os.path.join(docs_path, file), 'r', encoding='utf-8') as f:
                documents.append(f.read())
    
    print(f"已加载 {len(documents)} 个文档")
    rag.add_knowledge(documents)
    
    # 交互式问答
    print("\n现在您可以向系统提问：")
    print("(输入'exit'或'quit'退出)")
    
    while True:
        query = input("\nQ: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        if not query.strip():
            continue
            
        answer = rag.answer_question(query)
        print(f"\nA: {answer}")
        
    print("\n谢谢使用!")

if __name__ == "__main__":
    main() 