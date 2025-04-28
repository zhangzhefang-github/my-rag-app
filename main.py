from app.rag_pipeline import RAGPipeline
from utils.env_helper import get_api_config, get_system_config, print_env_setup_guide
from utils.logger import setup_logger
from utils.document_manager import DocumentManager
from utils.gpu_manager import GPUManager
import os
import argparse
import logging

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RAG系统")
    
    # 系统参数
    parser.add_argument('--low-memory', action='store_true', 
                        help='启用低内存模式，优化2GB以下显存的GPU使用')
    
    # OpenAI相关参数
    parser.add_argument('--use-local-model', action='store_true',
                       help='使用本地模型而非OpenAI API')
    parser.add_argument('--openai-model', type=str,
                       help='指定使用的OpenAI模型，默认使用环境变量设置')
    parser.add_argument('--openai-api-key', type=str,
                       help='OpenAI API密钥，默认使用环境变量设置')
    parser.add_argument('--openai-base-url', type=str,
                       help='OpenAI API基础URL，默认使用环境变量设置')
    
    # 检索相关参数
    parser.add_argument('--embedding-model', type=str,
                       help='指定使用的嵌入模型，默认使用环境变量中的RETRIEVER_MODEL')
    parser.add_argument('--local-model-dir', type=str,
                       help='指定本地模型保存目录，默认使用环境变量中的LOCAL_MODEL_DIR')
    parser.add_argument('--top-k', type=int, default=3,
                       help='检索top-k个相关文档，默认3')
    
    # 工具参数
    parser.add_argument('--show-env-guide', action='store_true',
                       help='显示环境变量设置指南并退出')
    
    # 高级选项
    parser.add_argument('--no-gpu', action='store_true',
                       help='禁用GPU，即使GPU可用')
    parser.add_argument('--reload-index', action='store_true',
                       help='重新加载索引和文档')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='日志级别')
    parser.add_argument('--log-file', type=str, default='logs/rag_app.log',
                       help='日志文件路径')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level)
    logger = setup_logger(log_file=args.log_file, console_level=log_level)
    
    # 处理工具参数
    if args.show_env_guide:
        print_env_setup_guide()
        return
    
    # 检查GPU
    gpu_manager = GPUManager()
    if gpu_manager.gpu_available:
        gpu_manager.print_gpu_info()
    
    # 获取系统配置
    sys_config = get_system_config()
    
    # 命令行参数覆盖环境变量
    use_local_model = args.use_local_model or sys_config["use_local_model"]
    low_memory_mode = args.low_memory or sys_config["low_memory_mode"]
    use_gpu = not args.no_gpu
    
    # 配置OpenAI
    openai_config = None
    if not use_local_model:
        # 优先使用命令行参数，其次使用环境变量
        openai_config = get_api_config()
        
        # 命令行参数覆盖环境变量
        if args.openai_model:
            openai_config["model"] = args.openai_model
        if args.openai_api_key:
            openai_config["api_key"] = args.openai_api_key
        if args.openai_base_url:
            openai_config["base_url"] = args.openai_base_url
            
        logging.info(f"OpenAI配置: 模型={openai_config['model']}, API URL={openai_config['base_url']}")
    
    # 配置检索器
    retriever_config = {
        "model_name": args.embedding_model if args.embedding_model else sys_config["retriever_model"],
        "local_model_dir": args.local_model_dir if args.local_model_dir else sys_config["local_model_dir"],
        "use_gpu": use_gpu
    }
    
    logging.info(f"检索器配置: 模型={retriever_config['model_name']}, 使用GPU={retriever_config['use_gpu']}")
    
    # 初始化RAG，传入相关参数
    rag = RAGPipeline(
        low_memory_mode=low_memory_mode,
        use_openai=not use_local_model,
        openai_config=openai_config,
        retriever_config=retriever_config
    )
    
    # 初始化文档管理器
    doc_manager = DocumentManager()
    
    # 加载文档并添加到检索器
    try:
        documents, doc_ids = doc_manager.load_documents(incremental=not args.reload_index)
        if documents:
            rag.add_knowledge(documents, doc_ids)
            logging.info(f"成功加载并索引 {len(documents)} 个文档")
        else:
            logging.info("没有新文档需要索引")
    except Exception as e:
        logging.error(f"加载文档出错: {str(e)}")
    
    # 启动对话
    chat(rag, args.top_k)

# 简单命令行界面
def chat(rag, top_k=3):
    logging.info("\n=== RAG Demo 启动 ===")
    print("\n=== Welcome to RAG Demo ===")
    print("输入问题进行对话，输入'exit'或'quit'退出\n")
    print(f"将为每个问题检索 {top_k} 个最相关的文档片段")
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Bye!")
            break
        
        if not query.strip():
            continue
        
        try:
            answer = rag.answer_question(query, top_k=top_k)
            print(f"\nRAG: {answer}")
        except Exception as e:
            logging.error(f"处理查询出错: {str(e)}")
            print(f"\nRAG: 抱歉，处理您的问题时出现错误。请稍后再试。")

if __name__ == "__main__":
    main() 