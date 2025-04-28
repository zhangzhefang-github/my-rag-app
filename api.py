from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
import logging
import os
import json
import asyncio
# 导入 StreamingResponse
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# 导入必要的 RAG 组件和配置工具
from app.rag_pipeline import RAGPipeline
from app.retriever import Retriever
from utils.env_helper import load_env_config, get_api_config, get_system_config
from utils.logger import setup_logger
from utils.document_manager import DocumentManager
from utils.gpu_manager import GPUManager
from config import APP_PORT # 从 config.py 导入端口号
# 导入 torch 用于检查 cuda (在 startup 中用到)
import torch 

# ---------------------------------------------------------------------------
# 初始化：加载配置、设置日志、准备 RAG 实例
# ---------------------------------------------------------------------------

# 1. 加载环境变量 (应在访问 os.environ 之前完成)
load_env_config()

# 2. 设置日志
#    注意: 日志级别可以从环境变量 LOG_LEVEL 读取，或在这里硬编码
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
log_file = os.environ.get("LOG_FILE", "logs/rag_api.log") # 使用不同的日志文件
logger = setup_logger(log_file=log_file, console_level=log_level)

# 3. 全局 RAG pipeline 实例 (将在 startup 事件中初始化)
rag_pipeline_instance: RAGPipeline = None

# ---------------------------------------------------------------------------
# FastAPI 应用定义
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Demo API",
    description="API for interacting with the Retrieval Augmented Generation system.",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，也可以指定特定域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
)

# ---------------------------------------------------------------------------
# Startup 事件：初始化 RAG Pipeline
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global rag_pipeline_instance
    logger.info("Starting RAG Pipeline initialization...")

    try:
        # 检查 GPU
        gpu_manager = GPUManager()
        if gpu_manager.gpu_available:
            gpu_manager.print_gpu_info()

        # 获取配置 (不再处理命令行参数，直接使用环境变量/默认值)
        sys_config = get_system_config()
        api_config = get_api_config()

        # 从配置确定设置
        use_local_model = sys_config["use_local_model"]
        low_memory_mode = sys_config["low_memory_mode"]
        # 注意：FastAPI 环境下通常不方便传递 --no-gpu，
        # use_gpu 应主要由环境配置决定，或在部署时考虑。
        # 假设我们总是尝试使用GPU（如果可用且配置允许）。
        use_gpu = torch.cuda.is_available() # 简化处理，直接看 torch

        openai_config = None
        if not use_local_model:
            openai_config = api_config
            logger.info(f"OpenAI config loaded: model={openai_config.get('model')}")
        else:
            logger.info("OpenAI usage disabled by configuration (USE_LOCAL_MODEL=true).")

        # 配置检索器
        retriever_config = {
            "model_name": sys_config["retriever_model"],
            "local_model_dir": sys_config["local_model_dir"],
            "use_gpu": use_gpu,
            "index_dir": sys_config["index_dir"],
            "docs_dir": sys_config["docs_dir"]
        }
        logger.debug(f"Retriever configuration: {retriever_config}")
        logger.info(f"Embedding model: {retriever_config['model_name']}")
        logger.info(f"Retriever GPU usage enabled: {retriever_config['use_gpu']}")

        # 初始化 Retriever
        logger.info("Initializing Retriever...")
        retriever = Retriever(
            model_name=retriever_config['model_name'],
            use_gpu=retriever_config['use_gpu'],
            local_model_dir=retriever_config['local_model_dir'],
            index_dir=retriever_config['index_dir'],
            docs_dir=retriever_config['docs_dir']
        )
        logger.info("Retriever initialized successfully.")

        # 初始化 RAG Pipeline
        logger.info("Initializing RAGPipeline...")
        rag_pipeline_instance = RAGPipeline(
            retriever=retriever,
            low_memory_mode=low_memory_mode,
            use_openai=not use_local_model,
            openai_config=openai_config
        )
        logger.info("RAGPipeline initialized successfully.")

        # 加载文档 (不处理 --reload-index，默认增量加载)
        logger.info("Loading documents into RAG pipeline...")
        doc_manager = DocumentManager(docs_dir=retriever_config['docs_dir'])
        documents, doc_ids = doc_manager.load_documents(incremental=True)
        if documents:
            rag_pipeline_instance.add_knowledge(documents, doc_ids)
            logger.info(f"Requested RAG Pipeline to add {len(documents)} documents.")
        else:
            logger.info("No new documents found to add.")

        logger.info("RAG Pipeline initialization complete.")

    except Exception as e:
        logger.error(f"FATAL: RAG Pipeline initialization failed: {e}", exc_info=True)
        # 如果初始化失败，API 将无法正常工作
        # 可以考虑让应用退出或保持运行但返回错误
        rag_pipeline_instance = None # 确保实例为 None

# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------

# --- 非流式端点 (保持不变) ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    success: bool
    error: str | None = None

@app.post("/query", response_model=QueryResponse, summary="非流式查询")
async def handle_query(request: QueryRequest):
    """接收用户查询，通过 RAG 系统处理并一次性返回完整答案和来源。"""
    if rag_pipeline_instance is None:
        logger.error("Query received, but RAG pipeline is not initialized.")
        raise HTTPException(status_code=503, detail="RAG system is not ready. Initialization failed.")

    try:
        logger.info(f"Received non-streaming query: '{request.query}' with top_k={request.top_k}")
        # 调用 RAG pipeline 处理查询 (非流式方法)
        result_dict = await rag_pipeline_instance.answer_question(request.query, top_k=request.top_k)

        if result_dict and result_dict.get("success"):
            logger.info(f"Successfully processed non-streaming query: '{request.query}'")
            return QueryResponse(
                answer=result_dict.get("answer", ""),
                sources=result_dict.get("sources", []),
                success=True,
                error=None
            )
        else:
            logger.error(f"Failed to process non-streaming query: '{request.query}'. Reason: {result_dict.get('error')}")
            # 返回包含错误信息的响应
            return QueryResponse(
                answer=result_dict.get("answer", "处理查询时出错。请查看服务器日志。") ,
                sources=[],
                success=False,
                error=result_dict.get("error", "Unknown error during query processing.")
            )

    except Exception as e:
        logger.error(f"Error handling non-streaming query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error while processing query.")

# --- SSE 流式端点 --- 
async def stream_generator(query: str, top_k: int):
    """异步生成器，用于产生 SSE 事件流。"""
    if rag_pipeline_instance is None:
        logger.error("Streaming query received, but RAG pipeline is not initialized.")
        # 在流中发送错误事件
        yield f"event: error\ndata: {json.dumps({'detail': 'RAG system is not ready.'})}\n\n"
        return

    try:
        logger.info(f"Received streaming query: '{query}' with top_k={top_k}")
        
        # 调用 RAG pipeline 的流式处理方法
        stream = rag_pipeline_instance.stream_answer_question(query, top_k=top_k)
        
        token_count = 0
        async for chunk in stream:
            token_count += 1
            # 按照 SSE 格式 yield 数据:
            # data: ...\n\n
            # 可以选择添加 event 类型，例如 event: message
            # 确保数据是 JSON 格式或纯文本，客户端需要能解析
            # 这里我们简单地发送文本块
            yield f"data: {json.dumps(chunk)}\n\n" # 发送 JSON 编码的字符串块
            await asyncio.sleep(0.01) # 短暂暂停，避免潜在的阻塞问题

        # 流结束时可以发送一个完成事件 (可选)
        yield f"event: end\ndata: Stream ended. Total tokens: {token_count}\n\n"
        logger.info(f"Finished streaming {token_count} chunks for query: '{query}'")

    except Exception as e:
        logger.error(f"Error during streaming query '{query}': {e}", exc_info=True)
        # 在流中发送错误事件
        yield f"event: error\ndata: {json.dumps({'detail': f'Internal server error during stream: {e}'})}\n\n"

# 可以用 GET 或 POST，GET 更符合 SSE 语义，但查询可能过长
# 这里我们保留 POST，但请求体仅用于触发，实际参数来自路径或查询参数
# 或者，可以直接用 GET: @app.get("/query/stream") async def handle_stream_query(query: str = Query(...), top_k: int = Query(3)):
@app.post("/query/stream", summary="流式查询 (SSE)")
async def handle_stream_query(request: QueryRequest):
    """接收用户查询，通过 RAG 系统处理并通过 SSE 流式返回答案。"""
    # 返回 StreamingResponse
    return StreamingResponse(stream_generator(request.query, request.top_k), media_type="text/event-stream")

@app.get("/")
async def read_root():
    """根路径，提供基本信息。"""
    return {"message": "Welcome to the RAG Demo API. Use the /query endpoint to ask questions."}

# 挂载静态文件
app.mount("/", StaticFiles(directory="public", html=True), name="public")

# 健康检查端点
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# ---------------------------------------------------------------------------
# Uvicorn 启动入口 (如果直接运行 python app.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on host 0.0.0.0, port {APP_PORT}")
    # 注意：直接运行此脚本启动 uvicorn 可能不利于生产环境管理
    # 推荐使用 uvicorn app:app --host ... 命令启动
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)