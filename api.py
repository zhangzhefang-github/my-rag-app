from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks, Query, Body
from pydantic import BaseModel, Field
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
from pathlib import Path
import uuid
from typing import List, Optional, Any, AsyncGenerator, Dict
import time  # 添加time模块用于计算请求耗时
import re # Import re for regex
import numpy as np # Add numpy import

# 导入必要的 RAG 组件和配置工具
from app.rag_pipeline import RAGPipeline
from app.retriever import Retriever
from utils.env_helper import load_env_config, get_system_config
from utils.logger import setup_logger
from utils.document_manager import DocumentManager
from utils.gpu_manager import GPUManager
# from config import APP_PORT # 移除导入
# 导入 torch 用于检查 cuda (在 startup 中用到)
import torch 
from app.embedding_strategies.config import EmbeddingConfig, HuggingFaceConfig # Add necessary imports
from app.embedding_strategies.factory import get_embedder # Remove get_embedder_config

# Import the Citation model we defined
from app.models.document import Citation

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
    title="RAG API",
    description="检索增强生成API",
    version="1.0",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，也可以指定特定域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有HTTP头
    expose_headers=["*"],  # 暴露所有响应头
)

# 添加日志中间件，记录所有HTTP请求
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有HTTP请求的中间件"""
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]  # 为每个请求生成短ID
    
    # 记录请求开始
    path = request.url.path
    method = request.method
    client = request.client.host if request.client else "unknown"
    logger.info(f"[{request_id}] 开始处理请求: {method} {path} from {client}")
    
    # 处理请求
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 记录请求结束和状态
        logger.info(f"[{request_id}] 完成请求: {method} {path} - 状态:{response.status_code} - 耗时:{process_time:.4f}秒")
        
        # 对于错误响应，记录更多信息
        if response.status_code >= 400:
            logger.warning(f"[{request_id}] 请求返回错误: {method} {path} - 状态:{response.status_code}")
            
        return response
    except Exception as e:
        # 记录未捕获的异常
        process_time = time.time() - start_time
        logger.error(f"[{request_id}] 请求处理异常: {method} {path} - 耗时:{process_time:.4f}秒 - 错误:{str(e)}", exc_info=True)
        raise

# ---------------------------------------------------------------------------
# Startup 事件：初始化 RAG Pipeline
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global rag_pipeline_instance
    logger.info("Starting RAG Pipeline initialization...")
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info(f"环境变量: LOG_LEVEL={os.environ.get('LOG_LEVEL', '未设置')}")
    
    # 检查静态文件目录是否存在，不存在则创建
    static_dirs = ["public/static", "public/assets"]
    for dir_path in static_dirs:
        if not os.path.exists(dir_path):
            logger.warning(f"静态文件目录 '{dir_path}' 不存在，正在创建...")
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"已创建静态文件目录: {dir_path}")

    try:
        # 检查 GPU
        gpu_manager = GPUManager()
        if gpu_manager.gpu_available:
            gpu_manager.print_gpu_info()

        # 获取配置 (不再处理命令行参数，直接使用环境变量/默认值)
        sys_config = get_system_config()

        # 从配置确定设置
        # 移除 low_memory_mode 的访问
        # low_memory_mode = sys_config["low_memory_mode"]
        # 注意：FastAPI 环境下通常不方便传递 --no-gpu，
        # use_gpu 应主要由环境配置决定，或在部署时考虑。
        # 假设我们总是尝试使用GPU（如果可用且配置允许）。
        use_gpu = torch.cuda.is_available() and sys_config.get("retriever_use_gpu", True) # Make retriever_use_gpu optional in config

        # 可以保留一个通用日志，说明LLM配置将动态加载
        logger.info("LLM configuration will be loaded dynamically based on LLM_PROVIDER.")

        # --- Refactor Retriever Initialization ---
        # Get embedding model details from system config
        retriever_model_name = sys_config.get("retriever_model", "moka-ai/m3e-base") # Default HF model
        local_model_dir = sys_config.get("local_model_dir", "models")
        index_dir = sys_config.get("index_dir", "data/indexes")
        docs_dir = sys_config.get("docs_dir", "data/documents")

        # Assume HuggingFace provider based on typical model name format
        # TODO: Consider adding an explicit 'retriever_provider' to sys_config for clarity
        embedding_provider = "huggingface" # Default assumption

        logger.info(f"Embedding provider: {embedding_provider}")
        logger.info(f"Embedding model: {retriever_model_name}")
        logger.info(f"Retriever GPU usage enabled: {use_gpu}")
        logger.info(f"Local model directory: {local_model_dir}")
        logger.info(f"Index directory: {index_dir}")
        logger.info(f"Docs directory: {docs_dir}")


        # Create the specific config based on provider (only HF shown here)
        # In a real scenario, you might need logic to choose based on provider
        if embedding_provider == "huggingface":
            embedding_config_data = {
                "provider": "huggingface",
                "model_path": retriever_model_name,
                "use_gpu": use_gpu,
                "local_model_dir": local_model_dir
            }
        # Add elif blocks here for "openai", "ollama" if needed based on sys_config
        # elif embedding_provider == "openai":
        #     embedding_config_data = {...}
        # elif embedding_provider == "ollama":
        #     embedding_config_data = {...}
        else:
             raise ValueError(f"Unsupported retriever provider specified: {embedding_provider}")

        # Validate and create the EmbeddingConfig object
        embedding_config = EmbeddingConfig(config=embedding_config_data)

        # Initialize Retriever with EmbeddingConfig
        logger.info("Initializing Retriever...")
        retriever = Retriever(
            embedding_config=embedding_config, # Pass the config object
            index_dir=index_dir,
            docs_dir=docs_dir
            # Removed old arguments: model_name, use_gpu, local_model_dir
        )
        logger.info("Retriever initialized successfully.")
        # --- End Refactor ---


        # Initialize RAG Pipeline (remains the same)
        logger.info("Initializing RAGPipeline...")
        rag_pipeline_instance = RAGPipeline(
            retriever=retriever
        )
        logger.info("RAGPipeline initialized successfully.")

        # 加载文档 (不处理 --reload-index，默认增量加载)
        logger.info("Loading documents into RAG pipeline...")
        doc_manager = DocumentManager(docs_dir=docs_dir)
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
# 数据模型
# ---------------------------------------------------------------------------

# 查询相关模型
class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""
    query: str
    stream: bool = False
    top_k: Optional[int] = None # Allow overriding retriever's default K
    # Add other potential parameters like llm_config overrides, etc.
    # Example: llm_parameters: Optional[Dict[str, Any]] = None 

# --- 修改 QueryResponse 模型 --- 
class QueryResponse(BaseModel):
    """
    Defines the structured response for the /query endpoint.
    This is the FINAL structure we aim for after Sprint 2.
    """
    query: str = Field(..., description="The original query submitted by the user.")
    answer_text: str = Field(..., description="The generated answer text from the RAG pipeline.")
    citations: List[Citation] = Field(default_factory=list, description="List of structured citations supporting the answer.")
    success: bool = Field(..., description="Indicates if the query processing was successful.")
    error: Optional[str] = Field(default=None, description="Error message if processing failed.")
    # You might want to include other metadata, e.g., processing time
    # processing_time_ms: Optional[float] = Field(default=None, description="Total time taken to process the query in milliseconds.")

    class Config:
        # Example for Pydantic V2
        pass

# 会话管理模型
class Conversation(BaseModel):
    id: str = None
    title: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
class ConversationList(BaseModel):
    conversations: List[Conversation]
    
# 消息管理模型
class Message(BaseModel):
    id: str = None
    conversation_id: str
    content: str
    role: str  # 'user' 或 'assistant'
    created_at: Optional[datetime] = None
    
class MessageList(BaseModel):
    messages: List[Message]

# 内存存储 (简单实现，生产环境应替换为数据库)
conversations = {}
messages = {}

# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------

# --- 辅助函数：解析 LLM 原始输出 ---
def parse_llm_output(raw_output: str) -> tuple[str | None, str]:
    """Parses raw LLM output, extracting <think> block and main answer.
    
    Args:
        raw_output: The raw string output from the LLM.
        
    Returns:
        A tuple containing: (think_content, answer_content)
        think_content is None if no <think> block is found.
    """
    think_content = None
    answer_content = raw_output # Default to the full output
    
    # Use regex to find and extract <think> block (non-greedy)
    match = re.search(r"<think>(.*?)</think>", raw_output, flags=re.DOTALL)
    if match:
        think_content = match.group(1).strip() # Get content inside tags
        # Remove the think block and surrounding whitespace from the answer
        answer_content = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        # Optional: remove leading newline if present after removal
        answer_content = re.sub(r"^\s*\n", "", answer_content) 
        
    return think_content, answer_content
# --- 结束辅助函数 ---

# --- 非流式端点 (修改) ---
@app.post("/query", response_model=QueryResponse, summary="非流式查询")
async def handle_query(request: QueryRequest):
    """Receives a query, processes it through the RAG pipeline, and returns a structured answer.
    
    NOTE: For Sprint 1, the actual returned data might temporarily differ internally 
    and might not perfectly match QueryResponse (especially the 'citations' field) 
    until Sprint 2 is complete. However, we define the endpoint with the final model.
    """
    if rag_pipeline_instance is None:
        logger.error("RAG Pipeline not initialized.")
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG Pipeline not ready.")
        
    # Determine top_k: Use request's value if provided, otherwise use retriever's default
    top_k_to_use = request.top_k if request.top_k is not None else rag_pipeline_instance.retriever.top_k
    if top_k_to_use <= 0:
        raise HTTPException(status_code=400, detail="top_k must be a positive integer.")
        
    logger.info(f"Received query: '{request.query}', using top_k={top_k_to_use}")
    
    try:
        # In Sprint 1, this call will use the *temporarily adjusted* pipeline,
        # which might return a dict that needs adapting before fitting QueryResponse.
        # In Sprint 2, this call will return a dict directly matching QueryResponse.
        result = await rag_pipeline_instance.process_query(query=request.query, top_k=top_k_to_use)
        
        # --- Adaptation Layer (Remove/Simplify after Sprint 2) --- 
        # If result in Sprint 1 is {'answer': ..., 'sources': List[Dict]}, adapt it:
        if "citations" not in result and "sources" in result:
            logger.warning("API returning temporary structure (Sprint 1). Adapt 'sources' to empty 'citations' for now.")
            # Create a QueryResponse-like dict, but citations will be wrong/empty
            response_data = {
                "query": request.query,
                "answer_text": result.get("answer", "Error: Answer key missing"),
                "citations": [], # Cannot populate correctly yet
                "success": result.get("success", False),
                "error": result.get("error")
            }
            # Validate and return using the target model, even if citations are empty
            # This ensures the API contract is met structurally.
            return QueryResponse(**response_data)
        elif "citations" in result:
             # Assume result directly matches QueryResponse (Sprint 2 onwards)
             # Let FastAPI handle validation by returning the dict directly
             return result
        else:
            # Handle unexpected result structure
            logger.error(f"Unexpected result structure from pipeline: {result}")
            raise HTTPException(status_code=500, detail="Internal Server Error: Unexpected pipeline response.")
        # --- End Adaptation Layer ---
        
    except HTTPException as http_exc:
         # Re-raise HTTP exceptions directly
         raise http_exc
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        # Return a structured error response matching QueryResponse
        error_response = QueryResponse(
            query=request.query,
            answer_text="",
            citations=[],
            success=False,
            error=f"An internal error occurred: {str(e)}"
        )
        # Return with a 500 status code, but FastAPI expects the model instance
        # We need to raise HTTPException or return JSONResponse for status code control.
        # Let's return JSONResponse here to control status and body precisely.
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump(exclude_none=True) # Pydantic v2
            # content=error_response.dict(exclude_none=True) # Pydantic v1
        )

# --- SSE 流式端点 (修改 stream_generator 和 message_stream_generator) --- 
async def stream_generator(query: str, top_k: int):
    """异步生成器，用于产生 SSE 事件流。"""
    if rag_pipeline_instance is None:
        logger.error("Streaming query received, but RAG pipeline is not initialized.")
        # 在流中发送错误事件
        yield f"event: error\ndata: {json.dumps({'detail': 'RAG system is not ready.'})}\n\n"
        return

    try:
        logger.info(f"Received streaming query: '{query}' with top_k={top_k}")
        stream = rag_pipeline_instance.stream_answer_question(query, top_k=top_k)
        
        # Process the stream, expecting dicts {"type": ..., "content": ...}
        # The "content" for type "chunk" might contain the raw LLM output including <think>
        async for chunk_dict in stream:
            event_type = chunk_dict.get("type")
            content = chunk_dict.get("content")

            if event_type == "chunk" and isinstance(content, str):
                # --- 解析原始块内容 --- 
                think_content, answer_chunk = parse_llm_output(content)
                # --- 结束解析 --- 

                # Send think block if found (only once, typically at the start)
                if think_content:
                    yield f"event: think\ndata: {json.dumps({'content': think_content})}\n\n"
                
                # Send the actual answer chunk
                if answer_chunk: # Avoid sending empty data if only think block was present
                    yield f"data: {json.dumps({'token': answer_chunk})}\n\n"
            
            elif event_type == "sources" and isinstance(content, list):
                 # Convert scores to float before JSON serialization
                 serializable_content = []
                 for doc in content:
                     doc_copy = doc.copy() # Work on a copy
                     # Check for both score types and convert if they are float32
                     if 'score' in doc_copy and isinstance(doc_copy.get('score'), np.float32):
                         doc_copy['score'] = float(doc_copy['score'])
                     if 'rerank_score' in doc_copy and isinstance(doc_copy.get('rerank_score'), np.float32):
                         doc_copy['rerank_score'] = float(doc_copy['rerank_score'])
                     serializable_content.append(doc_copy)

                 yield f"event: sources\ndata: {json.dumps({'sources': serializable_content})}\n\n"
            
            elif event_type == "status" and isinstance(content, str):
                 yield f"event: status\ndata: {json.dumps({'status': content})}\n\n"
                 
            elif event_type == "error" and isinstance(content, str):
                 yield f"event: error\ndata: {json.dumps({'detail': content})}\n\n"
            
            else:
                 logger.warning(f"Unknown or malformed chunk in stream_generator: {chunk_dict}")
                 
            await asyncio.sleep(0.01)

        yield f"event: end\ndata: Stream ended\n\n"
        logger.info(f"Finished streaming for query: '{query}'")

    except Exception as e:
        logger.error(f"Error during streaming query '{query}': {e}", exc_info=True)
        # 在流中发送错误事件
        yield f"event: error\ndata: {json.dumps({'detail': f'Internal server error during stream: {e}'})}\n\n"

# 可以用 GET 或 POST，GET 更符合 SSE 语义，但查询可能过长
# 这里我们保留 POST，但请求体仅用于触发，实际参数来自路径或查询参数
# 或者，可以直接用 GET: @app.get("/query/stream") async def handle_stream_query(query: str = Query(...), top_k: int = Query(3)):
@app.post("/query/stream", summary="流式查询 (SSE)")
async def handle_stream_query(request: QueryRequest):
    """Receives a query and streams the RAG pipeline results back using SSE."""
    if rag_pipeline_instance is None:
        logger.error("RAG Pipeline not initialized for streaming.")
        raise HTTPException(status_code=503, detail="Service Unavailable: RAG Pipeline not ready.")
        
    top_k_to_use = request.top_k if request.top_k is not None else rag_pipeline_instance.retriever.top_k
    if top_k_to_use <= 0:
         raise HTTPException(status_code=400, detail="top_k must be a positive integer.")
         
    logger.info(f"Received streaming query: '{request.query}', using top_k={top_k_to_use}")
    
    # Return a StreamingResponse that uses the async generator
    return StreamingResponse(
        stream_generator(request.query, top_k_to_use),
        media_type="text/event-stream"
    )

# --- 会话管理接口 ---
@app.get("/conversations", response_model=ConversationList)
async def get_conversations():
    """获取所有会话列表"""
    logger.debug(f"获取会话列表, 当前会话数: {len(conversations)}")
    conv_list = list(conversations.values())
    sorted_conversations = sorted(conv_list, key=lambda x: x.get("updated_at", x.get("created_at", "")), reverse=True)
    return {"conversations": sorted_conversations}

@app.post("/conversations", response_model=Conversation)
async def create_conversation(conversation: Conversation):
    """创建新会话"""
    logger.info(f"创建新会话: '{conversation.title}'")
    conversation_id = str(uuid.uuid4())
    current_time = datetime.now()
    
    new_conversation = {
        "id": conversation_id,
        "title": conversation.title or "新会话",
        "created_at": current_time,
        "updated_at": current_time
    }
    
    conversations[conversation_id] = new_conversation
    messages[conversation_id] = []
    logger.info(f"会话创建成功, ID: {conversation_id}")
    
    return new_conversation

@app.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """获取单个会话详情"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="会话不存在")
        
    return conversations[conversation_id]

@app.put("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation(conversation_id: str, conversation: Conversation):
    """更新会话信息"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 更新会话属性
    current = conversations[conversation_id]
    current["title"] = conversation.title
    current["updated_at"] = datetime.now()
    
    conversations[conversation_id] = current
    return current

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """删除会话"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    # 删除会话和相关消息
    del conversations[conversation_id]
    if conversation_id in messages:
        del messages[conversation_id]
    
    return {"success": True}

# --- 消息管理接口 ---
@app.get("/conversations/{conversation_id}/messages", response_model=MessageList)
async def get_messages(conversation_id: str):
    """获取会话的消息历史"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    conversation_messages = messages.get(conversation_id, [])
    return {"messages": conversation_messages}

@app.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, message: Message):
    """发送消息并获取AI回复（流式）"""
    if conversation_id not in conversations:
        logger.warning(f"尝试发送消息到不存在的会话: {conversation_id}")
        raise HTTPException(status_code=404, detail="会话不存在")
    
    logger.info(f"收到用户消息: 会话={conversation_id}, 内容='{message.content[:30]}...'")
    
    # 保存用户消息
    message_id = str(uuid.uuid4())
    current_time = datetime.now()
    
    user_message = {
        "id": message_id,
        "conversation_id": conversation_id,
        "content": message.content,
        "role": "user",
        "created_at": current_time
    }
    
    if conversation_id not in messages:
        messages[conversation_id] = []
    
    messages[conversation_id].append(user_message)
    logger.debug(f"用户消息已保存, ID: {message_id}")
    
    # 更新会话的时间戳
    conversations[conversation_id]["updated_at"] = current_time
    
    # 返回流式AI回复
    logger.info(f"开始生成AI回复流: 会话={conversation_id}")
    return StreamingResponse(
        message_stream_generator(conversation_id, message.content),
        media_type="text/event-stream"
    )

# 修改流式生成器，添加更好的错误处理
async def message_stream_generator(conversation_id: str, query: str, top_k: int = 3):
    """异步生成器，用于产生带有会话上下文的SSE事件流"""
    if rag_pipeline_instance is None:
        logger.error("流式请求收到，但RAG管道未初始化")
        yield f"event: error\ndata: {json.dumps({'detail': 'RAG系统未准备好'})}\n\n"
        return

    try:
        logger.info(f"会话 {conversation_id} 开始处理查询: '{query[:50]}...'")
        
        # 创建AI消息
        message_id = str(uuid.uuid4())
        ai_message = {
            "id": message_id,
            "conversation_id": conversation_id,
            "content": "",
            "role": "assistant",
            "created_at": datetime.now()
        }
        
        logger.debug(f"创建AI回复消息框架, ID: {message_id}")
        
        # 添加到消息列表
        if conversation_id not in messages:
            messages[conversation_id] = []
        messages[conversation_id].append(ai_message)
        
        # 调用RAG pipeline的流式处理方法
        logger.debug(f"开始调用RAG pipeline获取流式答案")
        
        try:
            stream = rag_pipeline_instance.stream_answer_question(query, top_k=top_k)
            
            full_answer = "" # Only store the user-facing answer
            full_think = None
            sent_think = False # Flag to send think block only once
            start_time = time.time()
            token_count = 0
            
            async for chunk_dict in stream:
                # Validate chunk_dict is a dictionary
                if not isinstance(chunk_dict, dict):
                    logger.warning(f"Received non-dict item from RAG stream: {chunk_dict}")
                    continue 
                    
                event_type = chunk_dict.get("type")
                data_content = chunk_dict.get("data") # Renamed from 'content' to avoid confusion
                
                # --- Handle different event types --- 
                if event_type == "chunk" and isinstance(data_content, str):
                    token_count += 1
                    full_answer += data_content # Accumulate the answer
                    # Send token chunk to client
                    yield f"data: {json.dumps({'token': data_content})}\n\n"
                    
                # --- SPRINT 2: Remove handling for old 'sources' event ---
                # elif event_type == "sources" and isinstance(data_content, list):
                #      yield f"event: sources\ndata: {json.dumps({'sources': data_content})}\n\n"
                #      ai_message['sources'] = data_content 
                
                # --- SPRINT 2: Add handling for new 'citations' event ---
                elif event_type == "citations" and isinstance(data_content, list):
                     logger.debug(f"Received {len(data_content)} citations from RAG pipeline.")
                     # Assume data_content is already a list of serialized Citation dicts
                     yield f"event: citations\ndata: {json.dumps({'citations': data_content})}\n\n"
                     # Store citations in the message object for history
                     ai_message['citations'] = data_content 

                elif event_type == "error" and isinstance(data_content, str):
                    logger.error(f"Error received from RAG pipeline: {data_content}")
                    # Send error event to client
                    yield f"event: error\ndata: {json.dumps({'detail': data_content})}\n\n"
                    # Optionally append error to the displayed answer
                    if not full_answer.endswith("[ERROR]"):
                         full_answer += f"\n\n[ERROR: {data_content}]" 
                         # Send error as a data token as well? Maybe not needed if event is sent.
                         # yield f"data: {json.dumps({'token': f'\n\n[ERROR: {data_content}]'})}\n\n"

                elif event_type == "debug" and isinstance(data_content, dict):
                    logger.debug(f"Debug info received from RAG pipeline: {data_content}")
                    # Send debug event (client might ignore it)
                    yield f"event: debug\ndata: {json.dumps(data_content)}\n\n"
                    
                elif event_type == "final_answer" and isinstance(data_content, str):
                    # This might be yielded if RAG has no context
                    full_answer = data_content # Overwrite if this is the final intended answer
                    yield f"data: {json.dumps({'token': data_content})}\n\n"
                     
                # Remove the old else block that printed warnings for known types now
                # else:
                #      logger.warning(f"Unknown or malformed chunk in message_stream_generator: {chunk_dict}")

                # Update the in-memory message content as we go
                ai_message["content"] = full_answer 

                await asyncio.sleep(0.01) # Small sleep to prevent tight loop
                
        except Exception as e:
            # Catch errors during the streaming process itself (outside RAG pipeline)
            error_message = f"处理您的查询时出现错误: {str(e)}"
            logger.error(f"流式生成过程出错: {e}", exc_info=True)
            
            # 向客户端发送错误消息
            error_token_payload = {"token": f"\n\n{error_message}"}
            yield f"data: {json.dumps(error_token_payload)}\n\n"
            
            # 更新消息内容包含错误信息
            if not full_answer:
                full_answer = error_message
            else:
                full_answer += "\n\n" + error_message
            
            # 设置消息内容
            ai_message["content"] = full_answer
        
        # --- 在流结束后，清理可能残留的 <think> 标签再保存到内存 --- 
        final_clean_answer, _ = parse_llm_output(full_answer)
        if conversation_id in messages:
            # Find the specific message and update its content
            for msg in messages[conversation_id]:
                 if msg.get("id") == message_id:
                      msg["content"] = final_clean_answer
                      logger.debug(f"Saved cleaned final answer to message {message_id}")
                      break
        # --- 结束清理 --- 

        # 流结束记录
        elapsed = time.time() - start_time
        logger.info(f"流式生成完成: 会话={conversation_id}, 总块数:{token_count}, 总用时{elapsed:.2f}秒")
        
        # 流结束事件
        yield f"event: end\ndata: {json.dumps({'detail': '流式生成完成', 'chunks': token_count, 'time': elapsed})}\n\n"
        
        # 更新会话标题（如果是新会话且没有自定义标题）
        if conversations[conversation_id]["title"] == "新会话":
            # 使用用户查询的前20个字符作为标题
            title = query[:20] + ("..." if len(query) > 20 else "")
            conversations[conversation_id]["title"] = title
            logger.debug(f"更新会话标题: '{title}'")
            
    except Exception as e:
        # 捕获整个流程中的错误
        logger.error(f"会话 {conversation_id} 流式生成错误: {e}", exc_info=True)
        
        # 发送错误事件
        error_message = f"内部服务器错误: {str(e)}"
        error_token_payload = {"token": f"\n\n{error_message}"}
        yield f"data: {json.dumps(error_token_payload)}\n\n"
        yield f"event: error\ndata: {json.dumps({'detail': error_message})}\n\n"
        
        # 如果消息已经创建但内容为空，设置错误消息
        if conversation_id in messages:
            for msg in messages[conversation_id]:
                if msg.get("id") == message_id and not msg.get("content"):
                    msg["content"] = f"处理查询时发生错误: {str(e)}"

# --- 文档重载接口 ---
@app.post("/reload")
async def reload_documents():
    """手动触发文档重新加载的端点"""
    global rag_pipeline_instance
    if rag_pipeline_instance is None:
        raise HTTPException(status_code=503, detail="RAG系统未准备好")
    
    try:
        logger.info("手动重新加载文档...")
        sys_config = get_system_config()
        doc_manager = DocumentManager(docs_dir=sys_config['docs_dir'])
        documents, doc_ids = doc_manager.load_documents(incremental=False)  # 全量重载
        
        if documents:
            rag_pipeline_instance.add_knowledge(documents, doc_ids)
            logger.info(f"成功添加 {len(documents)} 个文档")
            return {"success": True, "doc_count": len(documents)}
        else:
            logger.info("未找到文档")
            return {"success": True, "doc_count": 0}
            
    except Exception as e:
        logger.error(f"重载文档错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"重载文档错误: {str(e)}")

# 健康检查端点 - 添加更多日志
@app.get("/health")
async def health():
    """健康检查端点，用于前端检测API服务器状态"""
    logger.debug("健康检查请求")
    status = "ok" if rag_pipeline_instance is not None else "initializing"
    return {
        "status": status, 
        "timestamp": datetime.now().isoformat(),
        "rag_ready": rag_pipeline_instance is not None,
        "version": "1.0",
        "uptime": time.time() - startup_time
    }

# 记录启动时间
startup_time = time.time()

# 根路径，提供基本信息
@app.get("/")
async def read_root():
    """根路径，提供基本信息。"""
    return {"message": "Welcome to the RAG Demo API. Use the /query endpoint to ask questions."}

# 挂载静态文件 - 确保放在文件末尾所有路由定义之后
@app.get("/ui")
async def serve_ui():
    """提供UI页面"""
    with open("public/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# 静态资源必须在所有API路由之后挂载
app.mount("/static", StaticFiles(directory="public/static", html=False), name="static")
app.mount("/assets", StaticFiles(directory="public/assets", html=False), name="assets")
# 不要在这里挂载根路径"/"，这会覆盖所有API路由

# 启动代码
if __name__ == "__main__":
    # --- 直接在这里读取 APP_PORT --- 
    load_dotenv() # 确保 .env 被加载
    app_port = int(os.getenv("APP_PORT", 8000))
    # --- 结束读取 --- 
    logger.info(f"启动Uvicorn服务器: 主机=0.0.0.0, 端口={app_port}")
    # 在startup前记录目录信息
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"检查public目录: {'存在' if os.path.exists('public') else '不存在'}")
    # 确保静态文件目录存在
    os.makedirs("public/static", exist_ok=True)
    os.makedirs("public/assets", exist_ok=True)
    logger.info("已确保静态文件目录存在")
    # 启动服务器
    uvicorn.run("api:app", host="0.0.0.0", port=app_port, reload=True)