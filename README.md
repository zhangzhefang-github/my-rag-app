# RAG Demo - 检索增强生成系统

<div id="language-switcher">
  <em><a href="#english-section">English</a> | <a href="#chinese-section">中文</a></em>
</div>

<div id="english-section"></div>

# RAG Demo - Retrieval Augmented Generation System

## Overview

A simple Retrieval Augmented Generation (RAG) system built with FastAPI. It combines efficient semantic retrieval using Sentence Transformers and FAISS with pluggable Large Language Model (LLM) backends for answering questions based on your documents.

## Project Structure

```
my-rag-app/
├── README.md
├── requirements.txt             # Python dependencies
├── .env                         # Environment configuration (create this file)
├── api.py                       # FastAPI application entry point
├── config.py                    # Basic configuration loading
├── app/                         # Core application logic
│   ├── __init__.py
│   ├── retriever.py             # Document embedding, indexing, and retrieval (Faiss + SentenceTransformer)
│   ├── rag_pipeline.py          # RAG pipeline orchestrating retrieval and generation
│   └── llm_service.py           # Service layer calling the selected LLM strategy
│   └── llm_strategies/          # Pluggable LLM backend strategies
│       ├── __init__.py          # Factory function (get_llm_strategy)
│       ├── base.py              # Abstract base class for LLM strategies
│       ├── ollama_strategy.py   # Strategy for local Ollama models via LangChain
│       ├── openai_strategy.py   # Strategy for OpenAI API via LangChain
│       └── custom_api_strategy.py # Strategy for custom OpenAI-compatible APIs using the openai library
├── data/
│   ├── documents/               # Place your source documents (.txt) here
│   └── indexes/                 # Stores Faiss index files and document mappings
├── models/                      # Local Sentence Transformer model cache directory
├── logs/                        # Log files directory
├── public/                      # Static files for the frontend UI
│   ├── index.html
│   └── ... (assets, static js/css)
└── utils/                       # Utility modules
    ├── __init__.py
    ├── env_helper.py            # Environment variable loading and parsing
    ├── gpu_manager.py           # GPU detection and info
    ├── document_manager.py      # Document loading and management
    ├── model_utils.py           # Model loading utilities (ModelManager)
    └── logger.py                # Logging setup
```

## Installation

First create and activate a conda environment (or use your preferred virtual environment):

```bash
conda create -n rag-app python=3.10 # Or your preferred Python version
conda activate rag-app
```

Install all dependencies:

```bash
pip install -r requirements.txt
```
Key dependencies include: `fastapi`, `uvicorn`, `langchain`, `langchain-ollama`, `langchain-openai`, `openai`, `sentence-transformers`, `faiss-cpu` (or `faiss-gpu`), `python-dotenv`.

## Configuration (.env file)

Create a `.env` file in the project root directory (`my-rag-app/`) to configure the application. Use the following template:

```dotenv
# --- LLM Provider Selection ---
# Select the backend LLM service. Options: 'ollama', 'custom_api', 'openai'
LLM_PROVIDER=ollama # Or custom_api, openai

# --- Ollama Configuration (if LLM_PROVIDER=ollama) ---
OLLAMA_HOST=http://localhost:11434       # Your Ollama service URL
OLLAMA_MODEL=qwen3:0.6b                 # Model name available in Ollama (e.g., run 'ollama list')
# OLLAMA_TEMPERATURE=0.7                 # (Optional) Sampling temperature
# OLLAMA_NUM_PREDICT=256                # (Optional) Max tokens for Ollama

# --- Custom API Configuration (if LLM_PROVIDER=custom_api) ---
CUSTOM_API_KEY="sk-your_custom_key"       # API Key for your custom endpoint
CUSTOM_API_BASE="https://api.fe8.cn/v1"   # Base URL for your custom endpoint
CUSTOM_API_MODEL="gpt-4o-mini"            # Model name available at the custom endpoint
# CUSTOM_API_TEMPERATURE=0.7             # (Optional)
# CUSTOM_API_MAX_TOKENS=1024              # (Optional)

# --- OpenAI Configuration (if LLM_PROVIDER=openai) ---
# Uses the official OpenAI API via LangChain.
# Requires OPENAI_API_KEY to be set. Base URL is optional.
OPENAI_API_KEY="sk-your_openai_key"       # Your official OpenAI API Key
# OPENAI_API_BASE="https://api.openai.com/v1" # (Optional) Use if you have a proxy
OPENAI_MODEL="gpt-4o-mini"                # Official OpenAI model name
# OPENAI_TEMPERATURE=0.7                 # (Optional)
# OPENAI_MAX_TOKENS=1024                  # (Optional)

# --- Retrieval Settings ---
TOP_K=3                               # Number of document chunks to retrieve per query
RETRIEVER_MODEL="moka-ai/m3e-base"    # Sentence Transformer model for embeddings
LOCAL_MODEL_DIR="models"              # Directory to cache embedding models
INDEX_DIR="data/indexes"              # Directory to store Faiss index and doc mappings
DOCS_DIR="data/documents"             # Directory containing source documents

# --- System Settings ---
LOG_LEVEL="INFO"                      # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# LOW_MEMORY_MODE=false              # (Currently less relevant as local generation is removed)
# APP_PORT=8000                      # Port is now read from config.py, but can be set here too
```

**Important**:
*   Set `LLM_PROVIDER` to choose your desired backend.
*   Fill in the corresponding configuration variables for the selected provider (API keys, URLs, model names).
*   Ensure the model name you specify for Ollama or Custom API is actually available at that service endpoint.
*   Make sure the `DOCS_DIR` and `INDEX_DIR` paths are correct.

## Features

*   **Retrieval Augmented Generation (RAG):** Answers questions based on provided documents.
*   **FastAPI Backend:** Provides a robust API interface.
*   **Efficient Semantic Search:** Uses Sentence Transformers (`moka-ai/m3e-base` default) and FAISS for finding relevant document chunks.
*   **Pluggable LLM Backends:** Easily switch between different LLM providers (Ollama, OpenAI, Custom OpenAI-compatible API) via configuration using a Strategy Pattern (`app/llm_strategies`).
*   **Streaming API:** Supports Server-Sent Events (SSE) for real-time answer generation (if the selected LLM strategy supports streaming).
*   **GPU Support:** Automatically utilizes GPU for Sentence Transformer embeddings and FAISS indexing (if `faiss-gpu` is installed and GPU is available).
*   **Simple Web UI:** A basic frontend is included for interaction (served from `/public`).

### GPU Acceleration

#### Installing CUDA and PyTorch
Ensure you have compatible CUDA and cuDNN versions installed. Install the correct PyTorch version:

```bash
# Example for CUDA 11.8
# pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Example for CUDA 12.1
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```
*(Check PyTorch website for the correct command for your CUDA version)*

#### FAISS GPU Support
For GPU-accelerated FAISS indexing (recommended for large datasets):

```bash
pip uninstall -y faiss-cpu
pip install faiss-gpu>=1.7.4
```

## Usage

### 1. Prepare Documents

Place your knowledge documents (plain text `.txt` files) in the `data/documents/` directory.

### 2. Configure Backend

Edit the `.env` file in the project root:
*   Set `LLM_PROVIDER` to `ollama`, `custom_api`, or `openai`.
*   Fill in the required API keys, URLs, and model names for your chosen provider.
*   Adjust `RETRIEVER_MODEL`, `TOP_K`, etc. if needed.

### 3. Run the API Server

Open your terminal in the project root (`my-rag-app/`) and run:

```bash
# Ensure your conda/virtual environment is activated
# Make sure OLLAMA_HOST env var is unset or set correctly if using Ollama
# unset OLLAMA_HOST # (If you suspect a shell variable overrides .env)

# Start the FastAPI server using Uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
*(The port (8000) is default, check `config.py` or set `APP_PORT` in `.env`)*

The server will start, load the embedding model, build/load the FAISS index for documents in `data/documents/`, and initialize the selected LLM strategy.

### 4. Interact

*   **Web UI:** Open your browser and navigate to `http://localhost:8000/ui`.
*   **API Docs:** Access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`. You can test API endpoints here.
*   **API Endpoints:**
    *   `/query` (POST): Non-streaming query.
    *   `/query/stream` (POST): Streaming query via SSE.
    *   `/conversations/.../messages` (POST): Streaming query within a conversation context via SSE.
    *   See `/docs` for more details on all endpoints.

---

<div id="chinese-section"></div>

# RAG Demo - 检索增强生成系统

## 概述

一个基于 FastAPI 构建的简单检索增强生成 (RAG) 系统。它结合了使用 Sentence Transformers 和 FAISS 实现的高效语义检索，以及可插拔的大语言模型 (LLM) 后端，用于根据您提供的文档回答问题。

## 项目结构

```
my-rag-app/
├── README.md
├── requirements.txt             # Python 依赖库
├── .env                         # 环境变量配置文件 (需自行创建)
├── api.py                       # FastAPI 应用入口点
├── config.py                    # 基础配置加载
├── app/                         # 核心应用逻辑
│   ├── __init__.py
│   ├── retriever.py             # 文档嵌入、索引和检索 (Faiss + SentenceTransformer)
│   ├── rag_pipeline.py          # RAG 流水线，协调检索和生成
│   └── llm_service.py           # 调用所选 LLM 策略的服务层
│   └── llm_strategies/          # 可插拔的 LLM 后端策略
│       ├── __init__.py          # 工厂函数 (get_llm_strategy)
│       ├── base.py              # LLM 策略的抽象基类
│       ├── ollama_strategy.py   # 通过 LangChain 与本地 Ollama 模型交互的策略
│       ├── openai_strategy.py   # 通过 LangChain 调用 OpenAI API 的策略
│       └── custom_api_strategy.py # 使用 openai 库调用自定义 OpenAI 兼容 API 的策略
├── data/
│   ├── documents/               # 存放原始知识文档 (.txt 文件)
│   └── indexes/                 # 存放 Faiss 索引文件和文档映射
├── models/                      # 本地 Sentence Transformer 模型缓存目录
├── logs/                        # 日志文件目录
├── public/                      # 前端 UI 静态文件
│   ├── index.html
│   └── ... (assets, static js/css)
└── utils/                       # 工具模块
    ├── __init__.py
    ├── env_helper.py            # 环境变量加载与解析
    ├── gpu_manager.py           # GPU 检测与信息
    ├── document_manager.py      # 文档加载与管理
    ├── model_utils.py           # 模型加载工具 (ModelManager)
    └── logger.py                # 日志配置
```

## 安装

首先创建并激活 conda 环境（或使用您偏好的虚拟环境）：

```bash
conda create -n rag-app python=3.10 # 或您偏好的 Python 版本
conda activate rag-app
```

安装所有依赖：

```bash
pip install -r requirements.txt
```
关键依赖包括: `fastapi`, `uvicorn`, `langchain`, `langchain-ollama`, `langchain-openai`, `openai`, `sentence-transformers`, `faiss-cpu` (或 `faiss-gpu`), `python-dotenv`。

## 配置 (.env 文件)

在项目根目录 (`my-rag-app/`) 创建一个名为 `.env` 的文件来配置应用程序。请参考以下模板：

```dotenv
# --- LLM 提供商选择 ---
# 选择后端 LLM 服务。可选值: 'ollama', 'custom_api', 'openai'
LLM_PROVIDER=ollama # 或者 custom_api, openai

# --- Ollama 配置 (如果 LLM_PROVIDER=ollama) ---
OLLAMA_HOST=http://localhost:11434       # 你的 Ollama 服务 URL
OLLAMA_MODEL=qwen3:0.6b                 # Ollama 中可用的模型名称 (例如，运行 'ollama list')
# OLLAMA_TEMPERATURE=0.7                 # (可选) 采样温度
# OLLAMA_NUM_PREDICT=256                # (可选) Ollama 的最大 Token 数

# --- 自定义 API 配置 (如果 LLM_PROVIDER=custom_api) ---
CUSTOM_API_KEY="sk-your_custom_key"       # 你的自定义端点的 API Key
CUSTOM_API_BASE="https://api.fe8.cn/v1"   # 你的自定义端点的 Base URL
CUSTOM_API_MODEL="gpt-4o-mini"            # 自定义端点可用的模型名称
# CUSTOM_API_TEMPERATURE=0.7             # (可选)
# CUSTOM_API_MAX_TOKENS=1024              # (可选)

# --- OpenAI 配置 (如果 LLM_PROVIDER=openai) ---
# 通过 LangChain 使用官方 OpenAI API。
# 需要设置 OPENAI_API_KEY。 Base URL 是可选的。
OPENAI_API_KEY="sk-your_openai_key"       # 你的官方 OpenAI API Key
# OPENAI_API_BASE="https://api.openai.com/v1" # (可选) 如果你使用代理
OPENAI_MODEL="gpt-4o-mini"                # 官方 OpenAI 模型名称
# OPENAI_TEMPERATURE=0.7                 # (可选)
# OPENAI_MAX_TOKENS=1024                  # (可选)

# --- 检索设置 ---
TOP_K=3                               # 每次查询检索的文档片段数量
RETRIEVER_MODEL="moka-ai/m3e-base"    # 用于嵌入的 Sentence Transformer 模型
LOCAL_MODEL_DIR="models"              # 缓存嵌入模型的目录
INDEX_DIR="data/indexes"              # 存储 Faiss 索引和文档映射的目录
DOCS_DIR="data/documents"             # 包含源文档的目录

# --- 系统设置 ---
LOG_LEVEL="INFO"                      # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
# LOW_MEMORY_MODE=false              # (目前关联性较低，因为本地生成已移除)
# APP_PORT=8000                      # 端口现在从 config.py 读取, 但也可在此处设置
```

**重要提示**:
*   设置 `LLM_PROVIDER` 来选择您希望使用的后端。
*   为所选的提供商填写相应的配置变量（API 密钥、URL、模型名称等）。
*   确保您为 Ollama 或自定义 API 指定的模型名称在该服务端点确实可用。
*   确保 `DOCS_DIR` 和 `INDEX_DIR` 路径正确。

## 特性

*   **检索增强生成 (RAG):** 基于提供的文档回答问题。
*   **FastAPI 后端:** 提供健壮的 API 接口。
*   **高效语义搜索:** 使用 Sentence Transformers (默认为 `moka-ai/m3e-base`) 和 FAISS 查找相关的文档片段。
*   **可插拔 LLM 后端:** 使用策略模式 (`app/llm_strategies`)，通过配置轻松切换不同的 LLM 提供商 (Ollama, OpenAI, 自定义 OpenAI 兼容 API)。
*   **流式 API:** 支持 Server-Sent Events (SSE) 以实现实时答案生成 (如果所选的 LLM 策略支持流式传输)。
*   **GPU 支持:** 自动利用 GPU 进行 Sentence Transformer 嵌入和 FAISS 索引（如果安装了 `faiss-gpu` 且 GPU 可用）。
*   **简单 Web UI:** 包含一个基本的前端界面 (`/public`) 用于交互。

### GPU 加速

#### 安装 CUDA 和 PyTorch
确保安装了兼容的 CUDA 和 cuDNN 版本。安装正确的 PyTorch 版本：

```bash
# CUDA 11.8 示例
# pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 示例
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```
*(请检查 PyTorch 官网以获取适合您 CUDA 版本的正确命令)*

#### FAISS GPU 支持
要使用 GPU 加速 FAISS 索引（推荐用于大型数据集）：

```bash
pip uninstall -y faiss-cpu
pip install faiss-gpu>=1.7.4
```

## 使用方法

### 1. 准备文档

将您的知识文档（纯文本 `.txt` 文件）放入 `data/documents/` 目录中。

### 2. 配置后端

编辑项目根目录下的 `.env` 文件：
*   将 `LLM_PROVIDER` 设置为 `ollama`, `custom_api`, 或 `openai`。
*   为您选择的提供商填写所需的 API 密钥、URL 和模型名称。
*   如果需要，调整 `RETRIEVER_MODEL`, `TOP_K` 等参数。

### 3. 运行 API 服务器

在项目根目录 (`my-rag-app/`) 打开终端并运行：

```bash
# 确保您的 conda/虚拟环境已激活
# 如果使用 Ollama，请确保 OLLAMA_HOST 环境变量未设置或设置正确
# unset OLLAMA_HOST # (如果您怀疑 shell 变量覆盖了 .env 文件)

# 使用 Uvicorn 启动 FastAPI 服务器
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
*(端口号 (8000) 是默认值，请检查 `config.py` 或在 `.env` 中设置 `APP_PORT`)*

服务器将启动，加载嵌入模型，为 `data/documents/` 中的文档构建或加载 FAISS 索引，并初始化所选的 LLM 策略。

### 4. 交互

*   **Web UI:** 打开浏览器并访问 `http://localhost:8000/ui`。
*   **API 文档:** 在 `http://localhost:8000/docs` 访问交互式 API 文档 (Swagger UI)。您可以在此处测试 API 端点。
*   **API 端点:**
    *   `/query` (POST): 非流式查询。
    *   `/query/stream` (POST): 通过 SSE 进行流式查询。
    *   `/conversations/.../messages` (POST): 在会话上下文中通过 SSE 进行流式查询。
    *   请参阅 `/docs` 获取所有端点的更多详细信息。
