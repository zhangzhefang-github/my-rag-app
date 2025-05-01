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
├── streamlit_app.py             # Streamlit UI application entry point
├── app/                         # Core application logic
│   ├── __init__.py
│   ├── retriever.py             # Document embedding, indexing, and retrieval (Faiss + Embedding Strategies)
│   ├── rag_pipeline.py          # RAG pipeline orchestrating retrieval and generation
│   └── embedding_strategies/    # Pluggable Embedding backend strategies
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base class for Embedding strategies
│   │   ├── config.py            # Pydantic configuration models for embeddings
│   │   ├── factory.py           # Factory function (get_embedder)
│   │   ├── hf_embed.py          # Strategy for local HuggingFace models
│   │   ├── ollama_embed.py      # Strategy for Ollama embeddings via API
│   │   └── openai_embed.py      # Strategy for OpenAI embeddings via API
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
├── tests/                       # Unit and integration tests
│   └── ...
└── utils/                       # Utility modules
    ├── __init__.py
    ├── env_helper.py            # Environment variable loading and parsing
    ├── gpu_manager.py           # GPU detection and info
    ├── document_manager.py      # Document loading and management
    ├── model_utils.py           # (Deprecated) Old Model loading utilities
    └── logger.py                # Logging setup
└── .gitignore                   # Files and directories ignored by Git
└── LICENSE                      # Project License (MIT)
```

## Installation

1.  **Create Environment and Install Core Libraries:**
    Use the `conda-env.yml` file to create the Conda environment and install core dependencies (Python, PyTorch, Faiss, NumPy) managed by Conda.
    ```bash
    # Ensure you have Conda installed
    conda env create -f conda-env.yml
    ```
    *(Note: You might need to edit `conda-env.yml` first to select the correct `faiss-cpu` or `faiss-gpu` line and potentially adjust the `pytorch-cuda` version based on your system.)*

2.  **Activate the Environment:**
    Activate the newly created environment (the default name is `rag-gpu`, check `conda-env.yml` if you changed it).
    ```bash
    conda activate rag-gpu
    ```

3.  **Install Application Dependencies using uv:**
    Install the remaining Python application dependencies using `uv`. `uv` reads the `requirements.txt` file (which contains exact versions locked by the developers using `requirements.in`) and ensures your environment matches precisely.
    ```bash
    # uv should have been installed in the previous step via conda-env.yml
    uv pip sync requirements.txt
    ```

Your environment is now ready!

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
# APP_PORT=8000                      # Port is now read from api.py (main block) or can be set here
```

**Important**:
*   Set `LLM_PROVIDER` to choose your desired backend.
*   Fill in the corresponding configuration variables for the selected provider (API keys, URLs, model names).
*   Ensure the model name you specify for Ollama or Custom API is actually available at that service endpoint.
*   Make sure the `DOCS_DIR` and `INDEX_DIR` paths are correct.

## Features

*   **Retrieval Augmented Generation (RAG):** Answers questions based on provided documents.
*   **FastAPI Backend:** Provides a robust API interface.
*   **Efficient Semantic Search:** Uses configurable embedding strategies (HuggingFace, OpenAI, Ollama via `app/embedding_strategies`) and FAISS for finding relevant document chunks.
*   **Pluggable LLM Backends:** Easily switch between different LLM providers (Ollama, OpenAI, Custom OpenAI-compatible API) via configuration using a Strategy Pattern (`app/llm_strategies`).
*   **Streaming API:** Supports Server-Sent Events (SSE) for real-time answer generation (if the selected LLM strategy supports streaming).
*   **GPU Support:** Automatically utilizes GPU for Sentence Transformer embeddings and FAISS indexing (if `faiss-gpu` is installed and GPU is available).
*   **Simple Web UI:** A basic frontend is included for interaction (served from `/public`).

### GPU Acceleration

GPU acceleration is primarily handled by ensuring the correct GPU versions of **PyTorch** and **Faiss** are installed during **Step 1** of the installation process via the `conda-env.yml` file.

*   Edit `conda-env.yml` **before** running `conda env create`:
    *   Ensure the `pytorch-cuda=XX.X` line matches your system's CUDA version.
    *   Comment out `faiss-cpu` and uncomment `faiss-gpu`.
*   If you need to switch between CPU and GPU versions *after* creating the environment:
    1.  Activate the environment: `conda activate rag-gpu`
    2.  Uninstall the incorrect Faiss version: `conda uninstall faiss-cpu` (or `faiss-gpu`)
    3.  Install the correct Faiss version: `conda install faiss-gpu -c pytorch` (or `faiss-cpu`)
    4.  You might also need to adjust PyTorch/CUDA versions if changing significantly.
    5.  Re-sync pip dependencies just in case: `uv pip sync requirements.txt`

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
*(The port (8000) is default, set `APP_PORT` in `.env` or check `api.py`)*

The server will start, load the embedding model, build/load the FAISS index for documents in `data/documents/`, and initialize the selected LLM strategy.

### 4. Interact

*   **Web UI:** Open your browser and navigate to `http://localhost:8000/ui`.
*   **API Docs:** Access the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`. You can test API endpoints here.
*   **API Endpoints:**
    *   `/query` (POST): Non-streaming query.
    *   `/query/stream` (POST): Streaming query via SSE.
    *   `/conversations/.../messages` (POST): Streaming query within a conversation context via SSE.
    *   See `/docs` for more details on all endpoints.

*   **Streamlit UI (Optional):** If `streamlit_app.py` is configured and run, access it at `http://localhost:8501` (default Streamlit port).

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
├── streamlit_app.py             # Streamlit UI 应用入口点
├── app/                         # 核心应用逻辑
│   ├── __init__.py
│   ├── retriever.py             # 文档嵌入、索引和检索 (Faiss + Embedding Strategies)
│   ├── rag_pipeline.py          # RAG 流水线，协调检索和生成
│   └── embedding_strategies/    # 可插拔的 Embedding 后端策略
│   │   ├── __init__.py
│   │   ├── base.py              # Embedding 策略的抽象基类
│   │   ├── config.py            # Embedding 的 Pydantic 配置模型
│   │   ├── factory.py           # 工厂函数 (get_embedder)
│   │   ├── hf_embed.py          # 本地 HuggingFace 模型策略
│   │   ├── ollama_embed.py      # 通过 API 调用 Ollama Embedding 的策略
│   │   └── openai_embed.py      # 通过 API 调用 OpenAI Embedding 的策略
│   └── llm_strategies/          # 可插拔的 LLM 后端策略
│       ├── __init__.py          # Factory function (get_llm_strategy)
│       ├── base.py              # Abstract base class for LLM strategies
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
├── tests/                       # 单元测试和集成测试
│   └── ...
└── utils/                       # 工具模块
    ├── __init__.py
    ├── env_helper.py            # 环境变量加载与解析
    ├── gpu_manager.py           # GPU 检测与信息
    ├── document_manager.py      # 文档加载与管理
    ├── model_utils.py           # (已弃用) 旧的模型加载工具
    └── logger.py                # 日志配置
└── .gitignore                   # Git 忽略的文件和目录
└── LICENSE                      # 项目许可证 (MIT)
```

## 安装

1.  **创建环境并安装核心库:**
    使用 `conda-env.yml` 文件创建 Conda 环境并安装由 Conda 管理的核心依赖（Python, PyTorch, Faiss, NumPy）。
    ```bash
    # 确保已安装 Conda
    conda env create -f conda-env.yml
    ```
    *(注意: 您可能需要先编辑 `conda-env.yml` 文件，选择正确的 `faiss-cpu` 或 `faiss-gpu` 行，并可能需要根据您的系统调整 `pytorch-cuda` 版本。)*

2.  **激活环境:**
    激活新创建的环境（默认名称为 `rag-gpu`，如果您修改了 `conda-env.yml`，请检查其中的名称）。
    ```bash
    conda activate rag-gpu
    ```

3.  **使用 uv 安装应用依赖:**
    使用 `uv` 安装剩余的 Python 应用依赖。`uv` 会读取 `requirements.txt` 文件（其中包含开发者使用 `requirements.in` 锁定的精确版本），并确保您的环境与之精确匹配。
    ```bash
    # uv 应已在上一步通过 conda-env.yml 安装
    uv pip sync requirements.txt
    ```

环境现已准备就绪！

## 配置 (.env 文件)

Create a `.env` file in the project root directory (`my-rag-app/`) to configure the application. Use the following template:

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
# APP_PORT=8000                      # 端口现在从 api.py (主代码块) 读取, 但也可在此处设置
```

**重要提示**:
*   设置 `LLM_PROVIDER` 来选择您希望使用的后端。
*   为所选的提供商填写相应的配置变量（API 密钥、URL、模型名称等）。
*   确保您为 Ollama 或自定义 API 指定的模型名称在该服务端点确实可用。
*   确保 `DOCS_DIR` 和 `INDEX_DIR` 路径正确。

## 特性

*   **检索增强生成 (RAG):** 基于提供的文档回答问题。
*   **FastAPI 后端:** 提供健壮的 API 接口。
*   **高效语义检索:** 使用可配置的 Embedding 策略 (HuggingFace, OpenAI, Ollama, 通过 `app/embedding_strategies`) 和 FAISS 查找相关文档块。
*   **可插拔 LLM 后端:** 通过配置轻松切换不同的 LLM 提供者 (Ollama, OpenAI, 自定义 OpenAI 兼容 API)，使用策略模式 (`app/llm_strategies`)。
*   **流式 API:** 支持 Server-Sent Events (SSE) 以实现实时答案生成 (如果所选的 LLM 策略支持流式传输)。
*   **GPU 支持:** 自动利用 GPU 进行 Sentence Transformer 嵌入和 FAISS 索引（如果安装了 `faiss-gpu` 且 GPU 可用）。
*   **简单 Web UI:** 包含一个基本的前端界面 (`/public`) 用于交互。

### GPU 加速

GPU 加速主要通过确保在安装过程的 **步骤 1** 中，通过 `conda-env.yml` 文件安装了正确 GPU 版本的 **PyTorch** 和 **Faiss** 来处理。

*   在运行 `conda env create` **之前**编辑 `conda-env.yml`：
    *   确保 `pytorch-cuda=XX.X` 行与您系统的 CUDA 版本匹配。
    *   注释掉 `faiss-cpu` 并取消注释 `faiss-gpu`。
*   如果需要在创建环境*后*切换 CPU 和 GPU 版本：
    1.  激活环境: `conda activate rag-gpu`
    2.  卸载错误的 Faiss 版本: `conda uninstall faiss-cpu` (或 `faiss-gpu`)
    3.  安装正确的 Faiss 版本: `conda install faiss-gpu -c pytorch` (或 `faiss-cpu`)
    4.  如果 CUDA 版本等有较大变化，可能也需要调整 PyTorch 版本。
    5.  为确保一致性，重新同步 pip 依赖: `uv pip sync requirements.txt`

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
*(端口号 (8000) 是默认值, 请在 `.env` 中设置 `APP_PORT` 或查看 `api.py`)*

服务器将启动，加载嵌入模型，为 `data/documents/` 中的文档构建或加载 FAISS 索引，并初始化所选的 LLM 策略。

### 4. Interact

*   **Web UI:** 打开浏览器并访问 `http://localhost:8000/ui`.
*   **API 文档:** 在 `http://localhost:8000/docs` 访问交互式 API 文档 (Swagger UI)。您可以在此处测试 API 端点。
*   **API 端点:**
    *   `/query` (POST): 非流式查询。
    *   `/query/stream` (POST): 通过 SSE 进行流式查询。
    *   `/conversations/.../messages` (POST): 在会话上下文中通过 SSE 进行流式查询。
    *   请参阅 `/docs` 获取所有端点的更多详细信息。

*   **Streamlit UI (可选):** 如果 `streamlit_app.py` 已配置并运行，请在 `http://localhost:8501` (默认 Streamlit 端口) 访问它。
