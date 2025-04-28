# RAG Demo - 检索增强生成系统

<div id="language-switcher">
  <em><a href="#english-section">English</a> | <a href="#chinese-section">中文</a></em>
</div>

<div id="english-section"></div>

# RAG Demo - Retrieval Augmented Generation System

## Overview

A simple Retrieval Augmented Generation (RAG) system that combines document retrieval and text generation to answer questions based on your documents.

## Project Structure

```
rag-demo/
├── README.md
├── requirements.txt
├── app/
│   ├── retriever.py          # Document retrieval component
│   ├── generator.py          # Local text generation model
│   ├── openai_generator.py   # OpenAI API integration
│   ├── rag_pipeline.py       # Main RAG pipeline
├── data/
│   └── documents/            # Document storage directory
├── models/                   # Local model cache
├── utils/
│   ├── env_helper.py         # Environment variable utilities
│   ├── gpu_manager.py        # GPU resource management
│   ├── document_manager.py   # Document handling
│   └── logger.py             # Logging utilities
└── main.py                   # Application entry point
```

## Installation

First create and activate a conda environment:

```bash
conda create -n rag-gpu python=3.10
conda activate rag-gpu
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

This project uses environment variables for configuration:

### Method 1: Using a .env file (Recommended)

Create a .env file with the following content:

```dotenv
# OpenAI API设置 (如果 USE_LOCAL_MODEL=false)
OPENAI_API_KEY="your_api_key_here"  # 替换为你的 OpenAI API Key
OPENAI_BASE_URL="https://api.fe8.cn/v1" # 替换为你的 API 代理地址或官方地址
OPENAI_MODEL="gpt-4o-mini"          # 使用的 OpenAI 模型

# 检索设置
TOP_K=3                            # 每次查询检索的文档片段数量
RETRIEVER_MODEL="moka-ai/m3e-base" # 使用的 Sentence Transformer 嵌入模型
LOCAL_MODEL_DIR="models"           # 本地嵌入模型缓存目录
# VECTOR_DB_PATH="./vector_db"       # (旧配置，下面 index_dir 更准确) Faiss 索引文件的存储路径
INDEX_DIR="data/indexes"         # Faiss 索引文件及文档映射 (.index, .docs.json) 的存储目录
DOCS_DIR="data/documents"          # 原始知识文档 (.txt) 的存储目录

# 系统设置
# 设置为 true 将完全禁用 OpenAI API 调用，仅显示检索到的上下文
USE_LOCAL_MODEL=false
# 设置为 true 启用低内存模式 (影响GPU使用策略，详见特性部分)
LOW_MEMORY_MODE=false
# 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL="INFO"
```
**重要**: 确保 `.env` 文件中的 `VECTOR_DB_PATH` (如果在这里配置) 与 `app/retriever.py` 中 `Retriever` 类初始化时使用的 `index_dir` 路径一致 (默认为 `'data/indexes'`)，以避免索引文件位置混乱。建议统一配置或修改代码以使用单一配置源。
**重要**: `Retriever` 使用 `INDEX_DIR` 来存储 Faiss 索引和文档映射文件，使用 `DOCS_DIR` 来读取原始文档。请确保这些路径配置正确。

### Method 2: Setting Environment Variables Directly

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=https://api.fe8.cn/v1
export OPENAI_MODEL=gpt-4o-mini
```

## Features

### Chinese Embedding Model Support

The system uses `moka-ai/m3e-base` for Chinese text embedding by default.

### GPU Support

The system automatically detects and uses available GPU resources.

#### Installing CUDA and PyTorch
Ensure you have compatible CUDA and cuDNN versions installed:

```bash
# For CUDA 11.8
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```

#### FAISS GPU Support
To use GPU acceleration for FAISS indexing:

```bash
pip uninstall -y faiss-cpu
pip install faiss-gpu>=1.7.4
```

### Low VRAM GPU Support

For GPUs with limited VRAM (e.g., 2GB), enable low-memory mode:

```bash
python main.py --low-memory
```

In low-memory mode:
- Uses distilgpt2 as the generation model (smaller footprint)
- Prioritizes embedding model on GPU
- Runs generation model on CPU
- Places FAISS index based on available VRAM

## Usage

### Preparing Documents

Place your knowledge documents in the `data/documents/` directory as .txt files.

### Running the System

```bash
# View environment variable setup guide
python main.py --show-env-guide

# Standard mode (using OpenAI API)
python main.py

# Specify embedding model and number of documents to retrieve
python main.py --embedding-model "moka-ai/m3e-base" --top-k 5

# Specify local model cache directory
python main.py --local-model-dir "my_models"

# Use local model instead of OpenAI API
python main.py --use-local-model
```

---

<div id="chinese-section"></div>

# RAG Demo - 检索增强生成系统

## 概述

一个简单的检索增强生成(RAG)系统，结合文档检索和文本生成技术，基于您的文档回答问题。系统使用 Sentence Transformers 进行文档嵌入，FAISS进行向量检索，并可通过配置使用 OpenAI API 进行最终答案生成。

## 项目结构

```
rag-demo/
├── README.md
├── requirements.txt
├── .env                         # 环境变量配置文件 (需自行创建)
├── app/
│   ├── __init__.py
│   ├── retriever.py             # 核心组件：负责文档嵌入、索引管理和检索 (Faiss + SentenceTransformer)
│   ├── rag_pipeline.py          # RAG处理流水线，协调检索和生成
│   ├── openai_generator.py      # OpenAI API 调用封装
│   └── generator.py             # (当前未使用) 用于本地文本生成模型的占位符
├── data/
│   ├── documents/               # 存放原始知识文档 (.txt 文件)
│   └── indexes/                 # 存放生成的 Faiss 索引文件和文档映射
├── models/                      # 本地 Sentence Transformer 模型缓存目录
├── utils/
│   ├── __init__.py
│   ├── env_helper.py            # 环境变量加载与解析
│   ├── gpu_manager.py           # GPU 资源检测与管理
│   ├── document_manager.py      # 文档加载、变更检测与元数据管理
│   ├── model_utils.py           # 模型加载与缓存 (ModelManager)
│   └── logger.py                # 日志配置与记录
├── llm_service.py               # 负责调用生成模型 (当前为 OpenAI 或占位符)
└── main.py                      # 命令行应用程序入口点
```

## 安装

首先创建并激活conda环境：

```bash
conda create -n rag-gpu python=3.10
conda activate rag-gpu
```

安装所有依赖：

```bash
pip install -r requirements.txt
# 注意：如果遇到 ModuleNotFoundError: No module named 'langchain_community'
# 请尝试手动安装: pip install langchain-community
```

## 配置

本项目优先使用 `.env` 文件进行配置，其次是系统环境变量。

### 方法1：使用.env文件（推荐）

在项目根目录创建 `.env` 文件，包含以下内容（根据需要修改）：

```dotenv
# OpenAI API设置 (如果 USE_LOCAL_MODEL=false)
OPENAI_API_KEY="your_api_key_here"  # 替换为你的 OpenAI API Key
OPENAI_BASE_URL="https://api.fe8.cn/v1" # 替换为你的 API 代理地址或官方地址
OPENAI_MODEL="gpt-4o-mini"          # 使用的 OpenAI 模型

# 检索设置
TOP_K=3                            # 每次查询检索的文档片段数量
RETRIEVER_MODEL="moka-ai/m3e-base" # 使用的 Sentence Transformer 嵌入模型
LOCAL_MODEL_DIR="models"           # 本地嵌入模型缓存目录
# VECTOR_DB_PATH="./vector_db"       # (旧配置，下面 index_dir 更准确) Faiss 索引文件的存储路径
INDEX_DIR="data/indexes"         # Faiss 索引文件及文档映射 (.index, .docs.json) 的存储目录
DOCS_DIR="data/documents"          # 原始知识文档 (.txt) 的存储目录

# 系统设置
# 设置为 true 将完全禁用 OpenAI API 调用，仅显示检索到的上下文
USE_LOCAL_MODEL=false
# 设置为 true 启用低内存模式 (影响GPU使用策略，详见特性部分)
LOW_MEMORY_MODE=false
# 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL="INFO"
```
**重要**: 确保 `.env` 文件中的 `VECTOR_DB_PATH` (如果在这里配置) 与 `app/retriever.py` 中 `Retriever` 类初始化时使用的 `index_dir` 路径一致 (默认为 `'data/indexes'`)，以避免索引文件位置混乱。建议统一配置或修改代码以使用单一配置源。
**重要**: `Retriever` 使用 `INDEX_DIR` 来存储 Faiss 索引和文档映射文件，使用 `DOCS_DIR` 来读取原始文档。请确保这些路径配置正确。

### 方法2：直接设置环境变量

例如，在 Linux 或 macOS 系统中：
```bash
export OPENAI_API_KEY="your_actual_key_here"
export USE_LOCAL_MODEL="false"
# ... 其他变量
```

## 特性

### 中文嵌入模型支持

系统默认使用 `moka-ai/m3e-base` Sentence Transformer 模型进行中文文本嵌入。模型会自动下载并缓存在 `LOCAL_MODEL_DIR` 指定的目录（默认为 `models/`）。

### GPU 支持

系统会自动检测可用的 NVIDIA GPU 并优先用于以下任务：
1.  **嵌入模型计算**: 使用 `sentence-transformers` 加载模型到 GPU 进行文档和查询的向量化。
2.  **FAISS 索引**: 如果安装了 `faiss-gpu` 包，并且显存足够，FAISS 索引将被加载到 GPU 进行更快的相似度搜索。

#### 安装CUDA和PyTorch
确保已正确安装 NVIDIA 驱动、CUDA Toolkit 和 cuDNN。然后安装与你的 CUDA 版本兼容的 PyTorch：
```bash
# 示例：CUDA 11.8
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 示例：CUDA 12.1
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```
*(请访问 PyTorch 官网获取适合你环境的最新安装命令)*

#### FAISS GPU 支持
默认安装的是 `faiss-cpu`。若需启用 FAISS 的 GPU 加速：
1.  确保 CUDA 环境已配置。
2.  卸载 CPU 版本并安装 GPU 版本：
    ```bash
    pip uninstall -y faiss-cpu
    pip install faiss-gpu>=1.7.4
    ```
*注意*: 如果你看到日志中有 `WARNING - FAISS不支持GPU，请安装faiss-gpu` 的提示，说明你需要执行此步骤才能利用 GPU 加速 FAISS。

### 低显存 GPU 支持 (`--low-memory`)

对于显存有限的 GPU（例如 <= 2GB），可以通过 `--low-memory` 命令行参数启用低内存模式。在此模式下：
- `GPUManager` 会调整策略，可能优先保证嵌入模型使用 GPU。
- FAISS 索引可能因显存不足而保留在 CPU 上（即使安装了 `faiss-gpu`）。
- *(注意：当前版本没有实现在低内存模式下自动切换到小型本地生成模型的功能)*

## 使用方法

### 准备文档

将你的知识文档（目前仅支持 `.txt` 格式）放入 `data/documents/` 目录中。`DocumentManager` 会自动检测新增或修改的文档。

### 运行系统

```bash
# 查看环境变量设置指南
python main.py --show-env-guide

# 标准模式 (默认使用 .env 配置，通常会调用 OpenAI API)
python main.py

# 强制禁用 GPU (即使检测到可用 GPU)
python main.py --no-gpu

# 强制重新加载所有文档并重建索引 (而不是增量加载)
python main.py --reload-index

# 明确指定使用 OpenAI API (覆盖 .env 或默认设置)
# (注意：实际效果取决于 .env 中的 USE_LOCAL_MODEL 和命令行参数 --use-local-model)
# 推荐通过 USE_LOCAL_MODEL=false 和不加 --use-local-model 参数来启用 OpenAI

# 强制使用本地模式 (禁用 OpenAI API 调用)
python main.py --use-local-model

# 指定不同的嵌入模型和检索数量
python main.py --embedding-model "other/embedding-model" --top-k 5

# 指定本地模型缓存目录
python main.py --local-model-dir "path/to/my/models"

# 以 DEBUG 日志级别运行，查看详细过程
python main.py --log-level DEBUG
```

启动后，系统会加载文档、初始化模型和索引，然后进入交互式问答模式。输入你的问题，按回车键获取答案。输入 'exit' 或 'quit' 退出。
