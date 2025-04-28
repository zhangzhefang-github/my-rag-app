# RAG Demo - 检索增强生成系统

_[English](#rag-demo---retrieval-augmented-generation-system) | [中文](#rag-demo---检索增强生成系统)_

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

```
# OpenAI API Settings
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.fe8.cn/v1
OPENAI_MODEL=gpt-4o-mini

# Retrieval Settings
TOP_K=3
RETRIEVER_MODEL=moka-ai/m3e-base
LOCAL_MODEL_DIR=models

# System Settings
USE_LOCAL_MODEL=false
LOW_MEMORY_MODE=false
```

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

# RAG Demo - 检索增强生成系统

## 概述

一个简单的检索增强生成(RAG)系统，结合文档检索和文本生成技术，基于您的文档回答问题。

## 项目结构

```
rag-demo/
├── README.md
├── requirements.txt
├── app/
│   ├── retriever.py          # 文档检索组件
│   ├── generator.py          # 本地文本生成模型
│   ├── openai_generator.py   # OpenAI API集成
│   ├── rag_pipeline.py       # 主RAG管道
├── data/
│   └── documents/            # 文档存储目录
├── models/                   # 本地模型缓存
├── utils/
│   ├── env_helper.py         # 环境变量工具
│   ├── gpu_manager.py        # GPU资源管理
│   ├── document_manager.py   # 文档处理
│   └── logger.py             # 日志工具
└── main.py                   # 应用程序入口点
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
```

## 配置

本项目使用环境变量进行配置：

### 方法1：使用.env文件（推荐）

创建.env文件，包含以下内容：

```
# OpenAI API设置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.fe8.cn/v1
OPENAI_MODEL=gpt-4o-mini

# 检索设置
TOP_K=3
RETRIEVER_MODEL=moka-ai/m3e-base
LOCAL_MODEL_DIR=models

# 系统设置
USE_LOCAL_MODEL=false
LOW_MEMORY_MODE=false
```

### 方法2：直接设置环境变量

```bash
export OPENAI_API_KEY=your_api_key_here
export OPENAI_BASE_URL=https://api.fe8.cn/v1
export OPENAI_MODEL=gpt-4o-mini
```

## 特性

### 中文嵌入模型支持

系统默认使用`moka-ai/m3e-base`进行中文文本嵌入。

### GPU支持

系统会自动检测并使用可用的GPU资源。

#### 安装CUDA和PyTorch
确保安装了兼容的CUDA和cuDNN版本：

```bash
# 对于CUDA 11.8
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 对于CUDA 12.1
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```

#### FAISS GPU支持
若要使用GPU加速FAISS索引：

```bash
pip uninstall -y faiss-cpu
pip install faiss-gpu>=1.7.4
```

### 低显存GPU支持

对于显存有限的GPU（如2GB），启用低内存模式：

```bash
python main.py --low-memory
```

在低内存模式下：
- 使用distilgpt2作为生成模型（占用资源更少）
- 优先将嵌入模型放在GPU上运行
- 在CPU上运行生成模型
- 根据可用显存决定FAISS索引的位置

## 使用方法

### 准备文档

将知识文档放入`data/documents/`目录中，使用.txt格式。

### 运行系统

```bash
# 查看环境变量设置指南
python main.py --show-env-guide

# 标准模式（使用OpenAI API）
python main.py

# 指定嵌入模型和检索文档数量
python main.py --embedding-model "moka-ai/m3e-base" --top-k 5

# 指定本地模型缓存目录
python main.py --local-model-dir "my_models"

# 使用本地模型代替OpenAI API
python main.py --use-local-model
```
