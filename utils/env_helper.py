"""
环境变量管理工具
本模块用于从.env文件或环境变量中加载配置信息。
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Get logger instance for this module
logger = logging.getLogger(__name__)

# 尝试导入dotenv，如果没有安装则提供备用方法
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logger.warning("python-dotenv未安装，将仅使用系统环境变量")

def load_env_config(env_file: str = ".env") -> None:
    """
    加载.env文件中的环境变量
    
    Args:
        env_file: .env文件路径
    """
    if DOTENV_AVAILABLE:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"已加载环境变量：{env_file}")
        else:
            logger.warning(f"环境变量文件不存在：{env_file}，仅使用系统环境变量")
    
def get_system_config() -> Dict[str, Any]:
    """
    获取系统配置信息 (通用和检索相关)
    
    Returns:
        包含系统配置的字典
    """
    # 获取系统相关配置
    config = {
        # 移除不再使用的配置项
        # "use_local_model": os.environ.get("USE_LOCAL_MODEL", "false").lower() in ["true", "1", "yes"],
        # "low_memory_mode": os.environ.get("LOW_MEMORY_MODE", "false").lower() in ["true", "1", "yes"],
        # 保留通用和检索配置
        "top_k": int(os.environ.get("TOP_K", 3)),
        "retriever_model": os.environ.get("RETRIEVER_MODEL", "moka-ai/m3e-base"),
        "local_model_dir": os.environ.get("LOCAL_MODEL_DIR", "models"),
        "index_dir": os.environ.get("INDEX_DIR", "data/indexes"),
        "docs_dir": os.environ.get("DOCS_DIR", "data/documents"),
        "log_level": os.environ.get("LOG_LEVEL", "INFO").upper(),
        "app_port": int(os.environ.get("APP_PORT", 8000))
    }
    
    return config

def print_env_setup_guide() -> None:
    """打印环境变量设置指南 (简化版)"""
    # 更新指南以反映当前的配置结构
    print("""
===== .env 文件配置指南 =====

在项目根目录创建 `.env` 文件，用于配置应用。模板如下：

# --- LLM 提供商选择 ---
# 可选值: 'ollama', 'custom_api', 'openai'
LLM_PROVIDER=ollama

# --- 根据 LLM_PROVIDER 填写对应配置 ---

# Ollama 配置示例:
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3:0.6b
# OLLAMA_TEMPERATURE=0.7
# OLLAMA_NUM_PREDICT=256

# 自定义 API 配置示例:
# CUSTOM_API_KEY=sk-your_custom_key
# CUSTOM_API_BASE=https://your.custom.endpoint/v1
# CUSTOM_API_MODEL=your_model_name
# CUSTOM_API_TEMPERATURE=0.7
# CUSTOM_API_MAX_TOKENS=1024

# OpenAI (官方 API) 配置示例:
# OPENAI_API_KEY=sk-your_openai_key
# OPENAI_API_BASE=https://api.openai.com/v1 # (可选, 用于代理)
# OPENAI_MODEL=gpt-4o-mini
# OPENAI_TEMPERATURE=0.7
# OPENAI_MAX_TOKENS=1024

# --- 检索设置 ---
TOP_K=3
RETRIEVER_MODEL="moka-ai/m3e-base"
LOCAL_MODEL_DIR="models"
INDEX_DIR="data/indexes"
DOCS_DIR="data/documents"

# --- 系统设置 ---
LOG_LEVEL="INFO" # (DEBUG, INFO, WARNING, ERROR, CRITICAL)
APP_PORT=8000    # API 服务器运行端口

=========================
""") 