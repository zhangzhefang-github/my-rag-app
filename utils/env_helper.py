"""
环境变量管理工具

本模块用于从.env文件或环境变量中加载配置信息。
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# 尝试导入dotenv，如果没有安装则提供备用方法
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("python-dotenv未安装，将仅使用系统环境变量")

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
            print(f"已加载环境变量：{env_file}")
        else:
            print(f"环境变量文件不存在：{env_file}，仅使用系统环境变量")
    
def get_api_config() -> Dict[str, Any]:
    """
    获取API配置信息
    
    Returns:
        包含API配置的字典
    """
    # 加载环境变量
    load_env_config()
    
    # 获取OpenAI相关配置
    config = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.environ.get("OPENAI_API_KEY", None),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.fe8.cn/v1")
    }
    
    return config

def get_system_config() -> Dict[str, Any]:
    """
    获取系统配置信息
    
    Returns:
        包含系统配置的字典
    """
    # 加载环境变量
    load_env_config()
    
    # 获取系统相关配置
    config = {
        "use_local_model": os.environ.get("USE_LOCAL_MODEL", "false").lower() in ["true", "1", "yes"],
        "low_memory_mode": os.environ.get("LOW_MEMORY_MODE", "false").lower() in ["true", "1", "yes"],
        "top_k": int(os.environ.get("TOP_K", 3)),
        "retriever_model": os.environ.get("RETRIEVER_MODEL", "moka-ai/m3e-base"),
        "local_model_dir": os.environ.get("LOCAL_MODEL_DIR", "models")
    }
    
    return config

def print_env_setup_guide() -> None:
    """打印环境变量设置指南"""
    print("""
===== 环境变量设置指南 =====

1. 创建.env文件，包含以下内容：
   
   # OpenAI API设置
   OPENAI_API_KEY=your_api_key_here
   OPENAI_BASE_URL=https://api.fe8.cn/v1
   OPENAI_MODEL=gpt-4o-mini

   # 检索设置
   TOP_K=3
   RETRIEVER_MODEL=moka-ai/m3e-base
   LOCAL_MODEL_DIR=models

   # 系统设置
   # 设置为true将使用本地模型而非OpenAI API
   USE_LOCAL_MODEL=false
   # 设置为true启用低内存模式
   LOW_MEMORY_MODE=false

2. 或者，您也可以直接设置环境变量：
   export OPENAI_API_KEY=your_actual_key_here
   
注意：.env文件中的设置优先级高于环境变量。
=========================
""") 