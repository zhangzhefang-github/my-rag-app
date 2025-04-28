import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.fe8.cn/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 检索配置
TOP_K = int(os.getenv("TOP_K", 3))
RETRIEVER_MODEL = os.getenv("RETRIEVER_MODEL", "moka-ai/m3e-base")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")

# 系统配置
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
LOW_MEMORY_MODE = os.getenv("LOW_MEMORY_MODE", "false").lower() == "true"
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "models")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
APP_PORT = int(os.getenv("APP_PORT", 8000))