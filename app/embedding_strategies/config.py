from pydantic import BaseModel, Field, HttpUrl
from typing import Literal, Dict, Any, Optional

# Using Literal for known providers enforces type safety
ProviderType = Literal["openai", "huggingface", "ollama"] # Add more as needed

class BaseProviderConfig(BaseModel):
    provider: ProviderType
    # Common parameters can be added here if applicable

class OpenAIConfig(BaseProviderConfig):
    provider: Literal["openai"]
    model: str = "text-embedding-ada-002" # Or newer models like text-embedding-3-small
    api_key: str = Field(..., env="OPENAI_API_KEY") # Load from env var
    base_url: Optional[HttpUrl] = Field(None, env="OPENAI_BASE_URL") # Optional base URL
    # Add other OpenAI client parameters if needed (e.g., timeout, max_retries)
    client_kwargs: Dict[str, Any] = {}

class HuggingFaceConfig(BaseProviderConfig):
    provider: Literal["huggingface"]
    # Example: "sentence-transformers/all-MiniLM-L6-v2"
    model_path: str = Field(..., description="Path or identifier of the SentenceTransformer model")
    # device: Optional[str] = Field(None, description="Device to run the model on, e.g., 'cpu', 'cuda'") # Removed
    use_gpu: bool = Field(True, description="Whether to attempt using GPU if available.")
    local_model_dir: str = Field("models", description="Directory to save/load downloaded models locally.")
    # Add other SentenceTransformer parameters if needed
    model_kwargs: Dict[str, Any] = {}

class OllamaConfig(BaseProviderConfig):
    provider: Literal["ollama"]
    model: str = Field(..., description="Name of the Ollama model to use, e.g., 'nomic-embed-text'")
    base_url: HttpUrl = Field("http://localhost:11434", description="Base URL of the Ollama server")
    request_timeout: int = Field(60, description="Request timeout in seconds")

# Union type for configuration loading/validation
# This allows Pydantic to automatically determine the correct model based on the 'provider' field
# See: https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions
class EmbeddingConfig(BaseModel):
    config: OpenAIConfig | HuggingFaceConfig | OllamaConfig = Field(..., discriminator='provider')

# Example Usage:
# from pydantic import ValidationError
#
# try:
#     # Example loading from a dictionary (could also load from JSON/YAML file)
#     data = {
#         "config": {
#             "provider": "openai",
#             "model": "text-embedding-3-small",
#             # api_key would typically be set via environment variable OPENAI_API_KEY
#         }
#     }
#     config = EmbeddingConfig(**data)
#     print(config.config)
#
#     data_hf = {
#         "config": {
#             "provider": "huggingface",
#             "model_path": "sentence-transformers/paraphrase-MiniLM-L6-v2"
#         }
#     }
#     config_hf = EmbeddingConfig(**data_hf)
#     print(config_hf.config)
#
# except ValidationError as e:
#     print(e) 