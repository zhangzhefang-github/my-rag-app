import pytest
from pydantic import ValidationError, HttpUrl
from app.embedding_strategies.config import (
    EmbeddingConfig, 
    HuggingFaceConfig, 
    OpenAIConfig, 
    OllamaConfig
)

# --- HuggingFaceConfig Tests ---

def test_hf_config_parsing_defaults():
    """Tests parsing HF config with only required fields, checking defaults."""
    data = {"config": {"provider": "huggingface", "model_path": "test/model"}}
    config_wrapper = EmbeddingConfig(**data)
    config = config_wrapper.config
    assert isinstance(config, HuggingFaceConfig)
    assert config.provider == "huggingface"
    assert config.model_path == "test/model"
    assert config.use_gpu is True  # Default
    assert config.local_model_dir == "models" # Default
    assert config.model_kwargs == {} # Default

def test_hf_config_parsing_explicit():
    """Tests parsing HF config with explicit values."""
    data = {
        "config": {
            "provider": "huggingface",
            "model_path": "another/model",
            "use_gpu": False,
            "local_model_dir": "custom_models/",
            "model_kwargs": {"trust_remote_code": True}
        }
    }
    config_wrapper = EmbeddingConfig(**data)
    config = config_wrapper.config
    assert isinstance(config, HuggingFaceConfig)
    assert config.provider == "huggingface"
    assert config.model_path == "another/model"
    assert config.use_gpu is False
    assert config.local_model_dir == "custom_models/"
    assert config.model_kwargs == {"trust_remote_code": True}

def test_hf_config_missing_required():
    """Tests that missing required model_path raises validation error."""
    data = {"config": {"provider": "huggingface"}} # Missing model_path
    with pytest.raises(ValidationError):
        EmbeddingConfig(**data)

# --- OpenAIConfig Tests ---

def test_openai_config_parsing(monkeypatch):
    """Tests parsing OpenAI config, mocking API key env var."""
    # Provide a dummy key directly in data, monkeypatch is less reliable here
    # monkeypatch.setenv("OPENAI_API_KEY", "test_key_123") 
    data = {"config": {"provider": "openai", "model": "text-embedding-3-large", "api_key": "test_dummy_key"}}
    config_wrapper = EmbeddingConfig(**data)
    config = config_wrapper.config
    assert isinstance(config, OpenAIConfig)
    assert config.provider == "openai"
    assert config.model == "text-embedding-3-large"
    assert config.api_key == "test_dummy_key" # Check against dummy key
    assert config.base_url is None
    assert config.client_kwargs == {}

def test_openai_config_with_base_url(monkeypatch):
    """Tests parsing OpenAI config with optional base_url."""
    # Provide a dummy key directly in data
    # monkeypatch.setenv("OPENAI_API_KEY", "test_key_456")
    base_url = "http://localhost:8080/v1"
    data = {
        "config": {
            "provider": "openai", 
            "model": "text-embedding-ada-002", 
            "api_key": "test_dummy_key_2", # Added dummy key
            "base_url": base_url,
            "client_kwargs": {"timeout": 30}
        }
    }
    config_wrapper = EmbeddingConfig(**data)
    config = config_wrapper.config
    assert isinstance(config, OpenAIConfig)
    assert config.api_key == "test_dummy_key_2" # Check against dummy key
    assert isinstance(config.base_url, HttpUrl)
    assert str(config.base_url).rstrip('/') == base_url # Check base URL (handle potential slash)
    assert config.client_kwargs == {"timeout": 30}

def test_openai_config_missing_api_key(monkeypatch):
    """Tests that missing OpenAI API key raises validation error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    data = {"config": {"provider": "openai", "model": "text-embedding-ada-002"}}
    with pytest.raises(ValidationError):
        EmbeddingConfig(**data)

# --- OllamaConfig Tests ---

def test_ollama_config_parsing_defaults():
    """Tests parsing Ollama config with only required field, checking defaults."""
    data = {"config": {"provider": "ollama", "model": "nomic-embed-text"}}
    config_wrapper = EmbeddingConfig(**data)
    config = config_wrapper.config
    assert isinstance(config, OllamaConfig)
    assert config.provider == "ollama"
    assert config.model == "nomic-embed-text"
    assert str(config.base_url) == "http://localhost:11434"
    assert config.request_timeout == 60 # Default

def test_ollama_config_parsing_explicit():
    """Tests parsing Ollama config with explicit values."""
    data = {
        "config": {
            "provider": "ollama", 
            "model": "mxbai-embed-large",
            "base_url": "http://192.168.1.100:11434",
            "request_timeout": 120
        }
    }
    config_wrapper = EmbeddingConfig(**data)
    config = config_wrapper.config
    assert isinstance(config, OllamaConfig)
    assert config.model == "mxbai-embed-large"
    assert str(config.base_url) == "http://192.168.1.100:11434/"
    assert config.request_timeout == 120

def test_ollama_config_missing_required():
    """Tests that missing required model raises validation error."""
    data = {"config": {"provider": "ollama"}} # Missing model
    with pytest.raises(ValidationError):
        EmbeddingConfig(**data)

# --- General EmbeddingConfig Tests ---

def test_invalid_provider():
    """Tests that an unknown provider raises validation error."""
    data = {"config": {"provider": "unknown_provider", "model": "some_model"}}
    with pytest.raises(ValidationError):
        EmbeddingConfig(**data) 