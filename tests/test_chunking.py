# tests/test_chunking.py

import pytest
import os
from unittest.mock import patch

# Assuming models are in app.models
from app.models.document import ParsedDocument, Chunk
# Assuming strategies are in app.chunking_strategies
from app.chunking_strategies.base import BaseChunkingStrategy
from app.chunking_strategies.recursive_character import RecursiveCharacterChunkingStrategy
from app.chunking_strategies.factory import get_chunker

# --- Test Fixtures ---

@pytest.fixture
def sample_parsed_doc() -> ParsedDocument:
    """Provides a sample ParsedDocument for testing."""
    return ParsedDocument(
        doc_id="test_doc_1",
        text="This is the first sentence. This is the second sentence, which is a bit longer. Finally, the third sentence.",
        metadata={"source": "test_source.txt", "author": "tester"}
    )

# --- Test Cases for RecursiveCharacterChunkingStrategy ---

def test_recursive_chunking_initialization():
    """Test initialization with default and custom parameters."""
    strategy_default = RecursiveCharacterChunkingStrategy()
    assert strategy_default.chunk_size == 1000
    assert strategy_default.chunk_overlap == 100

    strategy_custom = RecursiveCharacterChunkingStrategy(chunk_size=50, chunk_overlap=10)
    assert strategy_custom.chunk_size == 50
    assert strategy_custom.chunk_overlap == 10

def test_recursive_chunking_execution(sample_parsed_doc):
    """Test the chunking process itself."""
    # Use small chunk size for easier testing
    strategy = RecursiveCharacterChunkingStrategy(chunk_size=40, chunk_overlap=5)
    chunks = strategy.chunk(sample_parsed_doc)

    assert isinstance(chunks, list)
    assert len(chunks) > 1 # Expect multiple chunks for this text and chunk size

    for i, chunk in enumerate(chunks):
        assert isinstance(chunk, Chunk)
        assert chunk.text is not None
        assert len(chunk.text) <= strategy.chunk_size + 20 # Allow some leeway for splitting
        assert chunk.metadata is not None
        assert chunk.doc_id == sample_parsed_doc.doc_id
        assert chunk.metadata.chunk_index == i
        assert chunk.metadata.offset >= 0
        # Check if original metadata is preserved within chunk.metadata
        # With extra='allow', we should be able to access these as attributes
        assert chunk.metadata.source == sample_parsed_doc.metadata["source"]
        assert chunk.metadata.author == sample_parsed_doc.metadata.get("author") # Access as attribute

    # Check for overlap (simple check: end of one chunk might appear near start of next)
    if len(chunks) > 1:
        overlap_found = chunks[0].text[-strategy.chunk_overlap:] in chunks[1].text[:strategy.chunk_overlap+10] # Check first few chars of next
        assert overlap_found or len(chunks[0].text) < strategy.chunk_size # Overlap might not exist if first chunk is very short

def test_recursive_chunking_empty_doc():
    """Test chunking an empty document."""
    strategy = RecursiveCharacterChunkingStrategy()
    empty_doc = ParsedDocument(doc_id="empty", text="", metadata={"source": "empty.txt"})
    chunks = strategy.chunk(empty_doc)
    assert isinstance(chunks, list)
    assert len(chunks) == 0

def test_recursive_chunking_short_doc():
    """Test chunking a document shorter than chunk_size."""
    strategy = RecursiveCharacterChunkingStrategy(chunk_size=1000, chunk_overlap=100)
    short_doc = ParsedDocument(doc_id="short", text="Just one short sentence.", metadata={"source": "short.txt"})
    chunks = strategy.chunk(short_doc)
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one short sentence."
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[0].metadata.offset == 0
    assert chunks[0].doc_id == "short"

# --- Test Cases for get_chunker Factory ---

@patch.dict(os.environ, {}, clear=True) # Start with empty environment
def test_get_chunker_default():
    """Test factory default behavior (no env var set)."""
    chunker = get_chunker()
    assert isinstance(chunker, RecursiveCharacterChunkingStrategy)
    # Check default params are used
    assert chunker.chunk_size == 1000
    assert chunker.chunk_overlap == 100

@patch.dict(os.environ, {"CHUNKING_STRATEGY": "recursive_character", "CHUNK_SIZE": "200", "CHUNK_OVERLAP": "20"}, clear=True)
def test_get_chunker_recursive_from_env():
    """Test factory configuring RecursiveCharacter from env vars."""
    chunker = get_chunker()
    assert isinstance(chunker, RecursiveCharacterChunkingStrategy)
    assert chunker.chunk_size == 200
    assert chunker.chunk_overlap == 20

@patch.dict(os.environ, {"CHUNKING_STRATEGY": "recursive_character"}, clear=True)
def test_get_chunker_recursive_default_params_when_set():
    """Test factory uses defaults if size/overlap env vars are missing."""
    chunker = get_chunker()
    assert isinstance(chunker, RecursiveCharacterChunkingStrategy)
    assert chunker.chunk_size == 1000 # Default
    assert chunker.chunk_overlap == 100 # Default

@patch.dict(os.environ, {"CHUNKING_STRATEGY": "recursive_character", "CHUNK_SIZE": "invalid", "CHUNK_OVERLAP": "20"}, clear=True)
def test_get_chunker_invalid_env_vars():
    """Test factory handles non-integer values for size/overlap gracefully (uses defaults)."""
    chunker = get_chunker()
    assert isinstance(chunker, RecursiveCharacterChunkingStrategy)
    assert chunker.chunk_size == 1000 # Falls back to default
    assert chunker.chunk_overlap == 100 # Falls back to default (because size failed)

@patch.dict(os.environ, {"CHUNKING_STRATEGY": "unknown_strategy"}, clear=True)
def test_get_chunker_unknown_strategy():
    """Test factory raises error for an unknown strategy name."""
    with pytest.raises(ValueError, match="Unknown or unsupported chunking strategy"):
        get_chunker()

# Add tests for other strategies here when implemented
# e.g., test_get_chunker_semantic(), test_get_chunker_markdown() 