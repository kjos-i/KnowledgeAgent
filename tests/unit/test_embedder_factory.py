"""Tests for `embedder_factory` — provider-agnostic embedding dispatch.

Voyage keeps its native client path; the other three providers
(OpenAI / Google / HuggingFace) route through `init_embeddings`
via LangChain wrappers. All four are exercised under patches so
no real API calls are made.
"""

from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from knowledge_agent import embedder_factory
from knowledge_agent.embedder_factory import (
    clear_cache,
    embed_texts,
)
from knowledge_agent.llm_factory import ConfigError


class _FakeSettings:
    def __init__(
        self,
        *,
        embedding_provider: str = "voyage",
        voyage_api_key: str = "pa-voyage-stub",
        openai_api_key: str | None = None,
        google_api_key: str | None = None,
        embedding_model: str = "voyage-multimodal-3",
        openai_embedding_model: str = "text-embedding-3-small",
        google_embedding_model: str = "models/text-embedding-004",
        hf_embedding_model: str = "BAAI/bge-m3",
    ):
        self.embedding_provider = embedding_provider
        self.voyage_api_key = voyage_api_key
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
        self.embedding_model = embedding_model
        self.openai_embedding_model = openai_embedding_model
        self.google_embedding_model = google_embedding_model
        self.hf_embedding_model = hf_embedding_model


@pytest.fixture(autouse=True)
def _clear_factory_cache():
    clear_cache()
    yield
    clear_cache()


# ---- lazy validation ----


async def test_empty_texts_returns_empty_no_validation_no_api_call():
    """`await embed_texts([])` must short-circuit BEFORE validation so
    callers that haven't set up a provider yet (test bootstrap) still
    get the empty-input no-op behaviour."""
    # No patching of get_settings — the function should never call it.
    assert await embed_texts([]) == []


async def test_voyage_missing_api_key_raises_config_error():
    settings = _FakeSettings(
        embedding_provider="voyage", voyage_api_key=""
    )
    with patch(
        "knowledge_agent.embedder_factory.get_settings",
        return_value=settings,
    ):
        with pytest.raises(ConfigError, match="VOYAGE_API_KEY"):
            await embed_texts(["hello"])


async def test_openai_missing_api_key_raises_config_error():
    settings = _FakeSettings(
        embedding_provider="openai", openai_api_key=None
    )
    with patch(
        "knowledge_agent.embedder_factory.get_settings",
        return_value=settings,
    ):
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            await embed_texts(["hello"])


async def test_google_missing_api_key_raises_config_error():
    settings = _FakeSettings(
        embedding_provider="google", google_api_key=None
    )
    with patch(
        "knowledge_agent.embedder_factory.get_settings",
        return_value=settings,
    ):
        with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
            await embed_texts(["hello"])


# ---- dispatch: voyage (native client path) ----


async def test_voyage_dispatch_calls_multimodal_embed():
    settings = _FakeSettings(
        embedding_provider="voyage", voyage_api_key="pa-stub"
    )
    fake_client = MagicMock()
    fake_client.multimodal_embed.return_value = MagicMock(
        embeddings=[[0.1, 0.2], [0.3, 0.4]]
    )
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=settings,
        ),
        patch(
            "knowledge_agent.embedder_factory._build_voyage_client",
            return_value=fake_client,
        ),
    ):
        result = await embed_texts(["one", "two"], input_type="document")
    assert result == [[0.1, 0.2], [0.3, 0.4]]
    fake_client.multimodal_embed.assert_called_once()
    call_kwargs = fake_client.multimodal_embed.call_args.kwargs
    # Each text wrapped as a single-item list (multimodal API shape).
    assert call_kwargs["inputs"] == [["one"], ["two"]]
    assert call_kwargs["model"] == "voyage-multimodal-3"
    assert call_kwargs["input_type"] == "document"


# ---- dispatch: langchain providers ----


async def test_openai_dispatch_uses_embed_documents_for_document_input():
    settings = _FakeSettings(
        embedding_provider="openai",
        openai_api_key="sk-openai",
    )
    fake_embedder = MagicMock()
    fake_embedder.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=settings,
        ),
        patch(
            "knowledge_agent.embedder_factory._build_langchain_embedder",
            return_value=fake_embedder,
        ),
    ):
        result = await embed_texts(["hello"], input_type="document")
    assert result == [[0.1, 0.2]]
    fake_embedder.aembed_documents.assert_called_once_with(["hello"])
    fake_embedder.aembed_query.assert_not_called()


async def test_google_dispatch_uses_embed_query_for_query_input():
    settings = _FakeSettings(
        embedding_provider="google",
        google_api_key="goog-key",
    )
    fake_embedder = MagicMock()
    fake_embedder.aembed_query = AsyncMock(side_effect=[[0.9, 0.8]])
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=settings,
        ),
        patch(
            "knowledge_agent.embedder_factory._build_langchain_embedder",
            return_value=fake_embedder,
        ),
    ):
        result = await embed_texts(["what is X?"], input_type="query")
    assert result == [[0.9, 0.8]]
    fake_embedder.aembed_query.assert_called_once_with("what is X?")
    fake_embedder.aembed_documents.assert_not_called()


async def test_huggingface_dispatch_no_api_key_needed():
    settings = _FakeSettings(embedding_provider="huggingface")
    fake_embedder = MagicMock()
    fake_embedder.aembed_documents = AsyncMock(return_value=[[0.5, 0.5]])
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=settings,
        ),
        patch(
            "knowledge_agent.embedder_factory._build_langchain_embedder",
            return_value=fake_embedder,
        ),
    ):
        # No api key on settings — HF runs locally.
        result = await embed_texts(["x"], input_type="document")
    assert result == [[0.5, 0.5]]
