"""Tests for ingestion.embed - thin re-export over embedder_factory.

After the 2026-06-29 provider-swap refactor, `embed.py` is a 3-line
re-export of `embedder_factory.embed_texts`. These tests exercise
the same Voyage-multimodal path via the public `embed_texts` API to
make sure existing ingestion code that imports from
`knowledge_agent.ingestion.embed` keeps working.

For coverage of the OpenAI / Google / HuggingFace dispatch branches
see `tests/unit/test_embedder_factory.py`.
"""

from unittest.mock import Mock, patch

import pytest

from knowledge_agent.ingestion.embed import embed_texts

_VOYAGE_CLIENT_PATCH = "knowledge_agent.embedder_factory._build_voyage_client"
_SETTINGS_PATCH = "knowledge_agent.embedder_factory.get_settings"


def _voyage_settings():
    """Minimal stand-in: tells the factory to use voyage with a key set."""
    return Mock(
        embedding_provider="voyage",
        voyage_api_key="pa-voyage-stub",
        embedding_model="voyage-multimodal-3",
        voyage_requests_per_second=None,  # real Settings default: no rate cap
    )


async def test_embed_texts_empty_returns_empty_list():
    """No API call for empty input — short-circuits BEFORE provider
    validation, so no settings patching is needed."""
    assert await embed_texts([]) == []


async def test_embed_texts_returns_vectors_aligned_with_input():
    fake_result = Mock()
    fake_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]
    fake_client = Mock()
    fake_client.multimodal_embed = Mock(return_value=fake_result)

    with (
        patch(_SETTINGS_PATCH, return_value=_voyage_settings()),
        patch(_VOYAGE_CLIENT_PATCH, return_value=fake_client),
    ):
        result = await embed_texts(["text1", "text2"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


async def test_embed_texts_wraps_each_text_in_singleton_list_for_multimodal_api():
    fake_result = Mock()
    fake_result.embeddings = [[0.0], [0.0]]
    fake_client = Mock()
    fake_client.multimodal_embed = Mock(return_value=fake_result)

    with (
        patch(_SETTINGS_PATCH, return_value=_voyage_settings()),
        patch(_VOYAGE_CLIENT_PATCH, return_value=fake_client),
    ):
        await embed_texts(["a", "b"])

    call = fake_client.multimodal_embed.call_args
    # Multimodal API takes inputs as `list[list[str|Image]]`.
    assert call.kwargs["inputs"] == [["a"], ["b"]]


async def test_embed_texts_propagates_api_exception():
    """Typed-errors contract: Voyage API failures propagate; the
    orchestrator boundary (pipeline.ingest_document / re_embed) catches
    and records the typed error."""
    fake_client = Mock()
    fake_client.multimodal_embed = Mock(side_effect=RuntimeError("api down"))

    with (
        patch(_SETTINGS_PATCH, return_value=_voyage_settings()),
        patch(_VOYAGE_CLIENT_PATCH, return_value=fake_client),
    ):
        with pytest.raises(RuntimeError, match="api down"):
            await embed_texts(["text"])


async def test_embed_texts_default_input_type_is_document():
    fake_result = Mock()
    fake_result.embeddings = [[0.0]]
    fake_client = Mock()
    fake_client.multimodal_embed = Mock(return_value=fake_result)

    with (
        patch(_SETTINGS_PATCH, return_value=_voyage_settings()),
        patch(_VOYAGE_CLIENT_PATCH, return_value=fake_client),
    ):
        await embed_texts(["text"])

    call = fake_client.multimodal_embed.call_args
    assert call.kwargs["input_type"] == "document"


async def test_embed_texts_passes_query_input_type_through():
    fake_result = Mock()
    fake_result.embeddings = [[0.0]]
    fake_client = Mock()
    fake_client.multimodal_embed = Mock(return_value=fake_result)

    with (
        patch(_SETTINGS_PATCH, return_value=_voyage_settings()),
        patch(_VOYAGE_CLIENT_PATCH, return_value=fake_client),
    ):
        await embed_texts(["text"], input_type="query")

    call = fake_client.multimodal_embed.call_args
    assert call.kwargs["input_type"] == "query"
