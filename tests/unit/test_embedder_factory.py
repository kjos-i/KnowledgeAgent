"""Tests for `embedder_factory` — provider-agnostic embedding dispatch.

Voyage keeps its native client path; the other three providers
(OpenAI / Google / HuggingFace) route through `init_embeddings`
via LangChain wrappers. All four are exercised under patches so
no real API calls are made.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_agent.embedder_factory import (
    clear_cache,
    embed_chunks,
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
    settings = _FakeSettings(embedding_provider="voyage", voyage_api_key="")
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=settings,
        ),
        pytest.raises(ConfigError, match="VOYAGE_API_KEY"),
    ):
        await embed_texts(["hello"])


async def test_openai_missing_api_key_raises_config_error():
    settings = _FakeSettings(embedding_provider="openai", openai_api_key=None)
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=settings,
        ),
        pytest.raises(ConfigError, match="OPENAI_API_KEY"),
    ):
        await embed_texts(["hello"])


async def test_google_missing_api_key_raises_config_error():
    settings = _FakeSettings(embedding_provider="google", google_api_key=None)
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=settings,
        ),
        pytest.raises(ConfigError, match="GOOGLE_API_KEY"),
    ):
        await embed_texts(["hello"])


# ---- dispatch: voyage (native client path) ----


async def test_voyage_dispatch_calls_multimodal_embed():
    settings = _FakeSettings(embedding_provider="voyage", voyage_api_key="pa-stub")
    fake_client = MagicMock()
    fake_client.multimodal_embed.return_value = MagicMock(embeddings=[[0.1, 0.2], [0.3, 0.4]])
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


async def test_langchain_model_comes_from_embedding_model_not_per_provider():
    """Single source: the factory resolves the model from
    `settings.embedding_model` for EVERY provider — NOT the per-provider
    field (`openai_embedding_model`, ...), which is a GUI menu-default only.
    Give the two DIFFERENT values and assert the LangChain embedder is built
    with `embedding_model`. Guards the 2026-07-13 per-corpus change that
    dropped `_active_model_for`'s per-provider fan-out."""
    settings = _FakeSettings(
        embedding_provider="openai",
        openai_api_key="sk-openai",
        embedding_model="text-embedding-3-large",  # the corpus's resolved model
        openai_embedding_model="text-embedding-3-small",  # must be IGNORED
    )
    fake_embedder = MagicMock()
    fake_embedder.aembed_documents = AsyncMock(return_value=[[0.1]])
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=settings,
        ),
        patch(
            "knowledge_agent.embedder_factory._build_langchain_embedder",
            return_value=fake_embedder,
        ) as build,
    ):
        await embed_texts(["hi"], input_type="document")
    build.assert_called_once_with("openai", "text-embedding-3-large")


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


async def test_langchain_embedder_build_runs_off_the_event_loop():
    """The HuggingFace model load is a large synchronous call; embed_texts must
    build the embedder via asyncio.to_thread so the load doesn't freeze the
    event loop (the Voyage path already threads). Regression:
    _build_langchain_embedder ran inline on the loop."""
    import asyncio

    import knowledge_agent.embedder_factory as EF

    settings = _FakeSettings(embedding_provider="huggingface")
    fake_embedder = MagicMock()
    fake_embedder.aembed_documents = AsyncMock(return_value=[[0.1]])

    offloaded = []
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(fn, *a, **k):
        offloaded.append(fn)
        return await real_to_thread(fn, *a, **k)

    with (
        patch("knowledge_agent.embedder_factory.get_settings", return_value=settings),
        patch(
            "knowledge_agent.embedder_factory._build_langchain_embedder",
            return_value=fake_embedder,
        ) as mock_build,
        patch.object(EF.asyncio, "to_thread", spy_to_thread),
    ):
        result = await embed_texts(["x"], input_type="document")

    assert result == [[0.1]]
    assert mock_build in offloaded  # the blocking build was offloaded to a thread


def test_huggingface_embedder_pins_model_revision():
    """Security: the HF embedder must load at the curated `pinned_revision`,
    not upstream `main`, so SentenceTransformer resolves the vetted snapshot
    Installs downloaded.

    langchain-huggingface isn't installed at unit-test time, so it's injected
    as a mock; the real embedder load is exercised by the embedder smoke.
    """
    import sys

    from knowledge_agent.embedder_factory import _build_langchain_embedder
    from knowledge_agent.embedder_lifecycle import HF_EMBEDDING_MODELS

    model = "BAAI/bge-m3"
    pinned = HF_EMBEDDING_MODELS[model].pinned_revision
    fake_hf_cls = MagicMock()
    fake_module = MagicMock()
    fake_module.HuggingFaceEmbeddings = fake_hf_cls
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=_FakeSettings(embedding_provider="huggingface"),
        ),
        patch.dict(sys.modules, {"langchain_huggingface": fake_module}),
    ):
        _build_langchain_embedder("huggingface", model)
    fake_hf_cls.assert_called_once()
    kw = fake_hf_cls.call_args.kwargs
    assert kw["model_name"] == model
    assert kw["model_kwargs"]["revision"] == pinned


# ---- multimodal (embed_chunks + _voyage_multimodal_call) ----


class _FakeChunk:
    """Minimal duck for ParsedChunk / LanceDB row shape."""

    def __init__(self, text: str = "", image_ref: str | None = None):
        self.text = text
        self.image_ref = image_ref


def _fake_pil_open(path):
    """Replace PIL.Image.open with a marker so tests don't touch disk."""

    class _M:
        def __init__(self, p):
            self.path = p

        def convert(self, mode):
            return self

    return _M(path)


async def test_embed_chunks_empty_returns_empty_no_api_call():
    """Empty input short-circuits before any get_settings call."""
    assert await embed_chunks([]) == []


async def test_embed_chunks_voyage_multimodal_builds_mixed_inputs(tmp_path):
    """Voyage path: figure with text → [text, image]; figure with empty
    text → [image]; text-only chunk → [text]. Each aligned 1:1."""
    settings = _FakeSettings(embedding_provider="voyage", voyage_api_key="pa-stub")
    fake_client = MagicMock()
    fake_client.multimodal_embed.return_value = MagicMock(embeddings=[[0.1], [0.2], [0.3]])
    fig1 = tmp_path / "1.png"
    fig1.write_bytes(b"fake-png-bytes")
    fig2 = tmp_path / "2.png"
    fig2.write_bytes(b"fake-png-bytes")

    chunks = [
        _FakeChunk(text="caption", image_ref=str(fig1)),  # figure + caption
        _FakeChunk(text="", image_ref=str(fig2)),  # figure + empty text
        _FakeChunk(text="hello"),  # plain text chunk
    ]
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=settings,
        ),
        patch(
            "knowledge_agent.embedder_factory._build_voyage_client",
            return_value=fake_client,
        ),
        patch(
            "PIL.Image.open",
            side_effect=_fake_pil_open,
        ),
    ):
        result = await embed_chunks(chunks)

    assert result == [[0.1], [0.2], [0.3]]
    fake_client.multimodal_embed.assert_called_once()
    inputs = fake_client.multimodal_embed.call_args.kwargs["inputs"]
    # inputs[0]: figure with caption → [text, image] (2 items)
    assert len(inputs[0]) == 2
    assert inputs[0][0] == "caption"
    # inputs[1]: figure with empty text → [image] (1 item, PIL image)
    assert len(inputs[1]) == 1
    # inputs[2]: plain text chunk → [text] (1 item, string)
    assert inputs[2] == ["hello"]


async def test_embed_chunks_non_voyage_warns_on_figure(caplog):
    """Non-Voyage providers can't do multimodal — figure chunks embed
    from text only and a warning fires once per call."""
    settings = _FakeSettings(
        embedding_provider="openai",
        openai_api_key="sk-openai",
    )
    fake_embedder = MagicMock()
    fake_embedder.aembed_documents = AsyncMock(return_value=[[0.9], [0.8]])
    chunks = [
        _FakeChunk(text="figure caption", image_ref="/tmp/pic.png"),
        _FakeChunk(text="plain text"),
    ]
    with (
        patch(
            "knowledge_agent.embedder_factory.get_settings",
            return_value=settings,
        ),
        patch(
            "knowledge_agent.embedder_factory._build_langchain_embedder",
            return_value=fake_embedder,
        ),
        caplog.at_level("WARNING"),
    ):
        result = await embed_chunks(chunks)

    assert result == [[0.9], [0.8]]
    # LangChain embedder gets ONLY text (image_ref ignored).
    fake_embedder.aembed_documents.assert_called_once_with(
        ["figure caption", "plain text"],
    )
    # Warning surfaced once per call, naming the provider.
    warnings = [
        r
        for r in caplog.records
        if r.levelname == "WARNING" and "does not support multimodal" in r.getMessage()
    ]
    assert len(warnings) == 1
    assert "openai" in warnings[0].getMessage()


async def test_embed_chunks_accepts_dict_rows_from_lancedb():
    """re_embed feeds LanceDB row dicts to `embed_chunks`; must
    duck-type on `text` + `image_ref` keys."""
    settings = _FakeSettings(embedding_provider="voyage", voyage_api_key="pa-stub")
    fake_client = MagicMock()
    fake_client.multimodal_embed.return_value = MagicMock(embeddings=[[0.7]])
    rows = [{"text": "row text", "image_ref": None}]
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
        result = await embed_chunks(rows)
    assert result == [[0.7]]
    inputs = fake_client.multimodal_embed.call_args.kwargs["inputs"]
    assert inputs == [["row text"]]
