"""Tests for `knowledge_agent._provenance.security_warning_text`.

Verifies the single source of truth for the install + download
dialog security warnings. The lifecycle modules' own tests
(`test_extractor_lifecycle`, `test_embedder_lifecycle`) cover the
end-to-end summaries; this module pins the helper's exact contract.
"""

from __future__ import annotations

from knowledge_agent._provenance import security_warning_text


def test_returns_empty_when_both_flags_safe():
    """safetensors=True + trust_remote_code=False → no warning."""
    assert (
        security_warning_text(
            safetensors=True,
            trust_remote_code=False,
        )
        == ""
    )


def test_returns_empty_when_only_safetensors_given_and_safe():
    """Embedder side passes only `safetensors=True`; helper defaults
    `trust_remote_code=False` so we don't warn falsely."""
    assert security_warning_text(safetensors=True) == ""


def test_warns_on_pickle_format():
    """safetensors=False surfaces the pickle warning by name."""
    text = security_warning_text(safetensors=False)
    assert text  # non-empty
    assert "pickle" in text.lower()
    assert "pytorch_model.bin" in text
    # The warning explains WHY it matters — "arbitrary code".
    assert "arbitrary code" in text.lower()


def test_warns_on_trust_remote_code():
    """trust_remote_code=True surfaces the load-time-code warning."""
    text = security_warning_text(
        safetensors=True,
        trust_remote_code=True,
    )
    assert text
    assert "trust_remote_code" in text
    assert "load time" in text.lower()


def test_warns_on_both_flags_in_one_text():
    """Both risks present → both warnings appear in the same string."""
    text = security_warning_text(
        safetensors=False,
        trust_remote_code=True,
    )
    assert "pickle" in text.lower()
    assert "trust_remote_code" in text


def test_warning_starts_with_leading_space():
    """The helper's output is concatenated AFTER the rest of the
    summary; leading space ensures it doesn't run into the previous
    sentence. Empty string returns NO leading space (clean append)."""
    assert security_warning_text(safetensors=False).startswith(" ")
    assert security_warning_text(safetensors=True) == ""


def test_warning_has_security_prefix():
    """Both warnings start with `SECURITY:` so they read as alerts,
    not as generic provenance lines."""
    assert "SECURITY:" in security_warning_text(safetensors=False)
    assert "SECURITY:" in security_warning_text(
        safetensors=True,
        trust_remote_code=True,
    )
