from spamandphishingdetection.preprocessing import (
    TRUNCATION_MARKER,
    normalized_text_hash,
    prepare_text,
)
from spamandphishingdetection.training import _head_tail_token_ids


def test_prepare_text_preserves_security_signals() -> None:
    prepared, truncated = prepare_text(
        "  URGENT!\nVisit HTTPS://Example.test/Login  ",
        1_000,
    )

    assert prepared == "URGENT! Visit https://example.test/Login"
    assert truncated is False


def test_prepare_text_retains_head_and_tail() -> None:
    text = "HEAD" + ("x" * 2_000) + "TAIL"
    prepared, truncated = prepare_text(text, 1_000)

    assert truncated is True
    assert len(prepared) == 1_000
    assert prepared.startswith("HEAD")
    assert prepared.endswith("TAIL")
    assert TRUNCATION_MARKER in prepared


def test_hash_is_whitespace_and_case_insensitive() -> None:
    assert normalized_text_hash("Urgent  MESSAGE") == normalized_text_hash(" urgent\nmessage ")


def test_missing_numeric_value_is_empty() -> None:
    prepared, truncated = prepare_text(float("nan"), 1_000)

    assert prepared == ""
    assert truncated is False


def test_transformer_head_tail_tokenization() -> None:
    class TokenizerStub:
        cls_token_id = 101
        sep_token_id = 102

        @staticmethod
        def num_special_tokens_to_add(*, pair: bool) -> int:
            assert pair is False
            return 2

        @staticmethod
        def __call__(
            _text: str,
            *,
            add_special_tokens: bool,
            truncation: bool,
        ) -> dict[str, list[int]]:
            assert add_special_tokens is False
            assert truncation is False
            return {"input_ids": list(range(100))}

    token_ids = _head_tail_token_ids(TokenizerStub(), "ignored", max_length=10)

    assert token_ids == [101, 0, 1, 2, 3, 4, 5, 98, 99, 102]
