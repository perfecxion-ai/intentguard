"""Tests for text normalization."""

from intentguard.normalize import has_encoding_tricks, normalize


class TestNormalize:
    def test_basic_text_unchanged(self):
        assert normalize("What are mortgage rates?") == "What are mortgage rates?"

    def test_nfkc_fullwidth(self):
        # Fullwidth characters should be converted to ASCII
        result = normalize("\uff37\uff48\uff41\uff54")  # "What" in fullwidth
        assert result == "What"

    def test_zero_width_stripped(self):
        # Zero-width space inserted between characters
        result = normalize("hel\u200blo wor\u200bld")
        assert result == "hello world"

    def test_whitespace_collapsed(self):
        result = normalize("  too   many    spaces  ")
        assert result == "too many spaces"

    def test_newlines_collapsed(self):
        result = normalize("line one\n\nline two\n\n\nline three")
        assert result == "line one line two line three"

    def test_truncation(self):
        long_text = "a" * 3000
        result = normalize(long_text, max_chars=100)
        assert len(result) == 100

    def test_empty_string(self):
        assert normalize("") == ""
        assert normalize("   ") == ""

    def test_soft_hyphen_stripped(self):
        result = normalize("fi\u00adnance")
        assert result == "finance"


class TestEncodingTricks:
    def test_normal_text_passes(self):
        assert has_encoding_tricks("What are current mortgage rates?") is False

    def test_base64_blob_detected(self):
        assert has_encoding_tricks("V2hhdCBhcmUgbW9ydGdhZ2UgcmF0ZXM=") is True

    def test_short_base64_passes(self):
        # Short strings that happen to look like base64 are fine
        assert has_encoding_tricks("Hello123") is False

    def test_normal_multilingual_passes(self):
        # Real multilingual text shouldn't trigger
        assert has_encoding_tricks("This is a normal English sentence about banking.") is False
