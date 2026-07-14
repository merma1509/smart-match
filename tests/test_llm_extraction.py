"""Tests for LLM-Assisted Extraction Service."""
import pytest
import json
from unittest.mock import patch, MagicMock
from app.services.llm_extraction import LLMAssistedExtractor, smart_extract


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def extractor():
    """Create LLM extractor with mocked dependencies."""
    with patch('app.services.llm_extraction.InformationExtractor') as mock_rule:
        mock_instance = MagicMock()
        mock_instance.extract.return_value = {
            "record_type": "birth",
            "child_name": {"value": "Unknown", "confidence": 0.0},
            "birth_date": {"value": "Unknown", "confidence": 0.0},
            "father_name": {"value": "Unknown", "confidence": 0.0},
            "mother_name": {"value": "Unknown", "confidence": 0.0},
            "needs_review": True,
            "_extraction": {
                "method": "rule_based",
                "average_confidence": 0.3,
                "field_count": 0,
                "found_fields": []
            }
        }
        mock_rule.return_value = mock_instance
        extractor = LLMAssistedExtractor(
            model_name="test-model",
            use_ollama=True
        )
        extractor.ollama_client = MagicMock()
        extractor.rule_extractor = mock_instance
        return extractor


@pytest.fixture
def sample_birth_text():
    return "У крестьянина Петра Иванова родился сын Иван 12 марта 1878 года."


@pytest.fixture
def valid_llm_response():
    return json.dumps({
        "record_type": "birth",
        "child_name": {"value": "Иван Петров", "confidence": 0.85},
        "birth_date": {"value": "1878-03-12", "confidence": 0.9},
        "father_name": {"value": "Петр Иванов", "confidence": 0.88},
        "mother_name": {"value": "Анна Иванова", "confidence": 0.86}
    })


# ── Test: Initialization ────────────────────────────────────────────────────
class TestLLMInit:
    def test_init_with_ollama(self):
        with patch('app.services.llm_extraction.InformationExtractor') as mock_rule:
            extractor = LLMAssistedExtractor(use_ollama=True)
            assert extractor.use_ollama is True
            assert extractor.confidence_threshold == 0.7

    def test_init_with_transformers(self):
        with patch('app.services.llm_extraction.InformationExtractor') as mock_rule:
            extractor = LLMAssistedExtractor(use_ollama=False)
            assert extractor.use_ollama is False

    def test_init_custom_threshold(self):
        with patch('app.services.llm_extraction.InformationExtractor') as mock_rule:
            extractor = LLMAssistedExtractor()
            extractor.confidence_threshold = 0.5
            assert extractor.confidence_threshold == 0.5


# ── Test: Rule-based Extraction ─────────────────────────────────────────────

class TestRuleBasedExtraction:
    def test_rule_based_high_confidence(self, extractor, sample_birth_text):
        """Test that high-confidence rule extraction doesn't use LLM."""
        extractor.rule_extractor.extract.return_value = {
            "record_type": "birth",
            "child_name": {"value": "Иван", "confidence": 0.8},
            "_extraction": {"average_confidence": 0.85, "method": "rule_based"}
        }
        result = extractor.extract(sample_birth_text)
        assert result["_extraction"]["method"] == "rule_based"

    def test_rule_based_low_confidence_triggers_llm(self, extractor, sample_birth_text):
        """Test that low confidence triggers LLM."""
        extractor.rule_extractor.extract.return_value = {
            "record_type": "birth",
            "child_name": {"value": "Unknown", "confidence": 0.0},
            "needs_review": True,
            "_extraction": {"average_confidence": 0.3, "method": "rule_based"}
        }
        extractor.extract_with_llm = MagicMock(return_value={
            "record_type": "birth",
            "child_name": {"value": "Иван", "confidence": 0.9},
            "_extraction": {"method": "llm"}
        })
        result = extractor.extract(sample_birth_text)
        assert extractor.extract_with_llm.called


# ── Test: LLM Query ─────────────────────────────────────────────────────────
class TestLLMQuery:
    def test_ollama_query_success(self, extractor):
        """Test successful Ollama query."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": '{"record_type": "birth"}'}
        extractor.ollama_client.post.return_value = mock_response
        result = extractor._query_ollama("test prompt")
        assert result == '{"record_type": "birth"}'

    def test_ollama_query_failure(self, extractor):
        """Test Ollama query failure falls back."""
        extractor.ollama_client.post.side_effect = Exception("Connection error")
        result = extractor._query_ollama("test prompt")
        assert result == ""

    def test_ollama_no_client(self, extractor):
        """Test when Ollama client is None."""
        extractor.ollama_client = None
        result = extractor._query_ollama("test prompt")
        assert result == ""

    def test_ollama_http_error(self, extractor):
        """Test non-200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        extractor.ollama_client.post.return_value = mock_response
        result = extractor._query_ollama("test prompt")
        assert result == ""


# ── Test: LLM Response Parsing ──────────────────────────────────────────────
class TestLLMParsing:
    def test_parse_valid_json(self, extractor, valid_llm_response):
        result = extractor._parse_llm_response(valid_llm_response)
        assert result is not None
        assert result["record_type"] == "birth"
        assert result["child_name"]["value"] == "Иван Петров"

    def test_parse_json_with_extra_text(self, extractor):
        response = "Here is the result: " + json.dumps({
            "record_type": "death",
            "deceased_name": {"value": "Михаил", "confidence": 0.9}
        })
        result = extractor._parse_llm_response(response)
        assert result is not None
        assert result["record_type"] == "death"

    def test_parse_truncated_json(self, extractor):
        """Test that truncated JSON is still parsed."""
        response = '{"record_type": "birth", "child_name": {"value": "Иван"'
        result = extractor._parse_llm_response(response)
        assert result is not None
        assert result["record_type"] == "birth"

    def test_parse_invalid_response(self, extractor):
        result = extractor._parse_llm_response("Not JSON at all")
        assert result is None

    def test_parse_empty_response(self, extractor):
        result = extractor._parse_llm_response("")
        assert result is None

    def test_parse_missing_record_type(self, extractor):
        """JSON without record_type should fail."""
        response = json.dumps({"name": "test"})
        result = extractor._parse_llm_response(response)
        assert result is None


# ── Test: extract_with_llm ──────────────────────────────────────────────────
class TestExtractWithLLM:
    def test_successful_llm_extraction(self, extractor, sample_birth_text):
        """Test full LLM extraction flow."""
        extractor._query_ollama = MagicMock(return_value=json.dumps({
            "record_type": "birth",
            "child_name": {"value": "Иван", "confidence": 0.9}
        }))
        result = extractor.extract_with_llm(sample_birth_text)
        assert result["record_type"] == "birth"
        assert result["_extraction"]["method"] == "llm"

    def test_llm_empty_response_falls_back(self, extractor, sample_birth_text):
        """Test empty LLM response falls back to rule-based."""
        extractor._query_ollama = MagicMock(return_value="")
        with patch.object(extractor.rule_extractor, 'extract') as mock_rule:
            mock_rule.return_value = {"record_type": "birth", "_extraction": {}}
            result = extractor.extract_with_llm(sample_birth_text)
            assert mock_rule.called

    def test_llm_parsing_failure_falls_back(self, extractor, sample_birth_text):
        """Test LLM parsing failure falls back to rule-based."""
        extractor._query_ollama = MagicMock(return_value="garbage")
        with patch.object(extractor.rule_extractor, 'extract') as mock_rule:
            mock_rule.return_value = {"record_type": "birth", "_extraction": {}}
            result = extractor.extract_with_llm(sample_birth_text)
            assert mock_rule.called

    def test_needs_review_flag(self, extractor, sample_birth_text):
        """Test needs_review is set based on average confidence."""
        extractor._query_ollama = MagicMock(return_value=json.dumps({
            "record_type": "birth",
            "child_name": {"value": "test", "confidence": 0.2}
        }))
        result = extractor.extract_with_llm(sample_birth_text)
        assert result["needs_review"] is True  # 0.2 < 0.5


# ── Test: Smart Extract ─────────────────────────────────────────────────────
class TestSmartExtract:
    def test_smart_extract_high_confidence(self, sample_birth_text):
        """Test smart_extract with high rule-based confidence."""
        with patch('app.services.llm_extraction.InformationExtractor') as mock_rule:
            mock_instance = MagicMock()
            mock_instance.extract.return_value = {
                "record_type": "birth",
                "child_name": {"value": "Иван", "confidence": 0.9},
                "_extraction": {"average_confidence": 0.9, "method": "rule_based"}
            }
            mock_rule.return_value = mock_instance
            result = smart_extract(sample_birth_text)
            assert result["_extraction"]["method"] == "rule_based"

    def test_smart_extract_force_llm(self, sample_birth_text):
        """Test force_llm parameter."""
        with patch('app.services.llm_extraction.LLMAssistedExtractor') as mock_ext:
            instance = MagicMock()
            instance.extract.return_value = {"result": "test"}
            mock_ext.return_value = instance
            result = smart_extract(sample_birth_text, force_llm=True)
            instance.extract.assert_called_with(sample_birth_text, force_llm=True)

    def test_smart_extract_custom_threshold(self, sample_birth_text):
        """Test that confidence threshold is configurable."""
        with patch('app.services.llm_extraction.LLMAssistedExtractor') as mock_ext:
            instance = MagicMock()
            instance.extract.return_value = {"result": "test"}
            mock_ext.return_value = instance
            result = smart_extract(sample_birth_text, confidence_threshold=0.5)
            assert result["result"] == "test"


# ── Test: Field Merging ─────────────────────────────────────────────────────
class TestFieldMerging:
    def test_merge_unknown_fields(self, extractor, sample_birth_text):
        """Test that LLM 'Unknown' fields are replaced by rule-based values."""
        extractor.extract_with_llm = MagicMock(return_value={
            "record_type": "birth",
            "child_name": {"value": "Unknown", "confidence": 0.5},
            "father_name": {"value": "Петр", "confidence": 0.8},
            "_extraction": {"method": "llm"}
        })
        extractor.rule_extractor.extract.return_value = {
            "record_type": "birth",
            "child_name": {"value": "Иван", "confidence": 0.6},
            "father_name": {"value": "Петр", "confidence": 0.8},
            "_extraction": {"average_confidence": 0.6}
        }
        result = extractor.extract(sample_birth_text)
        # child_name from LLM is Unknown, should be replaced by rule-based "Иван"
        assert result["child_name"]["value"] == "Иван"