# Business logic: LLM-Assisted Extraction for ambiguous Russian genealogical records
# Uses a small local LLM (Mistral 7B or Llama 3.1 8B) when rule-based confidence is low
import json
import re

from loguru import logger

from app.core.config import settings
from app.core.llm_client import LLMClientBase, create_llm_client
from app.services.extraction import InformationExtractor


class LLMAssistedExtractor:
    """Extracts structured genealogical data using LLM assistance.
    Only activates when rule-based extraction confidence is below threshold.
    Supports multiple LLM providers via the LLMClientBase abstraction:

        - ollama:     Локальный Ollama (по умолчанию)
        - openai:     OpenAI / OpenAI-compatible (vLLM, TGI, Together AI, etc.)
        - huggingface: Локальный HuggingFace Transformers
    """

    def __init__(self, provider: str = None, model_name: str = None):
        self.rule_extractor = InformationExtractor()
        self.confidence_threshold = settings.llm_confidence_threshold

        # Создаём LLM-клиент через фабрику
        self.llm_client: LLMClientBase | None = None
        try:
            self.llm_client = create_llm_client(provider=provider)
            if not self.llm_client.is_available():
                logger.warning(
                    f"LLM provider '{provider or settings.llm_provider}' not available. "
                    "Rule-based extraction will be used as fallback."
                )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")

        self.model_name = model_name or settings.llm_model_name

        # Russian-language extraction prompt
        self.extraction_prompt = """Ты ассистент по извлечению генеалогических данных из русских метрических книг.
Извлеки структурированную информацию из следующего OCR-текста.
Верни ТОЛЬКО валидный JSON со следующими полями (используй "Unknown" если не найдено, confidence от 0.0 до 1.0):

Для ЗАПИСЕЙ О РОЖДЕНИИ:
{{
"record_type": "birth",
"child_name": {{"value": "...", "confidence": 0.0}},
"birth_date": {{"value": "YYYY-MM-DD", "confidence": 0.0}},
"baptism_date": {{"value": "YYYY-MM-DD или Unknown", "confidence": 0.0}},
"father_name": {{"value": "...", "confidence": 0.0}},
"mother_name": {{"value": "...", "confidence": 0.0}}
}}

Для ЗАПИСЕЙ О СМЕРТИ:
{{
"record_type": "death",
"deceased_name": {{"value": "...", "confidence": 0.0}},
"death_date": {{"value": "YYYY-MM-DD", "confidence": 0.0}},
"burial_date": {{"value": "YYYY-MM-DD или Unknown", "confidence": 0.0}},
"age": {{"value": "...", "confidence": 0.0}}
}}

Для ЗАПИСЕЙ О БРАКЕ:
{{
"record_type": "marriage",
"groom_name": {{"value": "...", "confidence": 0.0}},
"bride_name": {{"value": "...", "confidence": 0.0}},
"marriage_date": {{"value": "YYYY-MM-DD", "confidence": 0.0}}
}}

Нормализуй все даты в формат YYYY-MM-DD.
Учитывай дореволюционную орфографию (ѣ, і, ѳ, ъ).

Текст для анализа:
---
{text}
---
Верни ТОЛЬКО JSON объект, без дополнительного текста."""

        logger.info(
            f"LLM Extractor initialized "
            f"(provider={provider or settings.llm_provider}, "
            f"model={self.model_name})"
        )

    def _query_llm(self, prompt: str) -> str:
        """Отправить запрос к LLM через универсальный клиент."""
        if self.llm_client is None or not self.llm_client.is_available():
            logger.warning("LLM client not available, skipping LLM query")
            return ""
        try:
            return self.llm_client.query(prompt)
        except Exception as e:
            logger.warning(f"LLM query failed: {e}")
            return ""

    def _parse_llm_response(self, response: str) -> dict | None:
        """Parse LLM response to extract JSON. Handles truncated responses."""
        # Try complete JSON first
        json_pattern = r"\{[^{}]*\}"
        matches = re.findall(json_pattern, response, re.DOTALL)

        for match in matches:
            try:
                parsed = json.loads(match)
                if "record_type" in parsed:
                    return parsed
            except (json.JSONDecodeError, ValueError):
                continue

        # Try partial JSON recovery
        start_idx = response.find("{")
        if start_idx >= 0:
            partial = response[start_idx:]
            open_braces = partial.count("{")
            close_braces = partial.count("}")

            if open_braces > close_braces:
                partial += "}" * (open_braces - close_braces)

            try:
                parsed = json.loads(partial)
                if "record_type" in parsed:
                    logger.debug("Parsed truncated JSON")
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

            # Extract fields from partial JSON
            rt_match = re.search(r'"record_type"\s*:\s*"([^"]+)"', partial)
            if rt_match:
                result = {"record_type": rt_match.group(1)}
                for field in [
                    "child_name",
                    "father_name",
                    "mother_name",
                    "deceased_name",
                    "groom_name",
                    "bride_name",
                    "birth_date",
                    "death_date",
                    "marriage_date",
                ]:
                    vmatch = re.search(rf'"{field}"\s*:\s*{{"value"\s*:\s*"([^"]+)"', partial)
                    if vmatch:
                        result[field] = {"value": vmatch.group(1), "confidence": 0.5}
                if len(result) > 1:
                    logger.info("Extracted partial JSON from truncated response")
                    return result

        logger.warning("Could not parse LLM response as JSON")
        logger.debug(f"Raw response: {response[:300]}")
        return None

    def extract_with_llm(self, text: str) -> dict:
        """Извлечение данных через LLM."""
        prompt = self.extraction_prompt.format(text=text)
        response = self._query_llm(prompt)

        if not response:
            logger.warning("LLM empty response, falling back to rule-based")
            return self.rule_extractor.extract(text)

        result = self._parse_llm_response(response)

        if result is None:
            logger.warning("LLM parsing failed, falling back to rule-based")
            return self.rule_extractor.extract(text)

        result["_extraction"] = {
            "method": "llm",
            "model": self.model_name,
            "provider": settings.llm_provider,
            "source_length": len(text),
        }

        confidences = [
            v["confidence"] for k, v in result.items() if isinstance(v, dict) and "confidence" in v
        ]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        result["needs_review"] = avg_conf < 0.5

        return result

    def extract(self, text: str, force_llm: bool = False) -> dict:
        # Step 1: Rule-based extraction
        rule_result = self.rule_extractor.extract(text)
        rule_confidence = rule_result.get("_extraction", {}).get("average_confidence", 0.0)

        # Step 2: Check if LLM is needed
        if not force_llm and rule_confidence >= self.confidence_threshold:
            rule_result["_extraction"]["method"] = "rule_based"
            return rule_result

        # Step 3: Use LLM
        logger.info(
            f"Using LLM (rule-based confidence {rule_confidence:.2f} < {self.confidence_threshold})"
        )
        llm_result = self.extract_with_llm(text)

        # Step 4: Merge fields
        if not force_llm:
            for key in llm_result:
                if (
                    isinstance(llm_result.get(key), dict)
                    and llm_result[key].get("value") == "Unknown"
                ):
                    if (
                        isinstance(rule_result.get(key), dict)
                        and rule_result[key].get("value") != "Unknown"
                    ):
                        llm_result[key] = rule_result[key]

        return llm_result


def extract_with_llm(text: str, force_llm: bool = False, **kwargs) -> dict:
    extractor = LLMAssistedExtractor(**kwargs)
    return extractor.extract(text, force_llm=force_llm)


def smart_extract(text: str, **kwargs) -> dict:
    extractor = LLMAssistedExtractor(**kwargs)
    force = kwargs.pop("force_llm", False)
    return extractor.extract(text, force_llm=force)
