# Business logic: Information Extraction parses OCR text into structured genealogical data
# Extracts names, dates, relationships from Russian metrical book text into structured JSON
import re
from loguru import logger

from app.core.config import settings


class InformationExtractor:
    """Extracts structured genealogical data from Russian OCR text.
    Uses rule-based extraction with optional LLM fallback for low-confidence results.
    """

    def __init__(self):
        # ── Russian date patterns ──
        self.date_patterns = [
            (r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|"
             r"июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})(?:\s+года)?", "ru_dmy"),
            (r"(\d{1,2})\s+(генваря|февраля|марта|апреля|маiя|июня|"
             r"июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})", "ru_old"),
            (r"(\d{1,2})\s*[.\s]\s*(I{1,3}|IV|V|VI{0,3}|IX|X[I]?)\s*[.\s]\s*(\d{4})", "roman"),
            (r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", "numeric"),
            (r"(\d{4})-(\d{2})-(\d{2})", "iso"),
        ]

        self.record_indicators = {
            "birth": [
                ("родился", 4), ("родилась", 4), ("род", 3),
                ("крещён", 3), ("крещена", 3), ("крещение", 2),
                ("рождение", 2), ("рожд", 2), ("сын", 1), ("дочь", 1),
            ],
            "death": [
                ("умер", 4), ("умерла", 4), ("скончался", 3),
                ("скончалась", 3), ("погребён", 2), ("погребена", 2),
                ("похоронен", 2), ("смерть", 2), ("возраст", 1), ("лет", 1),
            ],
            "marriage": [
                ("венчался", 4), ("венчалась", 4), ("брак", 3),
                ("бракосочетание", 3), ("жених", 2), ("невеста", 2),
                ("супруг", 2), ("супруга", 2), ("муж", 1), ("жена", 1),
            ],
        }

        self.roman_map = {
            "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
            "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
        }

        self.month_map = {
            "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
            "мая": 5, "июня": 6, "июля": 7, "августа": 8,
            "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
            "генваря": 1, "маiя": 5,
        }

        # LLM extractor (lazy init)
        self._llm_extractor = None

        logger.info("Information Extractor initialized (Russian)")

    def _get_llm_extractor(self):
        """Lazy initialize LLM extractor."""
        if self._llm_extractor is None:
            try:
                from app.services.llm_extraction import LLMAssistedExtractor
                self._llm_extractor = LLMAssistedExtractor()
                logger.info("LLM extractor loaded")
            except Exception as e:
                logger.warning(f"LLM extractor not available: {e}")
                self._llm_extractor = False
        return self._llm_extractor if self._llm_extractor is not False else None

    def _detect_record_type(self, text: str) -> tuple:
        """Detect record type (birth/death/marriage) from text."""
        text_lower = text.lower()
        scores = {}
        for record_type, indicators in self.record_indicators.items():
            score = 0
            for keyword, weight in indicators:
                if keyword in text_lower:
                    score += weight
            scores[record_type] = score

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score > 0:
            return best_type, best_score
        return "unknown", 0

    def _extract_date(self, text: str) -> str:
        """Extract first valid date from text."""
        for pattern, fmt in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if fmt == "ru_dmy":
                    day, month, year = match.groups()
                    month_num = self.month_map.get(month.lower(), 1)
                    return f"{year}-{month_num:02d}-{int(day):02d}"
                elif fmt == "ru_old":
                    day, month, year = match.groups()
                    month_num = self.month_map.get(month.lower(), 1)
                    return f"{year}-{month_num:02d}-{int(day):02d}"
                elif fmt == "roman":
                    day, roman, year = match.groups()
                    month_num = self.roman_map.get(roman.upper(), 1)
                    return f"{year}-{month_num:02d}-{int(day):02d}"
                elif fmt == "numeric":
                    d1, d2, year = match.groups()
                    return f"{year}-{int(d2):02d}-{int(d1):02d}"
                elif fmt == "iso":
                    year, month, day = match.groups()
                    return f"{year}-{month}-{day}"
        return "Unknown"

    def _extract_name_after(self, text: str, keyword: str) -> str:
        """Extract name that appears after a keyword."""
        text_lower = text.lower()
        idx = text_lower.find(keyword)
        if idx == -1:
            return "Unknown"
        after = text[idx + len(keyword):].strip()
        # Take up to next punctuation or line break
        name_match = re.match(r"([А-Яа-яЁё\s\-]+)", after)
        if name_match:
            name = name_match.group(1).strip()
            parts = name.split()
            if len(parts) >= 1:
                return " ".join(parts[:3])  # max 3 words (first, patronymic, last)
        return "Unknown"

    def extract_birth_record(self, text: str) -> dict:
        """Extract birth record fields from text."""
        result = {
            "record_type": "birth",
            "child_name": {"value": "Unknown", "confidence": 0.0},
            "birth_date": {"value": "Unknown", "confidence": 0.0},
            "baptism_date": {"value": "Unknown", "confidence": 0.0},
            "father_name": {"value": "Unknown", "confidence": 0.0},
            "mother_name": {"value": "Unknown", "confidence": 0.0},
            "needs_review": True,
        }

        # Child name: look after "родился" or "родилась"
        child = self._extract_name_after(text, "родился")
        if child == "Unknown":
            child = self._extract_name_after(text, "родилась")
        if child == "Unknown":
            # Try "сын" or "дочь"
            for kw in ["сын", "дочь", "род"]:
                child = self._extract_name_after(text, kw)
                if child != "Unknown":
                    break
        if child != "Unknown":
            result["child_name"] = {"value": child, "confidence": 0.6}

        # Birth date
        date = self._extract_date(text)
        if date != "Unknown":
            result["birth_date"] = {"value": date, "confidence": 0.8}

        # Father name: look after "отец" or "от"
        father = self._extract_name_after(text, "отец")
        if father == "Unknown":
            father = self._extract_name_after(text, "от")
            if father and father.lower().startswith("ец"):
                father = "Unknown"
        if father != "Unknown":
            result["father_name"] = {"value": father, "confidence": 0.5}

        # Mother name: look after "мать" or "мат"
        mother = self._extract_name_after(text, "мать")
        if mother == "Unknown":
            mother = self._extract_name_after(text, "мат")
            if mother and len(mother) < 3:
                mother = "Unknown"
        if mother != "Unknown":
            result["mother_name"] = {"value": mother, "confidence": 0.5}

        logger.debug(f"Extracted birth: {result['child_name']['value']}")
        return result

    def extract_death_record(self, text: str) -> dict:
        """Extract death record fields from text."""
        result = {
            "record_type": "death",
            "deceased_name": {"value": "Unknown", "confidence": 0.0},
            "death_date": {"value": "Unknown", "confidence": 0.0},
            "burial_date": {"value": "Unknown", "confidence": 0.0},
            "age": None,
            "needs_review": True,
        }

        # Deceased name: look after "умер" or "скончался"
        deceased = self._extract_name_after(text, "умер")
        if deceased == "Unknown":
            deceased = self._extract_name_after(text, "скончался")
        if deceased != "Unknown":
            result["deceased_name"] = {"value": deceased, "confidence": 0.6}

        # Death date
        date = self._extract_date(text)
        if date != "Unknown":
            result["death_date"] = {"value": date, "confidence": 0.8}

        # Age
        age_match = re.search(r"(\d+)\s+лет", text.lower())
        if age_match:
            result["age"] = {"value": age_match.group(1), "confidence": 0.7}

        logger.debug(f"Extracted death: {result['deceased_name']['value']}")
        return result

    def extract_marriage_record(self, text: str) -> dict:
        """Extract marriage record fields from text."""
        result = {
            "record_type": "marriage",
            "groom_name": {"value": "Unknown", "confidence": 0.0},
            "bride_name": {"value": "Unknown", "confidence": 0.0},
            "marriage_date": {"value": "Unknown", "confidence": 0.0},
            "needs_review": True,
        }

        # Groom: look after "жених" or "венчался"
        groom = self._extract_name_after(text, "жених")
        if groom == "Unknown":
            groom = self._extract_name_after(text, "венчался")
        if groom != "Unknown":
            result["groom_name"] = {"value": groom, "confidence": 0.6}

        # Bride: look after "невеста"
        bride = self._extract_name_after(text, "невеста")
        if bride == "Unknown":
            bride = self._extract_name_after(text, "венчалась")
        if bride != "Unknown":
            result["bride_name"] = {"value": bride, "confidence": 0.6}

        # Marriage date
        date = self._extract_date(text)
        if date != "Unknown":
            result["marriage_date"] = {"value": date, "confidence": 0.8}

        logger.debug(f"Extracted marriage: {result['groom_name']['value']} & {result['bride_name']['value']}")
        return result

    def extract(self, text: str, force_llm: bool = False) -> dict:
        """Extract structured data from OCR text.
        
        Strategy:
        1. Detect record type (birth/death/marriage)
        2. Run rule-based extraction
        3. If confidence is low, try LLM fallback
        4. Merge results
        5. Compute metadata
        
        Returns:
            dict with extracted fields and metadata
        """
        if not text or not text.strip():
            return {
                "record_type": "unknown",
                "needs_review": True,
                "_extraction": {
                    "average_confidence": 0.0,
                    "source_length": 0,
                    "method": "empty_input",
                    "language": "ru",
                },
            }

        # Step 1: Detect record type
        record_type, type_confidence = self._detect_record_type(text)

        # Step 2: Rule-based extraction
        if record_type == "birth":
            rule_result = self.extract_birth_record(text)
        elif record_type == "death":
            rule_result = self.extract_death_record(text)
        elif record_type == "marriage":
            rule_result = self.extract_marriage_record(text)
        else:
            rule_result = {
                "record_type": "unknown",
                "needs_review": True,
            }

        # Step 3: Compute rule-based confidence
        confidences = [
            v["confidence"]
            for k, v in rule_result.items()
            if isinstance(v, dict) and "confidence" in v
        ]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        rule_result["_extraction"] = {
            "average_confidence": round(avg_conf, 2),
            "source_length": len(text),
            "method": "rule_based",
            "language": "ru",
        }

        # Step 4: If confidence low, try LLM
        llm_extractor = self._get_llm_extractor()
        if llm_extractor and (force_llm or avg_conf < settings.ocr_confidence_threshold):
            logger.info(
                f"Trying LLM (rule confidence {avg_conf:.2f} < {settings.ocr_confidence_threshold})"
            )
            try:
                llm_result = llm_extractor.extract(text, force_llm=force_llm)
                # Merge: prefer LLM results for fields that are "Unknown"
                for key in llm_result:
                    if (
                        isinstance(llm_result.get(key), dict)
                        and llm_result[key].get("value", "Unknown") != "Unknown"
                    ):
                        if (
                            isinstance(rule_result.get(key), dict)
                            and rule_result[key].get("value") == "Unknown"
                        ):
                            rule_result[key] = llm_result[key]
                            logger.debug(f"LLM improved field '{key}': {llm_result[key]['value']}")
                rule_result["_extraction"]["method"] = "hybrid"
                rule_result["_extraction"]["llm_used"] = True
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")

        # Step 5: Set needs_review
        rule_result["needs_review"] = avg_conf < 0.5 or rule_result.get("needs_review", True)

        return rule_result


def extract_information(text: str, force_llm: bool = False, **kwargs) -> dict:
    """Quick extraction of information from OCR text.
    
    Args:
        text: Post-processed OCR text
        force_llm: Force LLM usage even if rule-based confidence is high
        **kwargs: Additional arguments
        
    Returns:
        dict with extracted fields
    """
    extractor = InformationExtractor()
    return extractor.extract(text, force_llm=force_llm)
