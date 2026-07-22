# Business logic: Information Extraction parses OCR text into structured genealogical data
# Extracts names, dates, relationships from Russian metrical book text into structured JSON
import re

from loguru import logger


class InformationExtractor:
    """Extracts structured genealogical data from Russian OCR text."""

    def __init__(self):
        # ── Russian date patterns ──
        self.date_patterns = [
            # "12 марта 1878" or "12 марта 1878 года"
            (
                r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|"
                r"июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})(?:\s+года)?",
                "ru_dmy",
            ),
            # Pre-1918: "12 генваря 1878" or "12 маiя 1878"
            (
                r"(\d{1,2})\s+(генваря|февраля|марта|апреля|маiя|июня|"
                r"июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})",
                "ru_old",
            ),
            # Roman numeral months: "12.III.1878" or "12 III 1878"
            (r"(\d{1,2})\s*[.\s]\s*(I{1,3}|IV|V|VI{0,3}|IX|X[I]?)\s*[.\s]\s*(\d{4})", "roman"),
            # Numeric: "12.03.1878" or "12/03/1878"
            (r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})", "numeric"),
            # ISO: "1878-03-12"
            (r"(\d{4})-(\d{2})-(\d{2})", "iso"),
        ]

        # ── Russian record type indicators ──
        self.record_indicators = {
            "birth": [
                ("родился", 4),
                ("родилась", 4),
                ("род", 3),
                ("крещён", 3),
                ("крещена", 3),
                ("крещение", 2),
                ("рождение", 2),
                ("рожд", 2),
                ("сын", 1),
                ("дочь", 1),
            ],
            "death": [
                ("умер", 4),
                ("умерла", 4),
                ("скончался", 3),
                ("скончалась", 3),
                ("погребён", 2),
                ("погребена", 2),
                ("похоронен", 2),
                ("смерть", 2),
                ("возраст", 1),
                ("лет", 1),
            ],
            "marriage": [
                ("венчался", 4),
                ("венчалась", 4),
                ("брак", 3),
                ("жених", 2),
                ("невеста", 2),
                ("супруг", 2),
                ("супруга", 2),
                ("женитьба", 2),
                ("бракосочетание", 2),
            ],
        }

        # ── Russian extraction patterns ──
        self.birth_patterns = {
            "child_name": [
                r"(?:родился|родилась|род|р\.)\s+([А-Я][а-яё]+\s+[А-Я][а-яё]+)",
                r"([А-Я][а-яё]+\s+[А-Я][а-яё]+)\s+родил[ся|сь]",
                r"^([А-Я][а-яё]+\s+[А-Я][а-яё]+)",
            ],
            "father_name": [
                r"(?:отец|от|о\.)\s*:?\s*([А-Я][а-яё]+\s+[А-Я][а-яё]+)",
                r"(?:сын|дочь)\s+([А-Я][а-яё]+\s+[А-Я][а-яё]+)",
            ],
            "mother_name": [
                r"(?:мать|мат|м\.)\s*:?\s*([А-Я][а-яё]+\s+[А-Я][а-яё]+)",
            ],
        }

        self.death_patterns = {
            "deceased_name": [
                r"(?:умер|умерла|скончался|скончалась|ум\.|у\.)\s+([А-Я][а-яё]+\s+[А-Я][а-яё]+)",
                r"([А-Я][а-яё]+\s+[А-Я][а-яё]+)\s+умер",
                r"^([А-Я][а-яё]+\s+[А-Я][а-яё]+)",
            ],
        }

        self.marriage_patterns = {
            "groom_name": [
                r"(?:жених|венчался)\s*:?\s*([А-Я][а-яё]+\s+[А-Я][а-яё]+)",
                r"([А-Я][а-яё]+\s+[А-Я][а-яё]+)\s+венчал[ся|сь]",
            ],
            "bride_name": [
                r"(?:невеста|венчалась)\s*:?\s*([А-Я][а-яё]+\s+[А-Я][а-яё]+)",
                r"венчал[ся|сь]\s+с\s+([А-Я][а-яё]+\s+[А-Я][а-яё]+)",
            ],
        }

        # ── Month mapping ──
        self.month_map = {
            # Modern Russian (genitive)
            "января": "01",
            "февраля": "02",
            "марта": "03",
            "апреля": "04",
            "мая": "05",
            "июня": "06",
            "июля": "07",
            "августа": "08",
            "сентября": "09",
            "октября": "10",
            "ноября": "11",
            "декабря": "12",
            # Pre-1918
            "генваря": "01",
            "маiя": "05",
        }

        # Roman numeral to month
        self.roman_months = {
            "I": 1,
            "II": 2,
            "III": 3,
            "IV": 4,
            "V": 5,
            "VI": 6,
            "VII": 7,
            "VIII": 8,
            "IX": 9,
            "X": 10,
            "XI": 11,
            "XII": 12,
        }

        logger.info("Information Extractor initialized (Russian)")

    def classify_record_type(self, text: str) -> str:
        """Determine if text describes a birth, death, or marriage record."""
        text_lower = text.lower()

        scores = {"birth": 0, "death": 0, "marriage": 0}

        for record_type, indicators in self.record_indicators.items():
            for word, weight in indicators:
                if word in text_lower:
                    scores[record_type] += weight

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score == 0:
            return "birth"

        # If tie, prefer birth
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
            if "родился" in text_lower or "родилась" in text_lower:
                return "birth"
            return "birth"

        logger.info(f"Record type: '{best_type}' (scores: {scores})")
        return best_type

    def normalize_date(self, match, format_type: str) -> tuple:
        groups = match.groups()

        if format_type == "ru_dmy":
            day, month_name, year = groups
            month = self.month_map.get(month_name.lower(), "00")
            return f"{year}-{month}-{day.zfill(2)}", 0.9

        elif format_type == "ru_old":
            day, month_name, year = groups
            month = self.month_map.get(month_name.lower(), "00")
            return f"{year}-{month}-{day.zfill(2)}", 0.85

        elif format_type == "roman":
            day, roman_month, year = groups
            month_num = str(self.roman_months.get(roman_month.upper(), 0)).zfill(2)
            return f"{year}-{month_num}-{day.zfill(2)}", 0.85

        elif format_type == "numeric":
            a, b, c = groups
            if int(a) > 12:
                return f"{c}-{b.zfill(2)}-{a.zfill(2)}", 0.8
            elif int(b) > 12:
                return f"{c}-{a.zfill(2)}-{b.zfill(2)}", 0.8
            else:
                return f"{c}-{b.zfill(2)}-{a.zfill(2)}", 0.7

        elif format_type == "iso":
            year, month, day = groups
            return f"{year}-{month}-{day}", 0.95

        return None, 0.0

    def extract_date(self, text: str) -> tuple:
        for pattern, format_type in self.date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self.normalize_date(match, format_type)
        return None, 0.0

    def extract_all_dates(self, text: str) -> list:
        dates = []
        for pattern, format_type in self.date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_str, conf = self.normalize_date(match, format_type)
                if date_str:
                    dates.append((date_str, conf))
        return dates

    def extract_field(self, text: str, patterns: list, use_multiline: bool = False) -> tuple:
        flags = re.MULTILINE | re.IGNORECASE if use_multiline else re.IGNORECASE

        for i, pattern in enumerate(patterns):
            match = re.search(pattern, text, flags)
            if match:
                value = match.group(1).strip()
                value = re.sub(r"\s+", " ", value)
                confidence = max(0.5, 0.9 - (i * 0.1))
                return value, round(confidence, 4)

        return None, 0.0

    def extract_birth_record(self, text: str) -> dict:
        child_name, child_conf = self.extract_field(
            text, self.birth_patterns["child_name"], use_multiline=True
        )
        father_name, father_conf = self.extract_field(text, self.birth_patterns["father_name"])
        mother_name, mother_conf = self.extract_field(text, self.birth_patterns["mother_name"])

        all_dates = self.extract_all_dates(text)
        birth_date, date_conf = all_dates[0] if all_dates else (None, 0.0)

        # Baptism date only if no death keywords
        baptism_date = None
        baptism_conf = 0.0
        if len(all_dates) > 1 and "умер" not in text.lower():
            baptism_date, baptism_conf = all_dates[1]

        record = {
            "record_type": "birth",
            "child_name": {"value": child_name or "Unknown", "confidence": child_conf},
            "birth_date": {"value": birth_date or "Unknown", "confidence": date_conf},
            "baptism_date": {"value": baptism_date or "Unknown", "confidence": baptism_conf},
            "father_name": {"value": father_name or "Unknown", "confidence": father_conf},
            "mother_name": {"value": mother_name or "Unknown", "confidence": mother_conf},
        }

        logger.info(f"Extracted birth: {child_name or 'Unknown'}")
        return record

    def extract_death_record(self, text: str) -> dict:
        deceased_name, dead_conf = self.extract_field(
            text, self.death_patterns["deceased_name"], use_multiline=True
        )

        all_dates = self.extract_all_dates(text)

        death_date = None
        date_conf = 0.0
        burial_date = None
        burial_conf = 0.0

        if len(all_dates) >= 2:
            death_date, date_conf = all_dates[1]
        elif len(all_dates) == 1:
            death_date, date_conf = all_dates[0]

        if len(all_dates) >= 3:
            burial_date, burial_conf = all_dates[2]

        # Extract age
        age_match = re.search(r"(?:возраст|лет)\s+(\d+)", text, re.IGNORECASE)
        age = age_match.group(1) if age_match else None

        record = {
            "record_type": "death",
            "deceased_name": {"value": deceased_name or "Unknown", "confidence": dead_conf},
            "death_date": {"value": death_date or "Unknown", "confidence": date_conf},
            "burial_date": {"value": burial_date or "Unknown", "confidence": burial_conf},
        }

        if age:
            record["age"] = {"value": age, "confidence": 0.8}

        logger.info(f"Extracted death: {deceased_name or 'Unknown'}")
        return record

    def extract_marriage_record(self, text: str) -> dict:
        groom_name, groom_conf = self.extract_field(text, self.marriage_patterns["groom_name"])
        bride_name, bride_conf = self.extract_field(text, self.marriage_patterns["bride_name"])
        all_dates = self.extract_all_dates(text)
        marriage_date, date_conf = all_dates[0] if all_dates else (None, 0.0)

        record = {
            "record_type": "marriage",
            "groom_name": {"value": groom_name or "Unknown", "confidence": groom_conf},
            "bride_name": {"value": bride_name or "Unknown", "confidence": bride_conf},
            "marriage_date": {"value": marriage_date or "Unknown", "confidence": date_conf},
        }

        logger.info(f"Extracted marriage: {groom_name or 'Unknown'} & {bride_name or 'Unknown'}")
        return record

    def _normalize_date(self, date_str: str) -> str:
        """Normalize various Russian date formats to YYYY-MM-DD.

        Handles:
        - "12 марта 1878" → "1878-03-12"
        - "12.III.1878" → "1878-03-12"
        - "12/03/1878" → "1878-03-12"
        - "1878-03-12" → "1878-03-12" (already normalized)
        """
        if not date_str or date_str == "Unknown":
            return "Unknown"

        # Already in YYYY-MM-DD format
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            return date_str

        # Russian month names → numbers
        month_map = {
            "января": 1,
            "февраля": 2,
            "марта": 3,
            "апреля": 4,
            "мая": 5,
            "июня": 6,
            "июля": 7,
            "августа": 8,
            "сентября": 9,
            "октября": 10,
            "ноября": 11,
            "декабря": 12,
        }

        # Roman numerals
        roman_map = {
            "I": 1,
            "II": 2,
            "III": 3,
            "IV": 4,
            "V": 5,
            "VI": 6,
            "VII": 7,
            "VIII": 8,
            "IX": 9,
            "X": 10,
            "XI": 11,
            "XII": 12,
        }

        try:
            # Pattern: "12 марта 1878"
            for month_name, month_num in month_map.items():
                if month_name in date_str.lower():
                    match = re.search(
                        r"(\d{1,2})\s+" + month_name + r"\s+(\d{4})", date_str, re.IGNORECASE
                    )
                    if match:
                        day, year = int(match.group(1)), match.group(2)
                        return f"{year}-{month_num:02d}-{day:02d}"

            # Pattern: "12.III.1878"
            for roman, month_num in roman_map.items():
                if f".{roman}." in date_str.upper() or f".{roman}" in date_str.upper():
                    match = re.search(
                        r"(\d{1,2})\." + roman + r"\.?(\d{4})", date_str, re.IGNORECASE
                    )
                    if match:
                        day, year = int(match.group(1)), match.group(2)
                        return f"{year}-{month_num:02d}-{day:02d}"

            # Pattern: "12/03/1878"
            match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str)
            if match:
                day, month, year = int(match.group(1)), int(match.group(2)), match.group(3)
                return f"{year}-{month:02d}-{day:02d}"

            # Pattern: "1878"
            match = re.search(r"\b(\d{4})\b", date_str)
            if match:
                return f"{match.group(1)}-01-01"

        except (ValueError, AttributeError):
            pass

        logger.warning(f"Could not normalize date: {date_str}")
        return "Unknown"

    def compute_confidence(self, record: dict) -> float:
        confidences = []
        for key, value in record.items():
            if isinstance(value, dict) and "confidence" in value:
                confidences.append(value["confidence"])
        return round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    def extract(self, text: str) -> dict:
        """Main extraction method for Russian texts."""
        record_type = self.classify_record_type(text)

        if record_type == "birth":
            record = self.extract_birth_record(text)
        elif record_type == "death":
            record = self.extract_death_record(text)
        elif record_type == "marriage":
            record = self.extract_marriage_record(text)
        else:
            record = self.extract_birth_record(text)
            record["record_type"] = "unknown"

        avg_confidence = self.compute_confidence(record)
        record["needs_review"] = avg_confidence < 0.5
        record["_extraction"] = {
            "average_confidence": avg_confidence,
            "source_length": len(text),
            "language": "ru",
        }

        logger.info(
            f"Extraction: {record['record_type']} (avg={avg_confidence:.2f}, review={record['needs_review']})"
        )
        return record


def extract_information(text: str) -> dict:
    extractor = InformationExtractor()
    return extractor.extract(text)
