# Business logic: OCR post-processing cleans up raw OCR output
# Fixes common OCR errors, corrects spelling, restores structure
# Specialized for: Russian (including pre-1918 orthography)
import re

from loguru import logger


class OCRPostProcessor:
    """Post-processes raw OCR text to improve quality.

    Specialized for Russian historical texts with:
    1. Dictionary-based spelling correction
    2. Pre-1918 orthography normalization (ѣ, і, ѳ, ѵ, ъ)
    3. Abbreviation expansion (р., у., к., etc.)
    4. Roman numeral date normalization
    """

    def __init__(self):
        # ── Russian genealogical terms dictionary ──
        self.genealogy_dict = {
            # Russian months (genitive case — as they appear in records)
            "января": "январь",
            "февраля": "февраль",
            "марта": "март",
            "апреля": "апрель",
            "мая": "май",
            "июня": "июнь",
            "июля": "июль",
            "августа": "август",
            "сентября": "сентябрь",
            "октября": "октябрь",
            "ноября": "ноябрь",
            "декабря": "декабрь",
            # Russian genealogical terms
            "родился": "родился",
            "родилась": "родилась",
            "родившись": "родился",
            "умер": "умер",
            "умерла": "умерла",
            "скончался": "скончался",
            "скончалась": "скончалась",
            "крещён": "крещён",
            "крещена": "крещена",
            "венчался": "венчался",
            "венчалась": "венчалась",
            "погребён": "погребён",
            "погребена": "погребена",
            "похоронен": "похоронен",
            "похоронена": "похоронена",
            # Russian relationships
            "сын": "сын",
            "дочь": "дочь",
            "отец": "отец",
            "от": "отец",
            "мать": "мать",
            "мат": "мать",
            "муж": "муж",
            "жена": "жена",
            "супруг": "супруг",
            "супруга": "супруга",
            "брак": "брак",
            "бракосочетание": "бракосочетание",
            # Russian common names
            "иван": "Иван",
            "анна": "Анна",
            "мария": "Мария",
            "пётр": "Пётр",
            "петр": "Пётр",
            "александр": "Александр",
            "алексей": "Алексей",
            "екатерина": "Екатерина",
            "елена": "Елена",
            "ольга": "Ольга",
            "татьяна": "Татьяна",
            "надежда": "Надежда",
            "николай": "Николай",
            "михаил": "Михаил",
            "владимир": "Владимир",
            "дмитрий": "Дмитрий",
            "сергей": "Сергей",
            # Russian common surnames
            "иванов": "Иванов",
            "петров": "Петров",
            "сидоров": "Сидоров",
            "кузнецов": "Кузнецов",
            "смирнов": "Смирнов",
            "попов": "Попов",
            "васильев": "Васильев",
            "зайцев": "Зайцев",
            "соколов": "Соколов",
            "михайлов": "Михайлов",
            "фёдоров": "Фёдоров",
            "федоров": "Фёдоров",
            "белов": "Белов",
            "козлов": "Козлов",
            "новиков": "Новиков",
            "морозов": "Морозов",
            "волков": "Волков",
        }

        # ── Date patterns ──
        # Russian: "12 марта 1878" or "12 марта 1878 года"
        self.date_pattern_ru = re.compile(
            r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|"
            r"июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})(?:\s+года)?",
            re.IGNORECASE,
        )

        # Roman numeral months: "12.III.1878" or "12 III 1878"
        self.date_pattern_roman = re.compile(
            r"(\d{1,2})\s*[.\s]\s*(I{1,3}|IV|V|VI{0,3}|IX|X[I]?)\s*[.\s]\s*(\d{4})"
        )

        # Year pattern
        self.year_pattern = re.compile(r"\b(1[4-9]\d{2}|20[0-2]\d)\b")

        # Roman numeral map
        self.roman_map = {
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

        logger.info("OCR Post-Processor initialized (Russian)")

    def correct_spelling(self, text: str) -> str:
        """Apply dictionary-based spelling correction for Russian terms."""
        words = text.split()
        corrected_words = []

        for word in words:
            clean_word = word.strip(".,:;!?\"'()[]{}")
            punctuation = word[len(clean_word) :] if len(clean_word) < len(word) else ""

            # Check dictionary (case-insensitive)
            corrected = self.genealogy_dict.get(clean_word.lower(), clean_word)

            # Preserve capitalization
            if clean_word.istitle() and corrected.islower():
                corrected = corrected.capitalize()
            elif clean_word.isupper() and corrected.islower():
                corrected = corrected.upper()

            corrected_words.append(corrected + punctuation)

        return " ".join(corrected_words)

    def normalize_dates(self, text: str) -> str:
        """Normalize Russian dates and Roman numeral dates."""

        # Normalize Russian dates
        def format_ru_date(match):
            day, month, year = match.groups()
            return f"{day} {month} {year}"

        text = self.date_pattern_ru.sub(format_ru_date, text)

        # Normalize Roman numeral dates → numeric
        def format_roman_date(match):
            day, roman, year = match.groups()
            month_num = self.roman_map.get(roman.upper(), 0)
            return f"{day}.{month_num:02d}.{year}"

        text = self.date_pattern_roman.sub(format_roman_date, text)

        return text

    def merge_hyphenated_words(self, text: str) -> str:
        """Merge words split by hyphens across lines."""
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
        text = re.sub(
            r"(\w+)-\s+(\w+)",
            lambda m: m.group(1) + m.group(2) if len(m.group(1)) > 2 else m.group(0),
            text,
        )
        return text

    def fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors for Russian texts."""
        fixes = [
            # Common Cyrillic OCR confusions
            (r"р[оа]дился", "родился"),
            (r"ум[еэ]р", "умер"),
            (r"крещ[её]н", "крещён"),
            (r"венч[ао]лся", "венчался"),
            (r"погр[её]б[её]н", "погребён"),
            # Pre-1918 orthography normalization
            (r"ѣ", "е"),  # Yat → е
            (r"і", "и"),  # i decimal → и
            (r"ѳ", "ф"),  # Fita → ф
            (r"ѵ", "и"),  # Izhitsa → и
            # Remove hard sign (ъ) at end of words (pre-1918 convention)
            (r"(\w+)ъ\b", r"\1"),
            # Clean up whitespace and punctuation
            (r"\s+\.", "."),
            (r"\.\s+\.", "."),
            (r"\s{2,}", " "),
        ]

        for pattern, replacement in fixes:
            text = re.sub(pattern, replacement, text)

        return text

    def expand_abbreviations(self, text: str) -> str:
        """Expand common Russian abbreviations found in metrical books."""
        expansions = [
            (r"\bр\.\s", "родился "),
            (r"\bрод\.\s", "родился "),
            (r"\bу\.\s", "умер "),
            (r"\bум\.\s", "умер "),
            (r"\bк\.\s", "крещён "),
            (r"\bкрещ\.\s", "крещён "),
            (r"\bвенч\.\s", "венчался "),
            (r"\bпогр\.\s", "погребён "),
            (r"\bот\.\s", "отец "),
            (r"\bмат\.\s", "мать "),
            (r"\bг\.\s", "год "),
            (r"\bгуб\.\s", "губерния "),
            (r"\bу\.\b", "уезд "),
            (r"\bвол\.\s", "волость "),
            (r"\bсвящ\.\s", "священник "),
            (r"\bдьяк\.\s", "дьякон "),
            (r"\bпис\.\s", "писарь "),
        ]

        for pattern, replacement in expansions:
            text = re.sub(pattern, replacement, text)

        return text

    def preserve_line_breaks(self, text: str, max_line_length: int = 80) -> str:
        """Restore sensible line breaks for readability."""
        lines = text.split("\n")
        formatted_lines = []

        for line in lines:
            if len(line) <= max_line_length:
                formatted_lines.append(line)
            else:
                words = line.split()
                current_line = []
                current_length = 0

                for word in words:
                    if current_length + len(word) + 1 > max_line_length and current_line:
                        formatted_lines.append(" ".join(current_line))
                        current_line = [word]
                        current_length = len(word)
                    else:
                        current_line.append(word)
                        current_length += len(word) + 1

                if current_line:
                    formatted_lines.append(" ".join(current_line))

        return "\n".join(formatted_lines)

    def process(self, text: str, preserve_lines: bool = True) -> dict:
        """Run full post-processing pipeline on raw OCR text.

        Args:
            text: Raw OCR text from TrOCR/Tesseract
            preserve_lines: Whether to format output with line breaks

        Returns:
            dict with corrected text and metadata
        """
        corrections_applied = []

        # Step 1: Merge hyphenated words
        text_after = self.merge_hyphenated_words(text)
        if text_after != text:
            corrections_applied.append("hyphenated_words_merged")
        text = text_after

        # Step 2: Expand abbreviations
        text_after = self.expand_abbreviations(text)
        if text_after != text:
            corrections_applied.append("abbreviations_expanded")
        text = text_after

        # Step 3: Fix common OCR errors
        text_after = self.fix_common_ocr_errors(text)
        if text_after != text:
            corrections_applied.append("common_ocr_fixes")
        text = text_after

        # Step 4: Spelling correction
        text_after = self.correct_spelling(text)
        if text_after != text:
            corrections_applied.append("spelling_correction")
        text = text_after

        # Step 5: Normalize dates
        text_after = self.normalize_dates(text)
        if text_after != text:
            corrections_applied.append("date_normalization")
        text = text_after

        # Step 6: Clean up whitespace
        text_after = re.sub(r"\s+", " ", text).strip()
        if text_after != text:
            corrections_applied.append("whitespace_cleanup")
        text = text_after

        # Step 7: Preserve line breaks
        if preserve_lines:
            text = self.preserve_line_breaks(text)

        logger.info(f"Post-processing: {len(corrections_applied)} corrections applied")

        return {
            "original_text": text_after if not preserve_lines else text,
            "corrected_text": text,
            "corrections_applied": corrections_applied,
        }


def postprocess_ocr_text(text: str, **kwargs) -> dict:
    """Quick post-processing of OCR text.

    Args:
        text: Raw OCR text
        **kwargs: Passed to OCRPostProcessor.process()

    Returns:
        dict with corrected text
    """
    processor = OCRPostProcessor()
    return processor.process(text, **kwargs)
