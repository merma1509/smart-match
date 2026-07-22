# Business logic: Entity Resolution normalizes and validates extracted genealogical data
# Handles Russian name variants, date references, age extraction, and family linking
import re
from datetime import datetime

from loguru import logger


class EntityResolver:
    """Resolves and normalizes entities extracted from Russian genealogical records.

    Features:
    1. Russian name variant normalization (Иван → Иоанн → Иван)
    2. Orthodox calendar date references (Пасха, Рождество → actual dates)
    3. Age extraction and validation against birth/death dates
    4. Family member linking across records
    """

    def __init__(self):
        # ── Russian name variants ──
        self.name_variants = {
            # Male names
            "иван": ["Иван", "Иоанн", "Ivan", "Ваня"],
            "иоанн": ["Иоанн", "Иван", "Ivan"],
            "пётр": ["Пётр", "Петр", "Peter", "Петя"],
            "петр": ["Пётр", "Петр", "Peter"],
            "александр": ["Александр", "Alexander", "Саша"],
            "алексей": ["Алексей", "Alexei", "Лёша"],
            "михаил": ["Михаил", "Mikhail", "Миша"],
            "николай": ["Николай", "Nikolai", "Коля"],
            "владимир": ["Владимир", "Vladimir", "Вова"],
            "дмитрий": ["Дмитрий", "Dmitry", "Дима"],
            "сергей": ["Сергей", "Sergei", "Серёжа"],
            "константин": ["Константин", "Konstantin", "Костя"],
            "василий": ["Василий", "Vasily", "Вася"],
            "павел": ["Павел", "Pavel", "Паша"],
            "степан": ["Степан", "Stepan", "Стёпа"],
            "григорий": ["Григорий", "Grigory", "Гриша"],
            "фёдор": ["Фёдор", "Федор", "Fedor"],
            "яков": ["Яков", "Yakov"],
            "евгений": ["Евгений", "Evgeny"],
            "семён": ["Семён", "Семен", "Semyon"],
            # Female names
            "мария": ["Мария", "Maria", "Mary", "Маша"],
            "анна": ["Анна", "Anna", "Anne", "Аня"],
            "екатерина": ["Екатерина", "Ekaterina", "Catherine", "Катя"],
            "елена": ["Елена", "Elena", "Helen", "Лена"],
            "ольга": ["Ольга", "Olga", "Оля"],
            "татьяна": ["Татьяна", "Tatiana", "Таня"],
            "надежда": ["Надежда", "Nadezhda", "Надя"],
            "любовь": ["Любовь", "Lyubov", "Люба"],
            "вера": ["Вера", "Vera"],
            "настасья": ["Настасья", "Анастасия", "Anastasia"],
            "анастасия": ["Анастасия", "Anastasia", "Настя"],
            "ирина": ["Ирина", "Irina"],
            "александра": ["Александра", "Alexandra", "Саша"],
            "софья": ["Софья", "София", "Sofia"],
            # Patronymics
            "иванович": ["Иванович"],
            "ивановна": ["Ивановна"],
            "петрович": ["Петрович"],
            "петровна": ["Петровна"],
            "александрович": ["Александрович"],
            "александровна": ["Александровна"],
            "михайлович": ["Михайлович"],
            "михайловна": ["Михайловна"],
            "николаевич": ["Николаевич"],
            "николаевна": ["Николаевна"],
            # Common surnames
            "иванов": ["Иванов", "Ivanov"],
            "петров": ["Петров", "Petrov"],
            "сидоров": ["Сидоров", "Sidorov"],
            "кузнецов": ["Кузнецов", "Kuznetsov"],
            "смирнов": ["Смирнов", "Smirnov"],
            "попов": ["Попов", "Popov"],
            "васильев": ["Васильев", "Vasiliev"],
            "зайцев": ["Зайцев", "Zaitsev"],
            "соколов": ["Соколов", "Sokolov"],
            "михайлов": ["Михайлов", "Mikhailov"],
            "фёдоров": ["Фёдоров", "Федоров", "Fedorov"],
            "белов": ["Белов", "Belov"],
            "козлов": ["Козлов", "Kozlov"],
            "новиков": ["Новиков", "Novikov"],
            "морозов": ["Морозов", "Morozov"],
            "волков": ["Волков", "Volkov"],
        }

        # ── Russian Orthodox calendar dates ──
        self.historical_dates = {
            # Fixed Orthodox holidays (Julian calendar dates)
            "пасха": "movable",  # Easter — computed
            "рождество": "12-25",  # Christmas (Julian: Dec 25)
            "рождество христово": "12-25",
            "крещение": "01-06",  # Epiphany (Julian: Jan 6)
            "богоявление": "01-06",
            "сретение": "02-02",  # Candlemas (Julian: Feb 2)
            "благовещение": "03-25",  # Annunciation (Julian: Mar 25)
            "вербное воскресенье": "movable",  # Palm Sunday — before Easter
            "вознесение": "movable",  # Ascension — 40 days after Easter
            "троица": "movable",  # Pentecost — 50 days after Easter
            "преображение": "08-06",  # Transfiguration (Julian: Aug 6)
            "успение": "08-15",  # Assumption (Julian: Aug 15)
            "рождество богородицы": "09-08",  # Nativity of Theotokos (Julian: Sep 8)
            "воздвижение": "09-14",  # Exaltation of Cross (Julian: Sep 14)
            "покров": "10-01",  # Intercession (Julian: Oct 1)
            "введение": "11-21",  # Presentation (Julian: Nov 21)
            "михайлов день": "11-08",  # St. Michael's Day (Julian: Nov 8)
            "николин день": "12-06",  # St. Nicholas Day (Julian: Dec 6)
            "никола зимний": "12-06",
            "никола вешний": "05-09",  # St. Nicholas Summer (Julian: May 9)
            "илин день": "07-20",  # St. Elijah's Day (Julian: Jul 20)
            "петров день": "06-29",  # Sts. Peter & Paul (Julian: Jun 29)
            "иванов день": "06-24",  # John the Baptist (Julian: Jun 24)
            # Approximate references
            "около": "circa",
            "приблизительно": "circa",
            "примерно": "circa",
        }

        # Month mapping
        self.month_map = {
            "январь": 1,
            "февраль": 2,
            "март": 3,
            "апрель": 4,
            "май": 5,
            "июнь": 6,
            "июль": 7,
            "август": 8,
            "сентябрь": 9,
            "октябрь": 10,
            "ноябрь": 11,
            "декабрь": 12,
        }

        logger.info("Entity Resolver initialized (Russian)")

    def normalize_name(self, name: str) -> dict:
        """Normalize a Russian name to its canonical form.

        Args:
            name: Raw name string (e.g., "Иван Петров")

        Returns:
            dict with canonical form and variants
        """
        if not name or name == "Unknown":
            return {
                "original": name,
                "canonical": name,
                "variants": [],
                "first_name": "",
                "last_name": "",
                "confidence": 0.0,
            }

        parts = name.strip().split()
        if len(parts) == 1:
            first_name = parts[0]
            last_name = ""
        else:
            first_name = parts[0]
            last_name = " ".join(parts[1:])

        # Normalize first name
        first_lower = first_name.lower()
        canonical_first = first_name
        variants = []

        if first_lower in self.name_variants:
            variants = self.name_variants[first_lower]
            canonical_first = variants[0]
        else:
            for key, var_list in self.name_variants.items():
                if first_name in var_list or first_lower == key:
                    canonical_first = var_list[0]
                    variants = var_list
                    break

        # Normalize last name
        last_lower = last_name.lower()
        canonical_last = last_name

        if last_lower in self.name_variants:
            if not variants:
                variants = self.name_variants[last_lower]
            canonical_last = self.name_variants[last_lower][0]
        else:
            for key, var_list in self.name_variants.items():
                if last_lower == key or last_name in var_list:
                    canonical_last = var_list[0]
                    break

        canonical = f"{canonical_first} {canonical_last}".strip()
        confidence = 0.9 if variants else 0.6

        result = {
            "original": name,
            "canonical": canonical,
            "variants": variants,
            "first_name": canonical_first,
            "last_name": canonical_last,
            "confidence": round(confidence, 2),
        }

        if name != canonical:
            logger.debug(f"Name normalized: '{name}' → '{canonical}'")

        return result

    def resolve_historical_date(self, text: str, year: int = None) -> dict:
        """Resolve Orthodox calendar references to actual dates.

        Handles: "Пасха 1878", "Рождество 1901", "около 1878"
        """
        text_lower = text.lower().strip()

        for ref, date_val in self.historical_dates.items():
            if ref in text_lower:
                if date_val == "movable":
                    # Easter or movable feast — needs computation
                    if year:
                        return self._compute_orthodox_easter(year)
                    year_match = re.search(r"\b(\d{4})\b", text)
                    if year_match:
                        return self._compute_orthodox_easter(int(year_match.group(1)))
                    return {
                        "original": text,
                        "resolved": f"Пасха {year or 'unknown'}",
                        "is_approximate": True,
                        "confidence": 0.3,
                    }
                elif date_val == "circa":
                    year_match = re.search(r"\b(\d{4})\b", text)
                    if year_match:
                        return {
                            "original": text,
                            "resolved": f"{year_match.group(1)}-01-01",
                            "is_approximate": True,
                            "confidence": 0.5,
                        }
                    return {
                        "original": text,
                        "resolved": "circa",
                        "is_approximate": True,
                        "confidence": 0.2,
                    }
                else:
                    # Fixed date
                    if year:
                        resolved = f"{year}-{date_val}"
                    else:
                        year_match = re.search(r"\b(\d{4})\b", text)
                        y = year_match.group(1) if year_match else "0000"
                        resolved = f"{y}-{date_val}"

                    return {
                        "original": text,
                        "resolved": resolved,
                        "is_approximate": False,
                        "confidence": 0.8,
                    }

        return {"original": text, "resolved": text, "is_approximate": False, "confidence": 1.0}

    def _compute_orthodox_easter(self, year: int) -> dict:
        """Compute Orthodox Easter (Julian calendar) using Meeus algorithm."""
        a = year % 4
        b = year % 7
        c = year % 19
        d = (19 * c + 15) % 30
        e = (2 * a + 4 * b - d + 34) % 7
        month = (d + e + 114) // 31
        day = ((d + e + 114) % 31) + 1

        # Julian date — add 13 days for Gregorian in 1900s
        resolved = f"{year}-{month:02d}-{day:02d}"

        logger.debug(f"Orthodox Easter {year}: {resolved}")

        return {
            "original": f"Пасха {year}",
            "resolved": resolved,
            "is_approximate": False,
            "confidence": 0.85,
        }

    def compute_age(
        self, birth_date: str, death_date: str = None, reference_date: str = None
    ) -> dict:
        """Compute age from birth and death dates."""

        def parse_date(date_str):
            if not date_str or date_str == "Unknown":
                return None
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                try:
                    return datetime(int(date_str), 1, 1)
                except (ValueError, TypeError):
                    return None

        birth = parse_date(birth_date)

        if death_date and death_date != "Unknown":
            death = parse_date(death_date)
            if birth and death and death > birth:
                age = death - birth
                return {
                    "age_years": age.days // 365,
                    "age_days": age.days,
                    "source": "birth_death_difference",
                    "confidence": 0.9,
                }

        if reference_date:
            ref = parse_date(reference_date)
            if birth and ref and ref > birth:
                age = ref - birth
                return {
                    "age_years": age.days // 365,
                    "age_days": age.days,
                    "source": f"reference_date_{reference_date}",
                    "confidence": 0.7,
                }

        if birth_date and birth_date != "Unknown" and len(birth_date) == 4:
            return {
                "age_years": None,
                "age_days": None,
                "source": "birth_year_only",
                "confidence": 0.3,
                "note": f"Родился {birth_date}",
            }

        return {
            "age_years": None,
            "age_days": None,
            "source": "insufficient_data",
            "confidence": 0.0,
        }

    def validate_age_consistency(
        self, extracted_age: int, birth_date: str, death_date: str
    ) -> dict:
        """Validate that extracted age is consistent with dates."""
        computed = self.compute_age(birth_date, death_date)
        computed_age = computed.get("age_years")

        if extracted_age and computed_age:
            diff = abs(extracted_age - computed_age)
            consistent = diff <= 2

            return {
                "is_consistent": consistent,
                "computed_age": computed_age,
                "extracted_age": extracted_age,
                "difference": diff,
                "confidence": 0.85 if consistent else 0.4,
            }

        return {
            "is_consistent": True,
            "computed_age": computed_age,
            "extracted_age": extracted_age,
            "difference": None,
            "confidence": 0.3,
        }

    def link_family_members(self, records: list) -> dict:
        """Link family members across multiple records."""
        family_groups = []
        relationships = []

        surname_groups = {}
        for record in records:
            for field in [
                "child_name",
                "father_name",
                "mother_name",
                "groom_name",
                "bride_name",
                "deceased_name",
            ]:
                if field in record and isinstance(record[field], dict):
                    name = record[field].get("value", "")
                    if name and name != "Unknown":
                        parts = name.split()
                        if len(parts) >= 2:
                            surname = parts[-1]
                            if surname not in surname_groups:
                                surname_groups[surname] = []
                            surname_groups[surname].append(
                                {
                                    "name": name,
                                    "role": field,
                                    "record_type": record.get("record_type", "unknown"),
                                    "record": record,
                                }
                            )

        for surname, members in surname_groups.items():
            if len(members) >= 2:
                family_groups.append({"surname": surname, "members": members, "size": len(members)})

                for i, member1 in enumerate(members):
                    for member2 in members[i + 1 :]:
                        rel = self._detect_relationship(member1, member2)
                        if rel:
                            relationships.append(rel)

        return {
            "family_groups": family_groups,
            "relationships": relationships,
            "total_records": len(records),
        }

    def _detect_relationship(self, member1: dict, member2: dict) -> dict | None:
        """Detect relationship between two family members."""
        if member1["role"] == "father_name" and member2["role"] == "child_name":
            return {
                "type": "parent_child",
                "parent": member1["name"],
                "child": member2["name"],
                "parent_role": "father",
                "confidence": 0.8,
            }
        if member2["role"] == "father_name" and member1["role"] == "child_name":
            return {
                "type": "parent_child",
                "parent": member2["name"],
                "child": member1["name"],
                "parent_role": "father",
                "confidence": 0.8,
            }
        if member1["role"] == "mother_name" and member2["role"] == "child_name":
            return {
                "type": "parent_child",
                "parent": member1["name"],
                "child": member2["name"],
                "parent_role": "mother",
                "confidence": 0.8,
            }
        if member2["role"] == "mother_name" and member1["role"] == "child_name":
            return {
                "type": "parent_child",
                "parent": member2["name"],
                "child": member1["name"],
                "parent_role": "mother",
                "confidence": 0.8,
            }
        if (
            member1["role"] == "groom_name"
            and member2["role"] == "bride_name"
            and member1["record"] == member2["record"]
        ):
            return {
                "type": "spouse",
                "person1": member1["name"],
                "person2": member2["name"],
                "confidence": 0.9,
            }
        return None

    def resolve_entity(self, record: dict) -> dict:
        """Full entity resolution on a single record."""
        resolved = record.copy()

        # Resolve names
        for field in [
            "child_name",
            "father_name",
            "mother_name",
            "deceased_name",
            "groom_name",
            "bride_name",
        ]:
            if field in record and isinstance(record[field], dict):
                name = record[field].get("value", "")
                if name and name != "Unknown":
                    resolved[f"{field}_resolved"] = self.normalize_name(name)

        # Resolve dates
        for field in ["birth_date", "death_date", "marriage_date", "burial_date", "baptism_date"]:
            if field in record and isinstance(record[field], dict):
                date_val = record[field].get("value", "")
                if date_val and date_val != "Unknown":
                    has_ref = any(ref in date_val.lower() for ref in self.historical_dates)
                    if has_ref or not re.match(r"\d{4}-\d{2}-\d{2}", date_val):
                        year_match = re.search(r"\b(\d{4})\b", date_val)
                        year = int(year_match.group(1)) if year_match else None
                        resolved[f"{field}_resolved"] = self.resolve_historical_date(date_val, year)

        # Compute age
        if all(k in record for k in ["birth_date", "death_date"]):
            birth = (
                record["birth_date"].get("value", "")
                if isinstance(record["birth_date"], dict)
                else ""
            )
            death = (
                record["death_date"].get("value", "")
                if isinstance(record["death_date"], dict)
                else ""
            )

            if birth and death and birth != "Unknown" and death != "Unknown":
                age_info = self.compute_age(birth, death)
                resolved["age_computed"] = age_info

                if "age" in record:
                    extracted_age = (
                        record["age"].get("value") if isinstance(record["age"], dict) else None
                    )
                    if extracted_age:
                        try:
                            age_int = int(extracted_age)
                            resolved["age_validation"] = self.validate_age_consistency(
                                age_int, birth, death
                            )
                        except (ValueError, TypeError):
                            pass

        return resolved

    def resolve_batch(self, records: list) -> dict:
        """Resolve entities across multiple records."""
        resolved_records = [self.resolve_entity(rec) for rec in records]
        family_info = self.link_family_members(resolved_records)

        return {"records": resolved_records, "family": family_info}


def resolve_entities(record: dict) -> dict:
    resolver = EntityResolver()
    return resolver.resolve_entity(record)


def resolve_batch(records: list) -> dict:
    resolver = EntityResolver()
    return resolver.resolve_batch(records)
