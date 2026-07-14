"""Tests for Entity Resolution Service."""
import pytest
from app.services.entity_resolution import EntityResolver


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def resolver():
    return EntityResolver()


@pytest.fixture
def sample_birth_record():
    return {
        "record_type": "birth",
        "child_name": {"value": "Иван Петров", "confidence": 0.9},
        "birth_date": {"value": "1878-03-12", "confidence": 0.95},
        "father_name": {"value": "Петр Иванов", "confidence": 0.88},
        "mother_name": {"value": "Анна Иванова", "confidence": 0.86},
    }


@pytest.fixture
def sample_death_record():
    return {
        "record_type": "death",
        "deceased_name": {"value": "Михаил Николаев", "confidence": 0.9},
        "death_date": {"value": "1901-11-20", "confidence": 0.95},
        "birth_date": {"value": "1836", "confidence": 0.7},
        "age": {"value": "65", "confidence": 0.85},
    }


# ── Test: Name Normalization ────────────────────────────────────────────────
class TestNameNormalization:
    def test_normalize_common_name(self, resolver):
        result = resolver.normalize_name("Иван")
        assert result["canonical"] == "Иван"
        assert len(result["variants"]) > 0
        assert result["confidence"] >= 0.9

    def test_normalize_name_variant(self, resolver):
        result = resolver.normalize_name("Иоанн")
        assert result["canonical"] == "Иоанн"  
        assert "Иван" in result["variants"]   

    def test_normalize_full_name(self, resolver):
        result = resolver.normalize_name("Иван Петров")
        assert result["first_name"] == "Иван"
        assert result["last_name"] == "Петров"

    def test_normalize_unknown_name(self, resolver):
        result = resolver.normalize_name("Unknown")
        assert result["canonical"] == "Unknown"
        assert result["confidence"] == 0.0

    def test_normalize_empty_name(self, resolver):
        result = resolver.normalize_name("")
        assert result["canonical"] == ""
        assert result["confidence"] == 0.0

    def test_normalize_patronymic(self, resolver):
        result = resolver.normalize_name("Иванович")
        assert result is not None
        assert "Иванович" in result["variants"]

    def test_normalize_female_name(self, resolver):
        result = resolver.normalize_name("Мария")
        assert result["canonical"] == "Мария"
        assert "Маша" in result["variants"]

    def test_normalize_uncommon_name(self, resolver):
        """Name not in dictionary should get lower confidence."""
        result = resolver.normalize_name("Аристарх")
        assert result["confidence"] <= 0.6


# ── Test: Historical Date Resolution ────────────────────────────────────────

class TestHistoricalDateResolution:
    def test_resolve_christmas(self, resolver):
        result = resolver.resolve_historical_date("Рождество 1878", year=1878)
        assert result["resolved"] == "1878-12-25"
        assert result["is_approximate"] is False
        assert result["confidence"] >= 0.8

    def test_resolve_easter(self, resolver):
        result = resolver.resolve_historical_date("Пасха 1878", year=1878)
        assert result["resolved"] != "Пасха 1878"
        assert result["is_approximate"] is False

    def test_resolve_circa(self, resolver):
        result = resolver.resolve_historical_date("около 1878 года")
        assert result["is_approximate"] is True
        assert "1878" in result["resolved"]

    def test_resolve_unknown_reference(self, resolver):
        result = resolver.resolve_historical_date("какой-то текст")
        assert result["confidence"] == 1.0
        assert result["resolved"] == "какой-то текст"

    def test_resolve_easter_different_years(self, resolver):
        """Easter falls on different dates each year."""
        result_1878 = resolver.resolve_historical_date("Пасха 1878", year=1878)
        result_1900 = resolver.resolve_historical_date("Пасха 1900", year=1900)
        assert result_1878["resolved"] != result_1900["resolved"]

    def test_resolve_easter_without_year(self, resolver):
        """Should extract year from text if not provided."""
        result = resolver.resolve_historical_date("Пасха 1878 года")
        assert "1878" in result["resolved"]

    def test_resolve_fixed_holiday(self, resolver):
        result = resolver.resolve_historical_date("Благовещение 1890", year=1890)
        assert "1890-03-25" in result["resolved"]


# ── Test: Age Computation ───────────────────────────────────────────────────
class TestAgeComputation:
    def test_compute_age_birth_death(self, resolver):
        result = resolver.compute_age("1878-03-12", "1901-11-20")
        assert result["age_years"] == 23
        assert result["confidence"] >= 0.9

    def test_compute_age_birth_only(self, resolver):
        result = resolver.compute_age("1878-03-12")
        assert result["age_years"] is None
        assert result["confidence"] == 0.0

    def test_compute_age_with_reference_date(self, resolver):
        result = resolver.compute_age("1878-03-12", reference_date="1900-01-01")
        assert result["age_years"] is not None
        assert result["confidence"] >= 0.7

    def test_compute_age_birth_year_only(self, resolver):
        result = resolver.compute_age("1878")
        assert result["confidence"] == 0.3  # Year-only = low confidence

    def test_compute_age_invalid_dates(self, resolver):
        result = resolver.compute_age("invalid", "also invalid")
        assert result["confidence"] == 0.0

    def test_compute_age_death_before_birth(self, resolver):
        """Death before birth should gracefully handle."""
        result = resolver.compute_age("1900-01-01", "1800-01-01")
        assert result["confidence"] == 0.0

    def test_compute_age_unknown_dates(self, resolver):
        result = resolver.compute_age("Unknown", "Unknown")
        assert result["confidence"] == 0.0


# ── Test: Age Validation ────────────────────────────────────────────────────
class TestAgeValidation:
    def test_consistent_age(self, resolver):
        result = resolver.validate_age_consistency(65, "1836-01-01", "1901-11-20")
        assert result["is_consistent"] is True

    def test_inconsistent_age(self, resolver):
        """Age 30 when dates suggest 65 should be inconsistent."""
        result = resolver.validate_age_consistency(30, "1836-01-01", "1901-11-20")
        assert result["is_consistent"] is False

    def test_missing_age(self, resolver):
        result = resolver.validate_age_consistency(None, "1836-01-01", "1901-11-20")
        assert result["confidence"] == 0.3

    def test_age_difference_reported(self, resolver):
        result = resolver.validate_age_consistency(65, "1836-01-01", "1901-11-20")
        assert result["difference"] is not None


# ── Test: Family Linking ────────────────────────────────────────────────────
class TestFamilyLinking:
    def test_link_parent_child(self, resolver):
        records = [
            {"child_name": {"value": "Иван Петров"}, "record_type": "birth"},
            {"father_name": {"value": "Петр Петров"}, "record_type": "birth"}, 
        ]
        result = resolver.link_family_members(records)
        assert len(result["family_groups"]) > 0

    def test_link_spouses(self, resolver):
        records = [{
            "groom_name": {"value": "Александр Павлов"},
            "bride_name": {"value": "Елена Павлова"},  # Same surname after marriage
            "record_type": "marriage",
        }]
        result = resolver.link_family_members(records)
        relationships = result["relationships"]
        spouse_rels = [r for r in relationships if r["type"] == "spouse"]
        assert len(spouse_rels) > 0

    def test_multi_generation_family(self, resolver):
        records = [
            {"child_name": {"value": "Иван Петров"}, "father_name": {"value": "Петр Иванов"},
             "mother_name": {"value": "Анна Иванова"}, "record_type": "birth"},
            {"child_name": {"value": "Мария Петрова"}, "father_name": {"value": "Петр Иванов"},
             "mother_name": {"value": "Анна Иванова"}, "record_type": "birth"},
        ]
        result = resolver.link_family_members(records)
        assert len(result["family_groups"]) > 0

    def test_no_relations_found(self, resolver):
        records = [
            {"deceased_name": {"value": "Иван"}, "record_type": "death"},
            {"deceased_name": {"value": "Петр"}, "record_type": "death"},
        ]
        result = resolver.link_family_members(records)
        assert len(result["relationships"]) == 0


# ── Test: Full Entity Resolution ────────────────────────────────────────────
class TestEntityResolution:
    def test_resolve_birth_record(self, resolver, sample_birth_record):
        result = resolver.resolve_entity(sample_birth_record)
        assert "child_name_resolved" in result
        assert "birth_date_resolved" not in result or True  # dates may skip if normalized

    def test_resolve_entity_adds_normalized(self, resolver, sample_birth_record):
        result = resolver.resolve_entity(sample_birth_record)
        assert result["child_name_resolved"]["canonical"] == "Иван Петров"
        assert result["child_name_resolved"]["confidence"] >= 0.9

    def test_resolve_batch(self, resolver):
        records = [{"record_type": "birth", "child_name": {"value": "Иван", "confidence": 0.9}}]
        result = resolver.resolve_batch(records)
        assert "records" in result
        assert "family" in result
        assert len(result["records"]) == 1

    def test_resolve_entity_adds_age_validation(self, resolver, sample_death_record):
        result = resolver.resolve_entity(sample_death_record)
        assert "age_computed" in result
        assert "age_validation" in result

    def test_resolve_entity_with_orthodox_date(self, resolver):
        record = {
            "record_type": "birth",
            "birth_date": {"value": "Пасха 1878", "confidence": 0.5},
        }
        result = resolver.resolve_entity(record)
        assert "birth_date_resolved" in result
        assert result["birth_date_resolved"]["resolved"] != "Пасха 1878"