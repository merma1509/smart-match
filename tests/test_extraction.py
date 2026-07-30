# ── Test: Pydantic Schema Validation ────────────────────────────────────────
class TestPydanticSchemas:
    def test_birth_schema_validation(self, extractor, birth_text):
        """Test that birth result can be validated with Pydantic schema."""
        from app.schemas.responses import BirthExtractionResult, FieldPrediction, ExtractionMetadata

        result = extractor.extract(birth_text)
        # Convert result to match schema structure (use extraction_meta alias)
        birth_result = BirthExtractionResult(
            record_type=result["record_type"],
            child_name=FieldPrediction(**result["child_name"]),
            birth_date=FieldPrediction(**result["birth_date"]),
            baptism_date=FieldPrediction(**result["baptism_date"]),
            father_name=FieldPrediction(**result["father_name"]),
            mother_name=FieldPrediction(**result["mother_name"]),
            needs_review=result["needs_review"],
            extraction_meta=ExtractionMetadata(**result["_extraction"]),
        )
        assert birth_result.record_type == "birth"

    def test_death_schema_validation(self, extractor, death_text):
        """Test that death result can be validated with Pydantic schema."""
        from app.schemas.responses import DeathExtractionResult, FieldPrediction, ExtractionMetadata

        result = extractor.extract(death_text)
        # Handle optional age field
        age = None
        if result.get("age"):
            age = FieldPrediction(**result["age"])

        death_result = DeathExtractionResult(
            record_type=result["record_type"],
            deceased_name=FieldPrediction(**result["deceased_name"]),
            death_date=FieldPrediction(**result["death_date"]),
            burial_date=FieldPrediction(**result["burial_date"]),
            age=age,
            needs_review=result["needs_review"],
            extraction_meta=ExtractionMetadata(**result["_extraction"]),
        )
        assert death_result.record_type == "death"

    def test_marriage_schema_validation(self, extractor, marriage_text):
        """Test that marriage result can be validated with Pydantic schema."""
        from app.schemas.responses import MarriageExtractionResult, FieldPrediction, ExtractionMetadata

        result = extractor.extract(marriage_text)
        marriage_result = MarriageExtractionResult(
            record_type=result["record_type"],
            groom_name=FieldPrediction(**result["groom_name"]),
            bride_name=FieldPrediction(**result["bride_name"]),
            marriage_date=FieldPrediction(**result["marriage_date"]),
            needs_review=result["needs_review"],
            extraction_meta=ExtractionMetadata(**result["_extraction"]),
        )
        assert marriage_result.record_type == "marriage"

