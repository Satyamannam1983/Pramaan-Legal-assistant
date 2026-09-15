from app.rag.document_analysis.metadata_extractor import extract_metadata


def test_metadata_extraction_is_conservative_and_grounded():
    metadata = extract_metadata(
        "doc-1",
        "IN THE SUPREME COURT OF INDIA\nCivil Appeal No. 12/2024\nDecided: 10 September 2024\nSection 302 applies.",
    )
    assert metadata.court == "IN THE SUPREME COURT OF INDIA"
    assert metadata.case_number == "Civil Appeal No. 12/2024"
    assert metadata.date == "10 September 2024"
    assert "Section 302" in metadata.sections_referenced


def test_missing_metadata_remains_none():
    metadata = extract_metadata("doc-2", "An unstructured legal note without identifiers.")
    assert metadata.court is None
    assert metadata.case_number is None
    assert metadata.date is None
