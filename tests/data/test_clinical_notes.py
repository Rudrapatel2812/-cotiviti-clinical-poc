from data.clinical_notes import CLINICAL_NOTES


def test_has_three_notes():
    assert len(CLINICAL_NOTES) == 3


def test_each_note_has_required_keys():
    required = {"id", "title", "note"}
    for note in CLINICAL_NOTES:
        assert required.issubset(note.keys()), f"Missing keys in {note.get('id')}"


def test_patient_ids_are_correct():
    ids = [n["id"] for n in CLINICAL_NOTES]
    assert ids == ["P001", "P002", "P003"]


def test_notes_are_non_empty_strings():
    for note in CLINICAL_NOTES:
        assert isinstance(note["note"], str) and len(note["note"]) > 50
