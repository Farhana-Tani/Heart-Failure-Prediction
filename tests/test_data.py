import pytest
from src.config import FEATURES
from src.data import validate_patient


def sample_record():
    return dict.fromkeys(FEATURES, 1)


def test_patient_record_validates():
    assert validate_patient(sample_record()).shape == (1, len(FEATURES))


def test_patient_record_requires_all_features():
    record = sample_record(); record.pop("age")
    with pytest.raises(ValueError, match="missing"):
        validate_patient(record)
