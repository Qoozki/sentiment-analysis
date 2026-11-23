from app.domain.value_objects.clean_options import CleanOptions
from app.domain.value_objects.sentiment_scale import THRESHOLD, label_for


def test_label_thresholds():
    assert label_for(THRESHOLD) == "Positive"
    assert label_for(-THRESHOLD) == "Negative"
    assert label_for(THRESHOLD - 0.01) == "Neutral"
    assert label_for(0.0) == "Neutral"


def test_clean_options_validate():
    assert CleanOptions().active
    assert not CleanOptions("none", "none", False).active
    try:
        CleanOptions(nulls="bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError")
