import pandas as pd

from featstory import Story


def create_test_data():

    return pd.DataFrame({
        "study_hours": [2, 4, 6, 8, 10],
        "attendance": [60, 70, 80, 90, 95],
        "previous_score": [55, 65, 75, 85, 90],
        "final_score": [58, 68, 78, 88, 93]
    })


def test_story_creation():

    df = create_test_data()

    story = Story(
        df,
        target="final_score"
    )

    assert story.data.shape == (5, 4)
    assert story.target == "final_score"


def test_summary():

    df = create_test_data()

    story = Story(
        df,
        target="final_score"
    )

    result = story.summary()

    assert result["rows"] == 5
    assert result["columns"] == 4
    assert result["target"] == "final_score"


def test_analysis():

    df = create_test_data()

    story = Story(
        df,
        target="final_score"
    )

    result = story.analyze()

    assert "correlations" in result
    assert "feature_stories" in result
    assert "categorical_summary" in result


def test_risk_detection():

    df = create_test_data()

    story = Story(
        df,
        target="final_score"
    )

    risks = story.detect_risks()

    assert isinstance(risks, list)


def test_model_importance():

    df = create_test_data()

    story = Story(
        df,
        target="final_score"
    )

    importance = story.model_importance()

    assert isinstance(importance, dict)
    assert len(importance) > 0