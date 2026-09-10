import os
import joblib
def test_model_artifact_exists():
    assert os.path.exists(
        "model.pkl"
    ), "Error: model.pkl artifact was not created."
def test_model_inference():
    clf = joblib.load("model.pkl")

    sample_input = [[5.1, 3.5, 1.4, 0.2]]

    prediction = clf.predict(sample_input)

    assert len(prediction) == 1, \
        "Error: Model failed to generate a prediction."