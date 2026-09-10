import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
def train():
    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )
    # Train lightweight model
    clf = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )
    clf.fit(X_train, y_train)
    # Evaluate basic accuracy
    accuracy = clf.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save model artifact
    joblib.dump(clf, "model.pkl")
    print("Model successfully saved as model.pkl")
if __name__ == "__main__":
    train()