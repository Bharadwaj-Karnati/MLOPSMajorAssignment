from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_data


def test_load_data():
    X, y = load_data()
    assert X.shape[0] == y.shape[0], "Feature and target row counts should match"
    assert X.shape[1] == 8, "California Housing dataset should have 8 features"

def test_model_creation():
    model = LinearRegression()
    assert isinstance(model, LinearRegression), "Model should be a LinearRegression instance"

def test_model_training_and_attributes():
    X, y = load_data()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    assert hasattr(model, 'coef_'), "Model should have 'coef_' attribute after training"
    assert model.coef_.shape[0] == X_train.shape[1], "Coefficient count should match feature count"

def test_r2_score_above_threshold():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    threshold = 0.5
    assert r2 > threshold, f"R2 score {r2:.2f} is below threshold {threshold}"

