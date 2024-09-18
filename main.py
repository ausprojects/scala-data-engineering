import mlflow 
import mlflow.sklearn 
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Start MLflow experiment
with mlflow.start_run():
    # Train model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    # Log model and params
    mlflow.sklearn.log_model(clf, "model")
    mlflow.log_param("n_estimators", clf.n_estimators)
    mlflow.log_metric("accuracy", clf.score(X_test, y_test))

    print(f"Model saved in run {mlflow.active_run().info.run_uuid}")

