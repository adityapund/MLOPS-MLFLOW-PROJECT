import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")


mlflow.set_experiment("Wine Experiment")
wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

max_depth = 5
n_estimators = 100

with mlflow.start_run():
    rf  = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    plt.figure(figsize=(8,6)) 
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names) 
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    plt.savefig('confusion_matrix.png')

    #mlflow log mode
    mlflow.sklearn.log_model(rf, "random_forest_model")

    #mlflow log params
    mlflow.log_params({"max_depth": max_depth, "n_estimators": n_estimators})
    
    #mlflow log metrics 
    mlflow.log_metrics({"accuracy": accuracy})

    #mlflow log test
    mlflow.log_text(classification_report(y_test, y_pred), "classification_report.txt")   
    
    #mlflow log artifact
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    #mlflow log tags
    mlflow.set_tags({"Author": "Aditya", "Experiment":"WineClassification"})
    