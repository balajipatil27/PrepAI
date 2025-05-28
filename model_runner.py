from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_model(X_train, X_test, y_train, y_test, algorithm):
    models = {
        'logistic': LogisticRegression(max_iter=1000),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(),
        'knn': KNeighborsClassifier(),
        'naive_bayes': GaussianNB(),
        'svm': SVC()
    }

    if algorithm not in models:
        raise ValueError("Invalid algorithm selected.")

    model = models[algorithm]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, accuracy_score(y_test, y_pred), y_pred