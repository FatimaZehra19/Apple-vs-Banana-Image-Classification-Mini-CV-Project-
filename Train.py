from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print("Accuracy:", acc)

    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
