from sklearn import svm
from sklearn.metrics import f1_score


def train_SVM(X_train, y_train):
    model = svm.SVC(kernel='poly', degree=2)
    model.fit(X_train, y_train)
    return model


def test_SVM(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def calculate_f1_score(y_test, y_pred):
    score = f1_score(y_test, y_pred)
    return score

