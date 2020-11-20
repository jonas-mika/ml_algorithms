from knn_classifier import KNN

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def accuracy_score(y_pred, y_test):
        return round(sum(y_pred==y_test)/len(y_test),2)

def load_seed_data():
    iris_data = np.loadtxt("test_data_iris.data")
    # print(iris_data.shape)

    # split into descriptive and target feature
    X = iris_data[:,:-1]
    y = iris_data[:,-1]
    # print(f'First Row of descriptive Features: {X[0,:]}\nFirst Row of target Feature: {y[0]}')
    # print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

    return X_train, X_test, y_train, y_test 

def test_data():
    pass

def my_prediction(X_train, X_test, y_train, y_test, k):
    knn = KNN(k=k)
    knn.fit(X_train, y_train)

    prediction = knn.predict(X_test)

    return (prediction, accuracy_score(prediction, y_test))

def sklearn_prediction(X_train, X_test, y_train, y_test, k):
    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)

    prediction = knn.predict(X_test)

    return (prediction, accuracy_score(prediction, y_test))

def main():
    X_train, X_test, y_train, y_test = load_seed_data()
    print(my_prediction(X_train, X_test, y_train, y_test, 3))
    print(sklearn_prediction(X_train, X_test, y_train, y_test, 3))

# run the tests
main()