from Regression_Classifier import LogisticRegressionC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import numpy as np


def main():
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        classifier = LogisticRegressionC()
        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        print("Custom Logistic Regression Predictions accuracy :", accuracy_score(y, y_pred))
        
        
def predefined():
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        classifier = LogisticRegression()
        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        print("Library Logistic Regression Predictions accuracy :", accuracy_score(y, y_pred))
        
        
        
        
if __name__=="__main__":
    main()
    predefined()