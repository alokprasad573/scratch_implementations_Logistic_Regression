from Regression_Classifier import LogisticRegression

import numpy as np


def main():
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 0, 1, 1])
        
        classifier = LogisticRegression()
        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        print(y_pred)

if __name__=="__main__":
    main()