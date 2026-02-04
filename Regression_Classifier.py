import numpy as np


class LogisticRegressionC:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.bias = None
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        
    def _sigmoid(self, z):
        return (1/(1 + np.exp(-z)))
    
    def fit(self, X, y):
        m,n = X.shape
        self.bias = 0
        self.weights =  np.zeros(n)
        
        for i in range(self.n_iters):
            z = self.bias +  np.dot(X, self.weights)    
            y_pred = self._sigmoid(z)
            db = (1/m) * (y_pred - y)
            dw = (1/m) * np.dot(np.transpose(X), (y_pred - y))
            self.bias -= self.learning_rate * db
            self.weights -= self.learning_rate * dw
    
    def get_probability(self, X):
        z = self.bias +  np.dot(X, self.weights) 
        return self._sigmoid(z)
        
    def predict(self, X, threshold = 0.5):
        probability = self.get_probability(X)
        y_pred_bool = probability >= threshold
        return y_pred_bool.astype(int)