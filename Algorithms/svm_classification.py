import numpy as np
X_train = np.array([[2,3],
            [2,5],
            [4,1],
            [6,1.5],
            [3,9],
            [10,20],
            [5,15]])

y_train = np.array([[-1],[-1],[-1],[-1],[1],[1],[1]])

X_test = np.array([[10,10],[2,3],[6,3]])
y_test = np.array([[1],[-1],[-1]])

class SupportVectorMachine:

    def __init__(self, epochs, learning_rate, bias=-1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.bias = bias

    def fit(self, X, Y):
        # weigght matrix
        self.w = np.zeros(len(X[0]))
        # the training and the gradient descent
        for epoch in range(1, self.epochs):
            # iterate on the inputs to update the weights
            for i , j in enumerate(X):
                # misclassification update
                if (Y[i]* np.dot(X[i], self.w) ) < 1:
                    self.w = self.w + self.learning_rate*( Y[i] *X[i] +(-2 *(1/self.epochs) * self.w))
                # correctly classified
                else:
                    self.w = self.w + self.learning_rate*(-2 * (1/self.epochs)* self.w)

        return self.w


    # predict method
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.bias)

    # get the accuracy of the model
    def score(self, X_test, Y_test):
        error = 0
        for i, j in enumerate(X_test):
            y_p = self.predict(X_test[i])
            if y_p != Y_test[i]:
                error += 1
        return 1- (error/len(X_test))


svm = SupportVectorMachine(epochs=100,learning_rate=0.01)
svm.fit(X_train, y_train)
accuracy = svm.score(X_test, y_test)
print(f"the model accuracy is {accuracy}")
