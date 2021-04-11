from numpy import log, dot, e
import numpy as np
from numpy.random import rand
import pandas as pd
class LogisticRegression:
	def __init__(self, epoch=100, lr=0.01):
		self.epoch = epoch
		self.lr = lr
    
	def sigmoid(self, z): 
		return 1 / (1 + np.exp(-z))

	def cost_function(self, X, y, weights):                 
		z = dot(X, weights)
		predict_1 = y * log(self.sigmoid(z))
		predict_0 = (1 - y) * log(1 - self.sigmoid(z))
		return -(sum(predict_1 + predict_0)) / len(X)
    
	def fit(self, X, y):        
		loss = []
		weights = np.zeros(X.shape[1], )
		N = len(X)

		for step in range(self.epoch):        
			# Gradient Descent
			y_hat = self.sigmoid(dot(X, weights))
			output_error_signal = y - y_hat
			gradient = dot(X.T, output_error_signal)
			weights += self.lr * gradient          
			# Saving Progress
			cost = self.cost_function(X, y, weights) 
			loss.append(cost) 
			if step % 100 == 0:
				print(f"step: {step} cost: {cost}")
		self.weights = weights
		self.loss = loss
    
	def predict(self, X):        
		# Predicting with sigmoid function
		z = dot(X, self.weights)
		# Returning binary result
		return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]

	def accuracy(self, predicted_labels, actual_labels):
		diff = predicted_labels - actual_labels
		return 1.0 - (float(np.count_nonzero(diff)) / len(diff))

df = pd.read_csv("../classification/diabetics.csv")
from sklearn import preprocessing
X = df.drop("Outcome", axis=1)
from sklearn import preprocessing
sc = preprocessing.StandardScaler()
X = sc.fit_transform(X)
y = df["Outcome"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)
a = LogisticRegression(epoch=10000)
a.fit(X_train, y_train)
y_pred = a.predict(X_test)
print(y_pred)
acc = a.accuracy(predicted_labels=y_pred, actual_labels=y_test)
print(acc)
from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
from sklearn import metrics
ac = metrics.accuracy_score(pred, y_test)
print(ac)
