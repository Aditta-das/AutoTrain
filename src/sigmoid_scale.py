import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn import linear_model, ensemble, tree, svm
from sklearn import model_selection, metrics
import pickle
df = pd.read_csv("G:/AutoTrain/autodeep/classification/breast_cancer.csv")
# df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
if (isinstance(df["diagnosis"][0], np.int64)):
	print("True")
else:
	print("False")
print(np.array(df["diagnosis"]))
# print(df["diagnosis"][0].dtype)
print(pd.Series(df["diagnosis"]).dtypes)
le = preprocessing.LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])
L = le.inverse_transform(df["diagnosis"].unique())
print(L)
# X = df.drop("diagnosis", axis=1)
# y = df["diagnosis"]
# print(X)
# minmax = preprocessing.MinMaxScaler().fit(X)
# X_minmax = minmax.transform(X)
# print(X_minmax)

# standard = preprocessing.StandardScaler().fit(X)
# X_std = standard.transform(X)
# X = X_std.reshape(X_std.shape[0], X_std.shape[1], 1)

# print(X_std)
# robust = preprocessing.RobustScaler().fit(X)
# ro_sc = robust.transform(X)
# print(ro_sc)
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, test_size=0.3)
# print(X_train.shape)
# print(y_train.shape)
# model = linear_model.LogisticRegression()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# acc = metrics.accuracy_score(y_pred, y_test)
# print(f" Using RobustScaler and Labelencoder\nAccuracy is :{acc}")

# class SigmoidScaler:
# 	def __init__(self):
# 		super(SigmoidScaler, self).__init__()

# 	def sigmoid(self, val):
# 		return np.array(1 / (1 + np.exp(-val)))

# 	def tanh(self, val):
# 		return np.array(np.exp(val) - np.exp(-val)) / (np.exp(val) + np.exp(-val))
	
# 	def fit_transform(self, x):
		# x is all the values from one feature
		# 1 / 1 + e^(-x)
		# fit those features
		# sig_ = self.sigmoid(val=x)
		# return sig_
		# data = []
		# for y in range(len(x.columns)):
		# 	p = np.array(x[x.columns[y]])
		# 	print(p)
		# 	for z in p:
		# 		sig = self.sigmoid(val=z)

		# 		print(sig)
			# print(a)
			# for z in y:
			# 	# sig = self.sigmoid(val=z)
			# 	print(z)
			

# a = SigmoidScaler()
# scale = a.fit_transform(X)
# # print(scale)

# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, test_size=0.3, shuffle=True)

# # model = linear_model.LogisticRegression()
# model = ensemble.RandomForestClassifier()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# acc = metrics.accuracy_score(y_pred, y_test)
# print(f" Using New and Labelencoder\nAccuracy is :{acc}")

# from tensorflow.keras.layers import Input, MaxPooling1D, Lambda, Dense, Conv1D, Flatten, Conv1D, MaxPooling2D, Dropout, Activation
# from tensorflow.keras.models import Model
# from tensorflow.keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator
# import tensorflow as tf

# model = Sequential()
# model.add(Conv1D(64, 3, activation="relu", input_shape=X_train.shape[1:]))
# model.add(Dense(16, activation="relu"))
# model.add(MaxPooling1D())
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss ='binary_crossentropy', optimizer = "adam", metrics = ['accuracy'])
# model.summary()

# history = model.fit(X_train, y_train, batch_size=2, epochs=10, validation_data=(X_test, y_test), verbose=2)
