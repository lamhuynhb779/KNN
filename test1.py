import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def myweight(distances):
	sigma2 = .5 #we can chnage this number
	return np.exp(-distances**2 / sigma2)

#Tiếp theo, chúng ta load dữ liệu và hiện thị vài dữ liệu mẫu. 
#Các class được gán nhãn là 0, 1, và 2.
iris = datasets.load_iris()
iris_X = iris.data # chua 150 diem 
iris_Y = iris.target # tuong ung vs 150 diem thi se co cac label cua no
# print(iris_X)
# print(iris_Y)
print("Number of classes: %d" %len(np.unique(iris_Y)))
print("Number of data points: %d" %len(iris_Y))

X0 = iris_X[iris_Y == 0,:] # 50 phan tu
print("Samples from class 0:"); print(X0[:5,:])

X1 = iris_X[iris_Y == 1,:]
print("Samples from class 1:"); print(X1[:5,:])

X2 = iris_X[iris_Y == 2,:]
print("Samples from class 2:"); print(X2[:5,:])

#-------------------------------Classification--------------------------------------------
#Tách training và test sets
X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=50)

print("Traning size: %d" %len(Y_train))
print("Test size: %d" %len(Y_test))

# Su dung K = 1, weights = 'uniform' (con tat ca cac diem co trong so "do tin tuong" nhu nhau)
# clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
# clf.fit(X_train, Y_train)#Fit the model using X as training data and y as target values
# Y_pred = clf.predict(X_test)

# print("Print result for 20 test data point:")
# print("Predicted labels: %s" %(Y_pred[20:40]))
# print("Ground truth    : %s" %(Y_test[20:40]))
# # Lay so du doan dung chia cho tong so phan tu 
# print("Accuracy of 1NN with major voting: %.2f %%" %(100 * accuracy_score(Y_test, Y_pred)))

# Su dung K = 10, weights = 'distance' (diem can gan thi trong so "do tin tuong" cang cao)
# clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
# clf.fit(X_train, Y_train)#Fit the model using X as training data and y as target values
# Y_pred = clf.predict(X_test)

# print("Print result for 20 test data point:")
# print("Predicted labels: %s" %(Y_pred[20:40]))
# print("Ground truth    : %s" %(Y_test[20:40]))
# # Lay so du doan dung chia cho tong so phan tu 
# print("Accuracy of 10NN (1/distance weights): %.2f %%" %(100 * accuracy_score(Y_test, Y_pred)))

# Su dung K = 10
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, Y_train)#Fit the model using X as training data and y as target values
Y_pred = clf.predict(X_test)

print("Print result for 20 test data point:")
print("Predicted labels: %s" %(Y_pred[20:40]))
print("Ground truth    : %s" %(Y_test[20:40]))
# Lay so du doan dung chia cho tong so phan tu 
print("Accuracy of 10NN (customized weights): %.2f %%" %(100 * accuracy_score(Y_test, Y_pred)))

#-------------------------------Regression--------------------------------------------
