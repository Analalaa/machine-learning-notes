import numpy as np
from sklearn import linear_model
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

data_path = r'E:\CODE\machine-learning-notes\chapter-3\3.4\Transfusion.txt'

data = np.loadtxt(data_path, delimiter=',',).astype(int)

X = data[:, :4]
y = data[:, 4]
m, n = X.shape

# normalize X
X = (X - X.mean(0)) / X.std(0)

# shuffle data
index = np.arange(m)
np.random.shuffle(index)
X = X[index]
y = y[index]

# 使用sklearn 的 api
# k-10 cross validation
lr = linear_model.LogisticRegression(C=2)
scores = cross_val_score(lr, X, y, cv=10)
print(scores.mean())


# leave one out cross validation
loo = LeaveOneOut()

accuracy = 0
for train, test in loo.split(X):
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    
    lr = linear_model.LogisticRegression(C=2)
    
    lr.fit(X_train, y_train)
    accuracy += lr.score(X_test, y_test)

print(accuracy / m)

# Output:比较类似如下
# 0.7700360360360361
# 0.7687165775401069

# 自己实现
# k-10 cross validation
num_splits = int(m / 10)
score_my = []
for i in range(10):
    lr_ = linear_model.LogisticRegression(C=2)
    test_index = range(i * num_splits, (i + 1) * num_splits)
    X_test, y_test = X[test_index], y[test_index]
    X_train = np.delete(X, test_index, axis=0)
    y_train = np.delete(y, test_index, axis=0)
    lr_.fit(X_train, y_train)
    score_my.append(lr_.score(X_test, y_test))
    
print(np.mean(score_my))

# leave one out cross validation
score_my_loo = []
for i in range(m):
    lr_ = linear_model.LogisticRegression(C=2)
    X_test_, y_test_ = X[i,:], y[i]
    X_train_ = np.delete(X, i, axis=0)
    y_train_ = np.delete(y, i, axis=0)
    
    lr_.fit(X_train_, y_train_)
    score_my_loo.append(int(lr_.predict(X_test_.reshape(1, -1))[0] ==  y_test_))
print(np.mean(score_my_loo))

# Output:比较类似如下
# 0.7716216216216216
# 0.7687165775401069