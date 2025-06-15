import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def J_cost(X, y, beta):
    '''
    :param X:  sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return: the result of formula 3.27
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))] # 构造X_hat
    # X_hat shape (n_samples, n_features + 1) 添加了偏置项
    # beta shape (n_features + 1, ) or (n_features + 1, 1)
    beta = beta.reshape(-1, 1) # 偏置的第一维自动计算，第二维为 1
    y = y.reshape(-1, 1)
    Lbeta = -y * np.dot(X_hat, beta) + np.log(1 + np.exp(np.dot(X_hat, beta))) # 对数似然
    return Lbeta.sum()


def gradient(X, y, beta):
    '''
    compute the first derivative of J(i.e. formula 3.27) with respect to beta   i.e. formula 3.30
    ------------------------------------------------------
    :param X:  sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return:
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    p1 = sigmoid(np.dot(X_hat, beta))
    gra = (-X_hat * (y - p1)).sum(0)
    return gra.reshape(-1, 1)



def hessian(X, y, beta):
    '''
    Fisher 信息矩阵，逆可以用来估计参数的协方差矩阵
    在逻辑回归的牛顿法（或拟牛顿法如 BFGS）中，Hessian 矩阵的逆近似为 Fisher 信息矩阵的逆
    compute the second derivative of J(i.e. formula 3.27) with respect to beta   i.e. formula 3.31
    ------------------------------------------------------
    :param X:  sample array, shape(n_samples, n_features)
    :param y: array-like, shape (n_samples,)
    :param beta: the beta in formula 3.27 , shape(n_features + 1, ) or (n_features + 1, 1)
    :return
    '''
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    beta = beta.reshape(-1, 1)
    y = y.reshape(-1, 1)
    p1 = sigmoid(np.dot(X_hat, beta))
    m, n = X.shape
    P = np.eye(m) * p1 * (1 - p1) # 创建对角矩阵，对角线元素为 p1 * (1 - p1)
    assert P.shape[0] == P.shape[1] # 确保 P 是方阵
    return np.dot(np.dot(X_hat.T, P), X_hat) # 计算 Hessian 矩阵


def update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost):
    '''
    update parameters with gradient descent method
    ----------------------------------------------
    :para beta:
    :para grad:
    :para learning_rate:
    :return:
    '''
    for i in range(num_iterations):
        grad = gradient(X, y, beta)
        beta = beta - learning_rate * grad
        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))
    return beta


def update_parameters_newton(X, y, beta, num_iterations, print_cost):
    '''
    update parameters with gradient descent method
    ----------------------------------------------
    :para beta:
    :para grad:
    :para learning_rate:
    :return:
    '''
    for i in range(num_iterations):
        grad = gradient(X, y, beta)
        hess = hessian(X, y, beta)
        beta = beta - np.dot(np.linalg.inv(hess), grad)

        if (i % 10 == 0) & print_cost:
            print('{}th iteration, cost is {}'.format(i, J_cost(X, y, beta)))
    return beta


def initialize_beta(n):
    beta = np.zeros((n + 1, 1))  # n features + 1 for intercept
    return beta  # ensure it's a column vector


def logistic_model(X, y, num_iterations=100, learning_rate=1.2, print_cost=False, method='gradDesc'):
    '''
    :para X:
    
    '''
    m, n = X.shape
    beta = initialize_beta(n)
    if method == 'gradDesc':
        return update_parameters_gradDesc(X, y, beta, learning_rate, num_iterations, print_cost)
    elif method == 'Newton':
        return update_parameters_newton(X, y, beta, num_iterations, print_cost)
    else:
        raise ValueError('Unknown solver %s' %method)
    
    
def predict(X, beta):
    X_hat = np.c_[X, np.ones((X.shape[0], 1))]
    p1 = sigmoid(np.dot(X_hat, beta))
    p1[p1 >= 0.5] = 1
    p1[p1 < 0.5] = 0
    return p1


if __name__ == '__main__':
    data_path = r'E:\CODE\machine-learning-notes\3.3\watermelon.csv'
    data = pd.read_csv(data_path).values
    print(data)
    is_good = data[:, 9] == "是"
    is_bad = data[:, 9] == "否"
    print(is_good, is_bad)
    
    X = data[:, 7:9].astype(float)
    y = data[:, 9]
    print(X, y)
    
    y[y == "是"] = 1
    y[y == "否"] = 0
    y = y.astype(int)
    
    plt.scatter(data[:, 7][is_good],data[:, 8][is_good], c='k', marker='o')
    plt.scatter(data[:, 7][is_bad],data[:, 8][is_bad], c='r', marker='x')
    
    plt.xlabel('密度')
    plt.ylabel('含糖量')
    
    # 可视化模型结果
    beta = logistic_model(X, y, num_iterations=1000, learning_rate=0.3, print_cost=True, method='gradDesc')
    w1, w2, intercept = beta
    x1 = np.linspace(0, 1)
    y1 = -(w1 * x1 + intercept) / w2
    
    ax1, = plt.plot(x1, y1, c='b', label='Logistic Regression')
    lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)
    lr.fit(X, y)

    lr_beta = np.c_[lr.coef_, lr.intercept_]
    print(J_cost(X, y, lr_beta))
    
    # 可视化sklearn的结果
    w1_sk, w2_sk = lr.coef_[0, :]
    x2 = np.linspace(0, 1)
    y2 = -(w1_sk * x2 + intercept) / w2_sk
    ax2, = plt.plot(x2, y2, c='g', label='sklearn_logistic')
    plt.legend(loc='upper right')
    plt.show()