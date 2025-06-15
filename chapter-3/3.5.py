import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class LDA(object):

    def fit(self, X, y, plot=False):
        pos = y == 1
        neg = y == 0
        X0 = X[neg]
        X1 = X[pos]
        
        u0 = X0.mean(0, keepdims=True) # (1, n_features)
        u1 = X1.mean(0, keepdims=True) # (1, n_features)
        
        sw = np.dot((X0 - u0).T, X0 - u0) + np.dot((X1 - u1).T, X1 - u1) # (2, 2) 类内散度矩阵
        w = np.dot(np.linalg.inv(sw), (u0 - u1).T).reshape(1, -1) # (1, n_features)

        
        if plot:
            fig, ax = plt.subplots()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_position(('data', 0))
            ax.spines['bottom'].set_position(('data', 0))

            plt.scatter(X1[:, 0], X1[:, 1], c='k', marker='o', label='good')
            plt.scatter(X0[:, 0], X0[:, 1], c='r', marker='x', label='bad')

            plt.xlabel('Density', labelpad=1)
            plt.ylabel('Sugar content')
            plt.legend(loc='upper right')

            x_tmp = np.linspace(-0.05, 0.15)
            y_tmp = x_tmp * w[0, 1] / w[0, 0]
            plt.plot(x_tmp, y_tmp, '#808080', linewidth=1)

            wu = w / np.linalg.norm(w) # 对投影向量 w 进行单位化（归一化）处理
            
            # 正负样本点
            X0_project = np.dot(X0, np.dot(wu.T, wu))  # 投影到 wu上
            plt.scatter(X0_project[:, 0], X0_project[:, 1], c='r', s=15)
            for i in range(X0.shape[0]):
                plt.plot([X0[i, 0], X0_project[i, 0]], [X0[i, 1], X0_project[i, 1]], '--r', linewidth=1)
                
            X1_project = np.dot(X1, np.dot(wu.T, wu))
            plt.scatter(X1_project[:, 0], X1_project[:, 1], c='k', s=15)
            for i in range(X1.shape[0]):
                plt.plot([X1[i, 0], X1_project[i, 0]], [X1[i, 1], X1_project[i, 1]], '--k', linewidth=1)                    

            # 为中心投影点添加带箭头的文本标注
            u0_project = np.dot(u0, np.dot(wu.T, wu))
            plt.scatter(u0_project[:, 0], u0_project[:, 1], c='#FF4500', s=60)
            u1_project = np.dot(u1, np.dot(wu.T, wu))
            plt.scatter(u1_project[:, 0], u1_project[:, 1], c='#696969', s=60)
            
            ax.annotate(r'u0 Projection point',
                        xy = (u0_project[:, 0], u0_project[:, 1]),
                        xytext = (u0_project[:, 0] - 0.2, u0_project[:, 1] - 0.1),
                        size=13,
                        va='center',
                        ha='left',
                        arrowprops=dict(arrowstyle='->', color='k'))
            
            ax.annotate(r'u1 Projection point',
                        xy = (u1_project[:, 0], u1_project[:, 1]),
                        xytext = (u1_project[:, 0] - 0.2, u1_project[:, 1] - 0.1),
                        size=13,
                        va='center',
                        ha='left',
                        arrowprops=dict(arrowstyle='->', color='k'))
            plt.axis('equal')
            plt.show()
            
        self.w = w
        self.u0 = u0
        self.u1 = u1
        return self


    def predict(self, X):
        project = np.dot(X,self.w.T)
        wu0 = np.dot(self.w, self.u0.T)
        wu1 = np.dot(self.w, self.u1.T)
        return (np.abs(project - wu1) < np.abs(project - wu0)).astype(int)
    

if __name__ == '__main__':
    
    data_path = r'E:\CODE\machine-learning-notes\chapter-3\3.3\watermelon.csv'
    data = pd.read_csv(data_path).values
    
    X = data[:, 7:9].astype(float)
    y = data[:, 9]
    
    y[y == "是"] = 1
    y[y == "否"] = 0
    y = y.astype(int)
    
    lda = LDA()
    lda.fit(X, y, plot=True)
    print(lda.predict(X))
    print(y)