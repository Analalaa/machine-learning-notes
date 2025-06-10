## 线性模型
拥有很好的可解释性
w b 是权重参数和偏置
### 线形回归
线形回归试图学得一个线形模型以尽可能准确地预测实际输出标记
衡量f(x)与y之间的差别来确定w和b
其中一个常用的性能度量就是均方误差，拥有很好的几何意义，对应了欧氏距离
基于均方误差最小化来进行模型求解的方法称为“最小二乘法”，求解w和b使距离和最小化的过程是线形回归模型的最小二乘“参数估计”
* 对于多元线形回归 
对于 $X^TX$为满秩矩阵或正定矩阵可以令求导为0
如果不满足满秩矩阵的条件，常见的做法是引入正则化项
* 对数线性回归 
试图让e为底，线形形式为指数来逼近y，实质上是在输入空间到输出空间的非线性函数映射

#### 对数几率回归
回归到分类的转变，只需要一个单调可微的函数将分类任务的真实标记y与线性回归的预测值联系起来 \
单位阶跃函数能够联系起来标记与预测值，但是不连续，就找到了对数几率函数(ligistic function)作为代替 \
将对数几率函数作为 $g^{-1}(z)$ 代入公式，得到：

$$
y = \frac{1}{1 + e^{-z}}
$$

将 $z = \boldsymbol{w}^T \boldsymbol{x} + b$ 代入上式得：

$$
y = \frac{1}{1 + e^{-(\boldsymbol{w}^T \boldsymbol{x} + b)}}
$$

$$
\ln \frac{y}{1 - y} = \boldsymbol{w}^T \boldsymbol{x} + b
$$

若将 $y$ 视为样本 $\boldsymbol{x}$ 作为正例的概率，则 $1 - y$ 是其反例的概率，两者的比值：

$$
\frac{y}{1 - y}
$$

称为 **几率** (*odds*)，反映了 $\boldsymbol{x}$ 作为正例的相对可能性。对几率取对数得到 **对数几率**（*log odds*，亦称 *logit*）：

$$
\ln \frac{y}{1 - y} \
$$

---
由此可看出，实质上是使用线性回归模型的预测结果去通过 Sigmoid 函数变换，预测输出为介于 0 和 1 之间的概率值，这种方法称为 **逻辑回归**（*logistic regression*，亦称 *logit regression*）。

特别需注意的是，虽然它名字里含“回归”，但其实是一种分类学习方法。这种方法有很多优点，例如它建立在对数几率函数的基础上，模型具有良好的概率解释性，并且模型形式简单，计算代价小，适合高维稀疏特征情形；同时对输入变量不作分布假设，因此对离散特征和连续特征都能很好兼容；此外，对数几率函数在 $z = 0$ 附近有较大梯度，方便优化算法收敛，因此在很多实际应用中被广泛采用。

用二分类的后验估计重写对数几率函数，能够分别表示出每一个种分类的后验概率 \
于是就可以通过“极大似然法”来估计 w 和 b，对似然函数（高阶连续可导凸函数），根据梯度下降或者牛顿法都可以得到最优解

### 📘 逻辑回归最大似然函数与梯度下降法

我们希望估计参数 $\mathbf{w}$ 和 $b$，通过最大化对数似然函数（log-likelihood）：

给定数据集 $\{(\mathbf{x}_i, y_i)\}_{i=1}^m$，其中 $y_i \in \{0, 1\}$，则逻辑回归模型的预测概率为：

$$
p(y = 1 | \mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}} = \sigma(\mathbf{w}^T \mathbf{x} + b)
$$

其中 $\sigma(z)$ 是 sigmoid 函数。

定义对数似然函数为：

$$
\ell(\mathbf{w}, b) = \sum_{i=1}^m \ln p(y_i | \mathbf{x}_i; \mathbf{w}, b) \tag{1}
$$

对每个样本 $(\mathbf{x}_i, y_i)$，有：

$$
p(y_i | \mathbf{x}_i; \mathbf{w}, b) = \sigma(\mathbf{w}^T \mathbf{x}_i + b)^{y_i} (1 - \sigma(\mathbf{w}^T \mathbf{x}_i + b))^{1 - y_i}
$$

因此对数似然函数可以写为：

$$
\ell(\mathbf{w}, b) = \sum_{i=1}^m \left[ y_i \ln \sigma(\mathbf{w}^T \mathbf{x}_i + b) + (1 - y_i) \ln (1 - \sigma(\mathbf{w}^T \mathbf{x}_i + b)) \right] \tag{2}
$$

### 🧮 梯度下降法优化

我们通常最大化对数似然函数，或者最小化其负数（即负对数似然，作为损失函数）：

$$
\mathcal{L}(\mathbf{w}, b) = - \ell(\mathbf{w}, b) \tag{3}
$$

对参数求导，可以得到梯度下降的更新公式：

- 对 $\mathbf{w}$ 的梯度：

$$
\nabla_{\mathbf{w}} \mathcal{L} = \sum_{i=1}^m \left[ \sigma(\mathbf{w}^T \mathbf{x}_i + b) - y_i \right] \mathbf{x}_i \tag{4}
$$

- 对 $b$ 的梯度：

$$
\frac{\partial \mathcal{L}}{\partial b} = \sum_{i=1}^m \left[ \sigma(\mathbf{w}^T \mathbf{x}_i + b) - y_i \right] \tag{5}
$$

使用梯度下降更新参数：

```python
w -= learning_rate * grad_w
b -= learning_rate * grad_b
