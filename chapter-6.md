# 支持向量机

## 支持向量
距离超平面最近的这几个训练样本点使得训练样本正确分类不等式组的等号成立，
它们被称为“支持向量”，两个异类支持向量到超平面的距离只和被称为 **“间隔”**
要找到具有“最大间隔”的划分超平面，也就是最小化w的欧氏距离

### 求解SVM基本型
* 对偶问题
拉格朗日乘子法（乘子a）得到拉格朗日函数，对L的w和b偏导为0带入L消去w和b，考虑约束，就得到了SVM基本型的对偶问题
解出对偶问题的a是拉格朗日乘子，对应着训练样本，上述过程满足KKT条件
* SVM重要性质
训练完成后，大部分的训练样本都不需要保留，最终模型仅仅与支持向量有关
* 对偶问题的高效算法
**SMO**

## 核函数
* 将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线形可分
* 同样，对其对偶问题，定义核函数k(·，·)，模型的最优解可以通过训练样本的核函数展开，称作“支持向量展开式”

## 软间隔与正则化
现实任务中往往很难确定适合的核函数是的训练样本在特征空间线形可分；即便恰好找到某个核函数，也很难断定这个结果是不是由于过拟合造成的。
* 软间隔 \
允许支持向量机在一些样本上出错，在最大化间隔的同时，不满足约束的样本应尽量少，于是优化目标的函数可以写为a由损失函数代替，变成了“软间隔支持向量机”
* 损失函数 \
损失函数常常是凸的连续函数而且是l的上界（hinge损失、指数损失、对率损失）
软间隔支持向量机的最终模型仅与支持向量有关，即通过hinge损失函数仍保持了稀疏性。
对率回归的优势在于其输出具有自然的概率意义，即在给出预测标记的同时也给出了概率；能直接用于多分类任务
hinge损失有一块“平坦”的0 区，使得支持向量机有很好的稀疏性，而对率损失光滑递减，不能导出类似支持向量的概念，因此对率回归的解依赖于更多的训练样本，其预测开销更大。
* 结构风险与经验风险
模型的性质与所用的替代函数直接相关，有一个共性：优化目标中的第一项用来描述划分超平面“间隔”大小，称为”结构风险“，另一项用来表述训练集上的误差，称为”经验风险“，参数 C 对二者进行折中

## 支持向量回归
e-间隔带，落入其中的样本不计算损失，SVR问题形式化，引入松弛变量，再形式化，引入拉格朗日乘子，得到拉格朗日函数

## 核方法
核函数直接决定了支持向量机与核方法的最终性能，核选择是一个未知问题，可以通过多核学习获得最优凸组合作为最终的核函数，借助了集成学习机制。






