## 3.1

当满足以下情况时，在使用线性模型 \( f(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b \) 时可以不用考虑偏置项 \( b \) ：

### 数据经过中心化处理
如果对输入的特征数据 \( \boldsymbol{x} \) 进行了中心化操作，即让所有特征的均值为 \( 0 \) 。数学上，若原始特征为 \( x_1, x_2, \dots, x_d \) ，经过中心化后变为 \( \tilde{x}_i = x_i - \mu_i \)（其中 \( \mu_i \) 是第 \( i \) 个特征的均值，且 \( \sum_{i = 1}^d \mu_i = 0 \) 整体满足某种中心对齐 ），此时在构建线性模型时，偏置项 \( b \) 的作用可被融入到数据的这种变换中，模型可简化为 \( f(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} \)  ，因为数据本身的“基准”已经通过中心化调整，不需要额外的偏置去修正预测的起始位置。比如在一些基于均值为0的标准化特征工程场景中，常可省略偏置项 。

### 从模型假设角度，若认为数据分布过原点
如果从业务逻辑或先验知识判断，所研究的数据规律本身满足当所有特征取值为 \( 0 \) 时，预测结果也应为 \( 0 \) ，也就是数据分布过原点 \( (0,0,\dots,0) \) 。例如，研究“物体质量与重量的线性关系”（在同一重力环境下，重量 = 质量×重力加速度 ），当质量为 \( 0 \) 时，重量必然为 \( 0 \) ，构建线性模型 \( f(\text{质量}) = w\times\text{质量} + b \) 时，就可认为 \( b = 0 \) ，直接用 \( f(\text{质量}) = w\times\text{质量} \)  ，因为符合实际物理规律下的原点性假设。 

不过，在大多数实际的机器学习、数据分析场景中，数据不一定满足上述严格条件，偏置项 \( b \) 常用来调整模型预测的“截距”，让模型能更好适配数据整体的偏移情况，所以是否省略要依据具体数据和问题背景判断 。 


## 3.2
### 一、关于式子\( y = \frac{1}{1 + e^{-(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b)}} \)凸性分析
要判断式子 \( y = \frac{1}{1 + e^{-(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b)}} \)（逻辑斯蒂回归预测函数，基于对数几率函数 ）的凸性，需从不同视角分析：

#### （一）单看预测函数自身（以\( z = \boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b \)为变量 ）
令\( z = \boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b \)，函数为\( y = \frac{1}{1 + e^{-z}} \)（Sigmoid函数 ）。通过求二阶导数判断：
- **一阶导数**：根据求导公式，\( y^\prime = \frac{e^{-z}}{(1 + e^{-z})^2} = y(1 - y) \)  。 
- **二阶导数**：对\( y^\prime \)再求导，\( y^{\prime\prime} = y^\prime(1 - y) + y(-y^\prime)= y^\prime(1 - 2y) \)，把\( y^\prime = y(1 - y) \)代入，得\( y^{\prime\prime} = y(1 - y)(1 - 2y) \) 。  
由于\( y \in (0,1) \)，当\( y \in (0, 0.5) \)时，\( y^{\prime\prime}>0 \)；当\( y = 0.5 \)时，\( y^{\prime\prime}=0 \)；当\( y \in (0.5, 1) \)时，\( y^{\prime\prime}<0 \) 。所以**单变量下，Sigmoid函数（对数几率函数 ）本身不是凸函数，也不是凹函数，是一个“S”形的非凸非凹函数** 。

#### （二）逻辑斯蒂回归完整场景（看关于参数\( \boldsymbol{w}, b \)的函数凸性 ）
逻辑斯蒂回归中，关注**经验风险（损失函数 ）关于模型参数\( \boldsymbol{w}, b \)的凸性** 。采用交叉熵损失（对数损失 ）：\( L = -\frac{1}{N}\sum_{i = 1}^N [y_i\ln\hat{y}_i + (1 - y_i)\ln(1 - \hat{y}_i)] \) ，其中\( \hat{y}_i = \frac{1}{1 + e^{-(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x}_i + b)}} \) ，\( N \)是样本数量，\( y_i \)是真实标签（\( 0 \)或\( 1 \) ）。  
对这个损失函数关于\( \boldsymbol{w} \)和\( b \)求二阶导（或看Hessian矩阵是否半正定 ），其Hessian矩阵是半正定的，所以**逻辑斯蒂回归的损失函数关于参数\( \boldsymbol{w}, b \)是凸函数** ，这是逻辑斯蒂回归能通过梯度下降等凸优化方法求解的理论基础。但单论式子\( y = \frac{1}{1 + e^{-(\boldsymbol{w}^{\mathrm{T}}\boldsymbol{x} + b)}} \)这个预测函数本身（不结合损失 ），从函数凸性定义看，它不是凸的 。 

简言之，若问式子作为预测函数自身是否凸（把\( \boldsymbol{w}, \boldsymbol{x}, b \)等当变量看 ），不是凸函数；若问逻辑斯蒂回归里结合损失后关于参数的凸性，损失函数是凸的，需区分不同视角 。通常说逻辑斯蒂回归“凸”，指损失函数关于参数凸，预测函数本身（式子）不是凸的 。 

### 二、对数似然函数\( \ell(\boldsymbol{\beta}) = \sum_{i=1}^{m} \left( -y_i \boldsymbol{\beta}^\mathrm{T} \boldsymbol{\tilde{x}}_i + \ln\left(1 + e^{\boldsymbol{\beta}^\mathrm{T} \boldsymbol{\tilde{x}}_i}\right) \right) \)凹凸性分析
要判断该对数似然函数的凹凸性，通过分析其**二阶导数（Hessian矩阵）的正定性/半正定性**确定，过程如下：

#### （一）符号与基本设定  
记\( z_i = \boldsymbol{\beta}^\mathrm{T} \boldsymbol{\tilde{x}}_i \)（第\( i \)个样本的线性预测值 ），函数简化为：  
\[
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{m} \left( -y_i z_i + \ln\left(1 + e^{z_i}\right) \right)
\]  
需对\( \boldsymbol{\beta} \)求二阶导（Hessian矩阵\( \boldsymbol{H} \) ），判断是否半正定（凸函数）或半负定（凹函数）。  


#### （二）一阶导数（梯度）  
对\( \ell(\boldsymbol{\beta}) \)关于\( \boldsymbol{\beta} \)求一阶偏导（梯度\( \nabla \ell \) ）：  
单个样本项\( -y_i z_i + \ln(1 + e^{z_i}) \)对\( \boldsymbol{\beta} \)的导数为：  
\[
\frac{\partial}{\partial \boldsymbol{\beta}} \left( -y_i z_i + \ln(1 + e^{z_i}) \right) 
= -y_i \boldsymbol{\tilde{x}}_i + \frac{e^{z_i}}{1 + e^{z_i}} \boldsymbol{\tilde{x}}_i 
= \left( \sigma(z_i) - y_i \right) \boldsymbol{\tilde{x}}_i 
\]  
其中\( \sigma(z_i) = \frac{1}{1 + e^{-z_i}} \)是Sigmoid函数（因\( \frac{e^{z_i}}{1 + e^{z_i}} = \sigma(z_i) \)  ）。  

整体梯度为：  
\[
\nabla \ell(\boldsymbol{\beta}) = \sum_{i=1}^{m} \left( \sigma(z_i) - y_i \right) \boldsymbol{\tilde{x}}_i 
\]  


#### （三）二阶导数（Hessian矩阵）  
对梯度再关于\( \boldsymbol{\beta} \)求导，得Hessian矩阵\( \boldsymbol{H} \) 。  

单个样本项\( \left( \sigma(z_i) - y_i \right) \boldsymbol{\tilde{x}}_i \)对\( \boldsymbol{\beta} \)的二阶导数为：  
\[
\frac{\partial^2}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^\mathrm{T}} \left( \left( \sigma(z_i) - y_i \right) \boldsymbol{\tilde{x}}_i \right) 
= \boldsymbol{\tilde{x}}_i \boldsymbol{\tilde{x}}_i^\mathrm{T} \cdot \sigma(z_i) \left( 1 - \sigma(z_i) \right)
\]  
推导依据：  
- \( \sigma(z) \)的导数为\( \sigma^\prime(z) = \sigma(z)(1 - \sigma(z)) \) ；  
- \( z_i = \boldsymbol{\beta}^\mathrm{T} \boldsymbol{\tilde{x}}_i \)对\( \boldsymbol{\beta} \)的导数是\( \boldsymbol{\tilde{x}}_i \) ，用乘积法则得二阶导数。  


整体Hessian矩阵为：  
\[
\boldsymbol{H} = \sum_{i=1}^{m} \sigma(z_i) \left( 1 - \sigma(z_i) \right) \boldsymbol{\tilde{x}}_i \boldsymbol{\tilde{x}}_i^\mathrm{T}
\]  



#### （四）凹凸性判断：Hessian半正定性  
观察\( \boldsymbol{H} \)结构：  
- 每一项\( \sigma(z_i)(1 - \sigma(z_i)) \)是正数（因\( \sigma(z) \in (0,1) \) ，故\( \sigma(z)(1 - \sigma(z)) > 0 \) ）；  
- \( \boldsymbol{\tilde{x}}_i \boldsymbol{\tilde{x}}_i^\mathrm{T} \)是半正定矩阵（对任意向量\( \boldsymbol{v} \) ，有\( \boldsymbol{v}^\mathrm{T} (\boldsymbol{\tilde{x}}_i \boldsymbol{\tilde{x}}_i^\mathrm{T}) \boldsymbol{v} = (\boldsymbol{v}^\mathrm{T} \boldsymbol{\tilde{x}}_i)^2 \geq 0 \) ）。  

因此，\( \boldsymbol{H} \)是多个半正定矩阵的正系数加权和，根据半正定矩阵性质：**半正定矩阵的正系数加权和仍为半正定矩阵**。  


#### 结论  
对数似然函数\( \ell(\boldsymbol{\beta}) \)的Hessian矩阵是半正定的，因此：  

\[
\boldsymbol{\ell}(\boldsymbol{\beta}) \text{ 是关于 } \boldsymbol{\beta} \text{ 的凸函数}
\]  

这是逻辑回归能通过梯度下降、牛顿法等凸优化方法高效求解的理论基础——**凸函数的局部最优解即为全局最优解**。 

## 3.3 编程实现对率回归，并给出西瓜数据集上的结果
