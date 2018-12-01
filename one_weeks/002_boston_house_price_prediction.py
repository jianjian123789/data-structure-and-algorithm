# coding:utf-8

import numpy as np  # 矩阵操作
import pandas as pd # SQL数据处理

from sklearn.metrics import r2_score  #评价回归预测模型的性能

import matplotlib.pyplot as plt   #画图
import seaborn as sns

# 图形出现在Notebook里而不是新窗口
%matplotlib inline


data = pd.read_csv("boston_housing.csv")

# 从原始数据中分离输入特征x和输出y
y = data['MEDV'].values
X = data.drop('MEDV', axis = 1)

#用于后续显示权重系数对应的特征
columns = X.columns
#将数据分割训练数据与测试数据
from sklearn.model_selection import train_test_split

# 随机采样20%的数据构建测试样本，其余作为训练样本
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.2)
X_train.shape



#发现各特征差异较大，需要进行数据标准化预处理
#标准化的目的在于避免原始特征值差异过大，导致训练得到的参数权重不归一，无法比较各特征的重要性
# 数据标准化
from sklearn.preprocessing import StandardScaler

# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

#对y做标准化不是必须
#对y标准化的好处是不同问题的w差异不太大，同时正则参数的范围也有限
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))




"""
(1):默认OLS线性回归模型
"""
#class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化
lr = LinearRegression()

# 训练模型参数
lr.fit(X_train, y_train)

# 预测
y_test_pred_lr = lr.predict(X_test)
y_train_pred_lr = lr.predict(X_train)


# 看看各特征的权重系数，系数的绝对值大小可视为该特征的重要性
fs = pd.DataFrame({"columns":list(columns), "coef":list((lr.coef_.T))})
fs.sort_values(by=['coef'],ascending=False)

# 使用r2_score评价模型在测试集和训练集上的性能，并输出评估结果
#测试集
print 'The r2 score of LinearRegression on test is', r2_score(y_test, y_test_pred_lr)
#训练集
print 'The r2 score of LinearRegression on train is', r2_score(y_train, y_train_pred_lr)
#在训练集上观察预测残差的分布，看是否符合模型假设：噪声为0均值的高斯噪声
f, ax = plt.subplots(figsize=(7, 5))
f.tight_layout()
ax.hist(y_train - y_train_pred_lr,bins=40, label='Residuals Linear', color='b', alpha=.5);
ax.set_title("Histogram of Residuals")
ax.legend(loc='best');


#还可以观察预测值与真值的散点图
plt.figure(figsize=(4, 3))
plt.scatter(y_train, y_train_pred_lr)
plt.plot([-3, 3], [-3, 3], '--k')   #数据已经标准化，3倍标准差即可
plt.axis('tight')
plt.xlabel('True price')
plt.ylabel('Predicted price')
plt.tight_layout()






"""
(2):针对大数据集合时,使用随机梯度下降的回归算法
"""
# 线性模型，随机梯度下降优化模型参数
# 随机梯度下降一般在大数据集上应用，其实本项目不适合用
from sklearn.linear_model import SGDRegressor

# 使用默认配置初始化线
sgdr = SGDRegressor(max_iter=1000)

# 训练：参数估计
sgdr.fit(X_train, y_train)

# 预测
#sgdr_y_predict = sgdr.predict(X_test)

sgdr.coef_
# 使用SGDRegressor模型自带的评估模块(评价准则为r2_score)，并输出评估结果
print 'The value of default measurement of SGDRegressor on test is', sgdr.score(X_test, y_test)
print 'The value of default measurement of SGDRegressor on train is', sgdr.score(X_train, y_train)



"""
(3):使用L2正则
"""

#岭回归／L2正则
#class sklearn.linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True,
#                                  normalize=False, scoring=None, cv=None, gcv_mode=None,
#                                  store_cv_values=False)
from sklearn.linear_model import  RidgeCV

#设置超参数（正则参数）范围
alphas = [ 0.01, 0.1, 1, 10,100]

#生成一个RidgeCV实例
ridge = RidgeCV(alphas=alphas, store_cv_values=True)

#模型训练
ridge.fit(X_train, y_train)

#预测
y_test_pred_ridge = ridge.predict(X_test)
y_train_pred_ridge = ridge.predict(X_train)


# 评估，使用r2_score评价模型在测试集和训练集上的性能
print 'The r2 score of RidgeCV on test is', r2_score(y_test, y_test_pred_ridge)
print 'The r2 score of RidgeCV on train is', r2_score(y_train, y_train_pred_ridge)


mse_mean = np.mean(ridge.cv_values_, axis = 0)
plt.plot(np.log10(alphas), mse_mean.reshape(len(alphas),1))

#这是为了标出最佳参数的位置，不是必须
#plt.plot(np.log10(ridge.alpha_)*np.ones(3), [0.28, 0.29, 0.30])

plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print ('alpha is:', ridge.alpha_)

# 看看各特征的权重系数，系数的绝对值大小可视为该特征的重要性
fs = pd.DataFrame({"columns":list(columns), "coef_lr":list((lr.coef_.T)), "coef_ridge":list((ridge.coef_.T))})
fs.sort_values(by=['coef_lr'],ascending=False)



"""
(4):使用L1正则
"""
#### Lasso／L1正则
# class sklearn.linear_model.LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True,
#                                    normalize=False, precompute=’auto’, max_iter=1000,
#                                    tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=1,
#                                    positive=False, random_state=None, selection=’cyclic’)
from sklearn.linear_model import LassoCV

#设置超参数搜索范围
alphas = [ 0.01, 0.1, 1, 10,100]

#生成一个LassoCV实例
lasso = LassoCV(alphas=alphas)

#训练（内含CV）
lasso.fit(X_train, y_train)

#测试
y_test_pred_lasso = lasso.predict(X_test)
y_train_pred_lasso = lasso.predict(X_train)


# 评估，使用r2_score评价模型在测试集和训练集上的性能
print 'The r2 score of LassoCV on test is', r2_score(y_test, y_test_pred_lasso)
print 'The r2 score of LassoCV on train is', r2_score(y_train, y_train_pred_lasso)

mses = np.mean(lasso.mse_path_, axis=1)
plt.plot(np.log10(lasso.alphas_), mses)
# plt.plot(np.log10(lasso.alphas_)*np.ones(3), [0.3, 0.4, 1.0])
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print ('alpha is:', lasso.alpha_)

# 看看各特征的权重系数，系数的绝对值大小可视为该特征的重要性
fs = pd.DataFrame({"columns": list(columns), "coef_lr": list((lr.coef_.T)), "coef_ridge": list((ridge.coef_.T)),
                   "coef_lasso": list((lasso.coef_.T))})
fs.sort_values(by=['coef_lr'], ascending=False)


"""
(4):L1正则:扩大超参数的搜索
"""
#### Lasso／L1正则
# class sklearn.linear_model.LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True,
#                                    normalize=False, precompute=’auto’, max_iter=1000,
#                                    tol=0.0001, copy_X=True, cv=None, verbose=False, n_jobs=1,
#                                    positive=False, random_state=None, selection=’cyclic’)
from sklearn.linear_model import LassoCV

#设置超参数搜索范围
alphas = [0.00001, 0.0001, 0.001, 0.01]

#生成一个LassoCV实例
lasso = LassoCV(alphas=alphas)

#训练（内含CV）
lasso.fit(X_train, y_train)

#测试
y_test_pred_lasso = lasso.predict(X_test)
y_train_pred_lasso = lasso.predict(X_train)


# 评估，使用r2_score评价模型在测试集和训练集上的性能
print 'The r2 score of LassoCV on test is', r2_score(y_test, y_test_pred_lasso)
print 'The r2 score of LassoCV on train is', r2_score(y_train, y_train_pred_lasso)

mses = np.mean(lasso.mse_path_, axis=1)
plt.plot(np.log10(lasso.alphas_), mses)
# plt.plot(np.log10(lasso.alphas_)*np.ones(3), [0.3, 0.4, 1.0])
plt.xlabel('log(alpha)')
plt.ylabel('mse')
plt.show()

print ('alpha is:', lasso.alpha_)





