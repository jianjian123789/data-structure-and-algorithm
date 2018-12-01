# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as  sns

# 1 读取数据
data=pd.read_csv('boston_housing.csv')

# 1.1 数据探索
data.head()#打印前五行
data.info()#特征描述
data.isnull().sum()#统计空值个数
data.describe()#统计数字特征信息:样本数目/均值/标准差/四分位数
data.shape
data = data[data.MEDV < 50]# 删除y大于50的样本


# 1.2 图探索:单个特征
#直方图:连续型特征使用distplot
fig = plt.figure()
sns.distplot(data.MEDV.values, bins=30, kde=True)
plt.xlabel('Median value of owner-occupied homes', fontsize=12)
plt.show()

fig = plt.figure()
sns.distplot(data.CRIM.values, bins=30, kde=False)
plt.xlabel('crime rate', fontsize=12)
plt.show()

#直方图:离散性特征使用countplot
sns.countplot(data.CHAS, order=[0, 1]);
plt.xlabel('Charles River');
plt.ylabel('Number of occurrences');

# 单个特征散点图
plt.scatter(range(data.shape[0]), data["MEDV"].values,color='purple')
plt.title("Distribution of Price");

# 1.3 两两特征
# Pearson相关系数图
#get the names of all the columns
cols=data.columns
# Calculates pearson co-efficient for all combinations，通常认为相关系数大于0.5的为强相关
data_corr = data.corr().abs()
data_corr.shape
plt.subplots(figsize=(13, 9))
sns.heatmap(data_corr,annot=True)
# Mask unimportant features
sns.heatmap(data_corr, mask=data_corr < 1, cbar=False)
plt.savefig('house_coor.png' )
plt.show()

# Pearson相关系数排序
#Set the threshold to select only highly correlated attributes
threshold = 0.5
# List of pairs along with correlation above threshold
corr_list = []
#size = data.shape[1]
size = data_corr.shape[0]
#Search for the highly correlated pairs
for i in range(0, size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index
#Sort to show higher ones first
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))
#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))


# 画出相关系数高的特征散点图
# Scatter plot of only the highly correlated pairs
for v,i,j in s_corr_list:
    sns.pairplot(data, size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()

