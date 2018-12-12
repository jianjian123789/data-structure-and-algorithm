# coding:utf-8

#熟悉各中聚类算法的调用 并用评价指标选择合适的超参数
#在使用K-means前,如果数据的维度很高的话我们可以先使用PCA进行降维操作


#导入必要的工具包
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.decomposition import PCA
import time

import matplotlib.pyplot as plt
%matplotlib inline


#读取训练数据
train = pd.read_csv('./data/MNIST_train.csv')

n_trains = 1000
y_train = train.label.values[:n_trains]
X_train = train.drop("label",axis=1).values[:n_trains]

#将像素值[0,255]  --> [0,1]
X_train = X_train / 255.0


#对数据进行PCA降维
pca = PCA(n_components=0.75)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)

# 降维后的特征维数
print(X_train_pca.shape)

# 将训练集合拆分成训练集和校验集，在校验集上找到最佳的模型超参数（PCA的维数）
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train_pca,y_train, train_size = 0.8,random_state = 0)


# 一个参数点（聚类数据为K）的模型，在校验集上评价聚类算法性能
def K_cluster_analysis(K, X_train, y_train, X_val, y_val):
    start = time.time()

    print("K-means begin with clusters: {}".format(K));

    # K-means,在训练集上训练
    mb_kmeans = MiniBatchKMeans(n_clusters=K)
    mb_kmeans.fit(X_train)

    # 在训练集和测试集上测试
    # y_train_pred = mb_kmeans.fit_predict(X_train)
    y_val_pred = mb_kmeans.predict(X_val)

    # 以前两维特征打印训练数据的分类结果
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred)
    # plt.show()

    # K值的评估标准
    # 常见的方法有轮廓系数Silhouette Coefficient和Calinski-Harabasz Index
    # 这两个分数值越大则聚类效果越好
    # CH_score = metrics.calinski_harabaz_score(X_train,mb_kmeans.predict(X_train))
    CH_score = metrics.silhouette_score(X_train, mb_kmeans.predict(X_train))

    # 也可以在校验集上评估K
    v_score = metrics.v_measure_score(y_val, y_val_pred)

    end = time.time()
    print("CH_score: {}, time elaps:{}".format(CH_score, int(end - start)))
    print("v_score: {}".format(v_score))

    return CH_score, v_score


# 设置超参数（聚类数目K）搜索范围
Ks = [10, 20, 30,40,50,60]
CH_scores = []
v_scores = []
for K in Ks:
    ch,v = K_cluster_analysis(K, X_train_part, y_train_part, X_val, y_val)
    CH_scores.append(ch)
    v_scores.append(v)

# 绘制不同PCA维数下模型的性能，找到最佳模型／参数（分数最高）
plt.plot(Ks, np.array(CH_scores), 'b-')

plt.plot(Ks, np.array(v_scores), 'g-')


#显示聚类结果
#画出聚类结果，每一类用一种颜色
colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']

n_clusters = 10
mb_kmeans = MiniBatchKMeans(n_clusters = n_clusters)
mb_kmeans.fit(X_train_pca)

y_train_pred = mb_kmeans.labels_
cents = mb_kmeans.cluster_centers_#质心

for i in range(n_clusters):
    index = np.nonzero(y_train_pred==i)[0]
    x1 = X_train_pca[index,0]
    x2 = X_train_pca[index,1]
    y_i = y_train[index]
    for j in range(len(x1)):
        if j < 20:  #每类打印20个
            plt.text(x1[j],x2[j],str(int(y_i[j])),color=colors[i],\
                fontdict={'weight': 'bold', 'size': 9})
    #plt.scatter(cents[i,0],cents[i,1],marker='x',color=colors[i],linewidths=12)

plt.axis([-5,10,-6,6])
plt.show()















































































