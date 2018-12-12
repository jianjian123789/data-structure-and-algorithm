# coding:utf-8

# 图像数据维数高，而且特征之间（像素之间）相关性很高，因此我们预计用很少的维数就能保留足够多的信息

#导入必要的工具包
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import time

#读取训练数据和测试数据
train = pd.read_csv('./data/MNIST_train.csv')
test = pd.read_csv('./data/MNIST_test.csv')

y_train = train.label.values
X_train = train.drop("label",axis=1).values
X_test = test.values

#将像素值[0,255]  --> [0,1]
X_train = X_train / 255.0
X_test = X_test / 255.0


# 将训练集合拆分成训练集和校验集，在校验集上找到最佳的模型超参数（PCA的维数）
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train,y_train, train_size = 0.8,random_state = 0)


# 一个参数点（PCA维数为n）的模型训练和测试，得到该参数下模型在校验集上的预测性能
def n_component_analysis(n, X_train, y_train, X_val, y_val):
    start = time.time()

    pca = PCA(n_components=n)
    print("PCA begin with n_components: {}".format(n));
    pca.fit(X_train)

    # 在训练集和测试集降维
    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)

    # 利用SVC训练
    print('SVC begin')
    clf1 = svm.SVC()
    clf1.fit(X_train_pca, y_train)

    # 返回accuracy
    accuracy = clf1.score(X_val_pca, y_val)

    end = time.time()
    print("accuracy: {}, time elaps:{}".format(accuracy, int(end - start)))
    return accuracy


# 2.找出最佳主成分
# 设置超参数（PCA维数）搜索范围
n_s = np.linspace(0.70, 0.85, num=15)
accuracy = []
for n in n_s:
    tmp = n_component_analysis(n, X_train_part, y_train_part, X_val, y_val)
    accuracy.append(tmp)


# 绘制不同PCA维数下模型的性能，找到最佳模型／参数（分数最高）
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(n_s, np.array(accuracy), 'b-')



# 3.在最佳主成分下重新进行训练
#最佳模型参数
pca = PCA(n_components=0.75)

#根据最佳参数，在全体训练数据上重新训练模型
pca.fit(X_train)

pca.n_components_

pca.explained_variance_ratio_

#根据最佳参数，对全体训练数据降维
X_train_pca = pca.transform(X_train)

#根据最佳参数，对测试数据降维
X_test_pca = pca.transform(X_test)

#在降维后的训练数据集上训练SVM分类器
clf = svm.SVC()
clf.fit(X_train_pca, y_train)

# 用在降维后的全体训练数据集上训练的模型对测试集进行测试
y_predict = clf.predict(X_test_pca)

#生成提交测试结果
import pandas as pd
df = pd.DataFrame(y_predict)
df.columns=['Label']
df.index+=1
df.index.name = 'Imageid'
df.to_csv('SVC_Minist_submission.csv', header=True)












