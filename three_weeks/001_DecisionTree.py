# coding:utf-8

#1. 对有连续特征的数据进行分类(基于决策树)
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.datasets import load_iris

iris=load_iris()
dir(iris)#查看数据集目录['DESCR', 'data', 'feature_names', 'target', 'target_names']
iris_feature_name=iris.feature_names
iris_feature=iris.data
iris_target_name=iris.target_names
iris_target=iris.target

iris_feature_name
iris_feature[:5,:]
iris_target_name
iris_target
iris_feature.shape

#构建模型
clf=tree.DecisionTreeClassifier(max_depth=4)
clf=clf.fit(iris_feature,iris_target)

import pydotplus
from IPython.display import  Image,display
dot_data=tree.export_graphviz(clf,
                             out_file=None,
                             feature_names=iris_feature_name,
                             class_names=iris_target_name,
                             filled=True,
                             rounded=True
                             )
graph=pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))







#2. 构建回归树
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn.datasets import load_boston
boston_house=load_boston()
boston_feature_name=boston_house.feature_names
boston_features=boston_house.data
boston_target=boston_house.target

print(boston_house.DESCR)

#构建模型
rgs=tree.DecisionTreeRegressor(max_depth=4)
rgs=rgs.fit(boston_features,boston_target)
import pydotplus
from IPython.display import  Image,display

dit_data = tree.export_graphviz(rgs,
                                out_file=None,
                                feature_names=boston_feature_name,
                                class_names=boston_target,
                                filled=True,
                                rounded=True

                                )
graph=pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))










import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# path to where the data lies
dpath = './data/'
data = pd.read_csv(dpath+"mushrooms.csv")
data.head()


#特征编码:对于类别型特征:如果无序则用onehot编码,类别有序则用labelencoder编码
#一般不能直接处理类别数据,需要转化成数值型
#特征全是类别型变量，很多模型需要数值型的输入（Logisstic回归、xgboost...)

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

#data.head()
#LableEncoder是不合适的，因为是有序的。
# 而颜色等特征是没有序关系。决策树等模型不care，但logistic回归不行。
# 也可以试试OneHotEncoder
#X = data.iloc[:,1:23]  # all rows, all the features and no labels
#y = data.iloc[:, 0]  # all rows, label only

y = data['class']    #用列名访问更直观
X = data.drop('class', axis = 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,,random_state=4)
columns = X_train.columns

# 数据标准化
from sklearn.preprocessing import StandardScaler

# 分别初始化对特征和目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

##  default Logistic Regression
from sklearn.linear_model import LogisticRegression

model_LR= LogisticRegression()
model_LR.fit(X_train,y_train)

# 看看各特征的系数，系数的绝对值大小可视为该特征的重要性
fs = pd.DataFrame({"columns":list(columns), "coef":list(abs(model_LR.coef_.T))})
fs.sort_values(by=['coef'],ascending=False)

y_prob = model_LR.predict_proba(X_test)[:, 1]  # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0)  # This will threshold the probabilities to give class predictions.

# accuracy
print 'The accuary of default Logistic Regression is', model_LR.score(X_test, y_pred)

print 'The AUC of default Logistic Regression is', roc_auc_score(y_test,y_pred)

'''
Logistic Regression(Tuned model)
logistic回归的需要调整超参数有：C（正则系数，一般在log域（取log后的值）均匀设置调优）和正则函数penalty（L2/L1） 目标函数为：J(theata) = sum(logloss(f(xi), yi)) + C * penalty logistic回归: f(xi) = sigmoid(sum(wj * xj)) logloss为负log似然损失（请见课件） L2 penalty：sum(wj^2) L1 penalty: sum(abs(wj))

在sklearn框架下，不同学习器的参数调整步骤相同：

设置候选参数集合
调用GridSearchCV
调用fit
'''
from sklearn.linear_model import LogisticRegression

LR_model= LogisticRegression()

#设置参数搜索范围（Grid，网格）
tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,
              'penalty':['l1','l2']
                   }

# fit函数执行会有点慢，因为要循环执行 参数数目 * CV折数 次模型 训练
LR= GridSearchCV(LR_model, tuned_parameters,cv=10)
LR.fit(X_train,y_train)
print(LR.best_params_)

y_prob = LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
LR.score(X_test, y_pred)

print 'The AUC of GridSearchCV Logistic Regression is', roc_auc_score(y_test,y_pred)





## Default Decision Tree model

from sklearn.tree import DecisionTreeClassifier

model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)
y_prob = model_tree.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_tree.score(X_test, y_pred)

print 'The AUC of default Desicion Tree is', roc_auc_score(y_test,y_pred)


df = pd.DataFrame({"columns":list(columns), "importance":list(model_tree.feature_importances_.T)})
df.sort_values(by=['importance'],ascending=False)

plt.bar(range(len(model_tree.feature_importances_)), model_tree.feature_importances_)
plt.show()


# 可根据特征重要性做特征选择
from numpy import sort
from sklearn.feature_selection import SelectFromModel

# Fit model using each importance as a threshold
thresholds = sort(model_tree.feature_importances_)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model_tree, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)

    # train model
    selection_model = DecisionTreeClassifier()
    selection_model.fit(select_X_train, y_train)

    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],
                                                   accuracy * 100.0))

    # Fit model using the best threshhold
thresh = 0.020
selection = SelectFromModel(model_tree, threshold=thresh, prefit=True)
select_X_train = selection.transform(X_train)

# train model
selection_model = DecisionTreeClassifier()
selection_model.fit(select_X_train, y_train)

import graphviz
from sklearn import tree
tree.export_graphviz(model_tree, out_file='best_tree.dot')
#$ dot -Tpng best_tree.dot -o best_tree.png

#import matplotlib.image as mpimg # mpimg 用于读取图片
#tree_omg = mpimg.imread('best_tree.png')
#plt.imshow(tree_omg) # 显示图片
#plt.axis('off') # 不显示坐标轴
#plt.show()
#pip install Pillow
from PIL import Image
img=Image.open('best_tree.png')
plt.imshow(img)
plt.show()
'''
Let us tune the hyperparameters of the Decision tree model
决策树的超参数有：max_depth（树的深度）或max_leaf_nodes（叶子结点的数目）、max_features（最大特征数目）、min_samples_leaf（叶子结点的最小样本数）、min_samples_split（中间结点的最小样本树）、min_weight_fraction_leaf（叶子节点的样本权重占总权重的比例） min_impurity_split（最小不纯净度）也可以调整

这个数据集的任务不难，深度设为2-10之间 两类分类问题，训练样本每类样本在3000左右，所以min_samples_leaf

'''

from sklearn.tree import DecisionTreeClassifier

model_DD = DecisionTreeClassifier()

max_depth = range(1,10,1)
min_samples_leaf = range(1,10,2)
tuned_parameters = dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf)

from sklearn.model_selection import GridSearchCV
DD = GridSearchCV(model_DD, tuned_parameters,cv=10)
DD.fit(X_train, y_train)

print("Best: %f using %s" % (DD.best_score_, DD.best_params_))

y_prob = DD.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
DD.score(X_test, y_pred)

DD.grid_scores_

# DD.grid_scores_

test_means = DD.cv_results_['mean_test_score']
# test_stds = DD.cv_results_[ 'std_test_score' ]
# pd.DataFrame(DD.cv_results_).to_csv('DD_min_samples_leaf_maxdepth.csv')

# plot results
test_scores = np.array(test_means).reshape(len(max_depth), len(min_samples_leaf))

for i, value in enumerate(max_depth):
    plt.plot(min_samples_leaf, test_scores[i], label='test_max_depth:' + str(value))

plt.legend()
plt.xlabel('min_samples_leaf')
plt.ylabel('accuray')
plt.show()































