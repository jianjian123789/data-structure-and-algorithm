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


