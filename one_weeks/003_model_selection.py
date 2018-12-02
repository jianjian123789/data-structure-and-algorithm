# coding:utf-8

df=pd.read_csv('diabetes.csv')


from sklearn import model_selection
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import warnings


### 1.投票器模型融合
array=df.values

X=array[:,0:8]
Y=array[:,8]
kfold=model_selection.KFold(n_splits=5,random_state=2018)

# 创建投票器的子模型
estimators=[]
model_1=LogisticRegression()
estimators.append(('logistic',model_1))

model_2=DecisionTreeClassifier()
estimators.append(('dt',model_2))

model_3=SVC()
estimators.append(('svm',model_3))

# 构建投票器融合
ensemble=VotingClassifier(estimators)
result=model_selection.cross_val_score(ensemble,X,Y,cv=kfold)
print(result.mean())




### 2.Bagging
from sklearn.ensemble import  BaggingClassifier
dt=DecisionTreeClassifier()
num=100
kfold=model_selection.KFold(n_splits=5,random_state=2018)
model=BaggingClassifier(base_estimator=dt,n_estimators=num,random_state=2018)
result=model_selection.cross_val_score(model,X,Y,cv=kfold)
print(result.mean())




### 3.RandomForest
from sklearn.ensemble import RandomForestClassifier
num_tree=100
max_feature=5#决策树构造时,我们每次不会用全部的特征
kfold=model_selection.KFold(n_splits=5,random_state=2018)
model=RandomForestClassifier(n_estimators=num_tree,max_features=max_feature)
result=model_selection.cross_val_score(model,X,Y,cv=kfold)
print(result.mean())




### 4.Adaboost
from sklearn.ensemble import AdaBoostClassifier
num_trees=50
kfold=model_selection.KFold(n_splits=5,random_state=2018)
model=AdaBoostClassifier(n_estimators=num_trees,random_state=2018)
result=model_selection.cross_val_score(model,X,Y,cv=kfold)
print(result.mean())




























