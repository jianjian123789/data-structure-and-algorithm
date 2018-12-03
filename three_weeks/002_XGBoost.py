# coding:utf-8
#.xgboost单独使用：
1.读取DMatrix数据
2.设置参数
param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'multi:softprob' }
num_round = 2
3.模型训练train：bst = xgb.train(param, xgtrain, num_round)
4.预测：
train_preds = bst.predict(dtrain)
train_predictions = [round(value) for value in train_preds]
y_train = dtrain.get_label()
train_accuracy = accuracy_score(y_train, train_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))









##.xgboost并用：
第一轮：确定n_estimators:可以封装成一个函数
a.读取数据：xgtrain=xgb.DMatrix(x_train,label=y_train)
数据读取过程：train--》x_train,y_train--》x_train,x_val,y_train,y_val(train_test_split(x_train,y_train,train_size=0.8,random_state=0)
b.设置参数实例：
B1:模型参数构造：返回模型
xgb1=XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=6,min_child_weight=1,gamma=0,subsample=0.5,colsample_bytree=0.8,colsample_bylevel=0.7,objective='multi:softprob',num_class=3,seed=3)
B2:cv参数构造：确定最佳n_estimators；param是上面构造的模型参数【返回的是DateFrame的结果】
cvresult=xgb.cv(param,xgtrain,num_boost_round=n_estimators,folds=5,metrics='mlogloss',early_stopping_rounds=early_stopping_rounds) （n_estimators = 1000，early_stopping_rounds = 10）
c.模型训练：xgb1.fit(x_train,y_train,eval_metric='mlogloss')
第二轮：确定max_depth和min_child_weight【或者分开单独进行确定】【粗调--》微调--》确定此时n_estimators】
b.设置参数实例：
B1.设置参数实例过程同第一轮，只是将n_estimators=132最佳值
xgb2_1=XGBClassifier(learning_rate=0.1,n_estimators=132,max_depth=6,min_child_weight=1,gamma=0,subsample=0.5,colsample_bytree=0.8,colsample_bylevel=0.7,objective='multi:softprob',num_class=3,seed=3)
B2:GridSearchCV参数构造：确定最佳max_depth和min_child_weight的组合值【Grid SearchCV评价指标scoring越高越好，因此要把损失变成负】【区别cv返回值，它返回的仍是模型】
C1:GridSearchCV中对于分类问题的参数cv设置用‘StratifiedKFold’分层采样设置，类样本不均衡，交叉验证是采用StratifiedKFold，在每折采样时各类样本按比例采样
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
gsearch2_1 = GridSearchCV(xgb2_1, param_grid = param_test2_1, scoring='neg_log_loss',n_jobs=-1, cv=kfold)
C2:GridSearchCV中对于回归问题参数cv用‘KFold’设置
kfold=KFold(n_splits=5,shuffle=False,random_state=3)
gsearch2_1 = GridSearchCV(xgb2_1, param_grid = param_test2_1
param_test2_1 ={max_depth':range(4,10,2),'min_child_weight':range(1,6,2)}
gsearch2_1 = GridSearchCV(xgb2_1, param_grid = param_test2_1, scoring='neg_log_loss',n_jobs=-1, cv=3)
c.模型训练：gsearch2_1.fit(x_train , y_train)
结果：gsearch2_1.grid_scores_, gsearch2_1.best_params_,     gsearch2_1.best_score_
第三轮：确定gamma参数【可忽略】
b.设置参数实例：
B1.模型参数设置过程同前
xgb3_1=XGBClassifier(learning_rate=0.1,n_estimators=132,max_depth=6,min_child_weight=1,gamma=0,subsample=0.5,colsample_bytree=0.8,colsample_bylevel=0.7,objective='multi:softprob',num_class=3,seed=3)
B2.GridSearchCV参数构造：确定最佳gamma值
param_test3 = {'gamma':[i/10.0 for i in range(0,5)]}
gsearch3 = GridSearchCV(estimator = xgb3_1,param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=3)
c.模型训练：gsearch3.fit(train[predictors],train[target])
第四轮：确定行列采样参数
b.设置参数实例：
B1.模型参数
B2.Grid Search CV参数构造
param_test5 = {'subsample':[i/100.0 for i in range(75,90,5)],'colsample_bytree':[i/100.0 for i in range(75,90,5)]}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,objective='binary:logistic',nthread=4,scale_pos_weight=1,seed=27),param_grid=param_test5,scoring='roc_auc',n_jobs=4,iid=False, cv=5)
c.模型训练：gsearch5.fit(train[predictors],train[target])
第五轮：确定两个正则参数
b.设置参数
param_test6 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],'reg_lambda':[0.1,1,2]}
gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177,max_depth=4,min_child_weight=6,gamma=0.1,subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',nthread=4,scale_pos_weight=1,seed=27),param_grid = param_test6, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
c.模型训练：gsearch6.fit(train[predictors],train[target])
第六轮：根据以上结果用cv再确定最佳n_estimators
b.设置参数实例：
B1：模型参数构造：用上面训练好的参数
xgb6 = XGBClassifier(learning_rate =0.02,n_estimators=2000,  #数值大没关系，cv会自动返回合适的n_estimatorsmax_depth=6,min_child_weight=7,gamma=0,subsample = 0.5,colsample_bytree=0.8,colsample_bylevel=0.7,reg_alpha = 0,reg_lambda = 3,objective= 'multi:softprob',seed=3)
B2：cv参数构造：确定最佳n_estimators
cvresult=xgb.cv(param,xgtrain,num_boost_round=n_estimators,folds=5,metrics='mlogloss',early_stopping_rounds=early_stopping_rounds) （n_estimators = 1000，early_stopping_rounds = 10）
c.模型训练：xgb6.fit(x_train,y_train,eval_metric='mlogloss')
d.训练结果测试：
train_predprob = xgb6.predict_proba(x_train)
logloss = log_loss(y_train, train_predprob)
print 'logloss of train is:', logloss
第七轮：保存第六轮结果和模型，在测试集上测试
a1.保存结果
cvresult.to_csv('6_nestimators.csv', index_label = 'n_estimators')
a2.保存模型
import cPickle
cPickle.dump(xgb6, open("xgb_model.pkl", 'wb'))
b1.加载模型
import cPickle
xgb = cPickle.load(open("xgb_model.pkl", 'rb'))
b2.测试集测试
y_test_pred = xgb.predict_proba(X_test)
out_df1 = pd.DataFrame(y_test_pred)
out_df1.columns = ["high", "medium", "low"]
out_df = pd.concat([test_id,out_df1], axis = 1)
out_df.to_csv("xgb_Rent.csv", index=False)






