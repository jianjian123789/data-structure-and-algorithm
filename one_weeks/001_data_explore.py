# coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as  sns

# 1 数据探索/数据基本处理
data=pd.read_csv('boston_housing.csv')# 数据保存:data=pd.to_csv('boston.csv')

# 1.1 数据探索
data.head()#打印前五行
data.info()#特征描述
data.isnull().sum()#统计空值个数
data.describe()#统计数字特征信息:样本数目/均值/标准差/四分位数
data.shape
data = data[data.MEDV < 50]# 删除y大于50的样本
max_age=df_train['Age'].max()
min_age=df_train['Age'].min()
# 分位数
age_quarter_1=df_train["Age"].quantile(0.25)
age_quarter_3=df_train["Age"].quantile(0.75)


# 1.2 图探索:\
# 1.2.1 单个特征:数字型特征
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
sns.countplot(data.CHAS, order=[0, 1])
plt.xlabel('Charles River')
plt.ylabel('Number of occurrences')

# 单个特征散点图
plt.scatter(range(data.shape[0]), data["MEDV"].values,color='purple')
plt.title("Distribution of Price")

# 1.2.2 单个特征:类别型特征
#对类别型特征，观察其取值范围及直方图
categorical_features = train.select_dtypes(include = ["object"]).columns
for col in categorical_features:
    print '\n%s属性的不同取值和出现的次数'%col
    print train[col].value_counts()




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




# 2. 特征工程

## 2.1 缺失/异常
#删除一列
train.drop(['Id'], inplace = True, axis = 1)# train=train.drop(['Id',axis=1])

#删除部分离群点
train = train[train.GrLivArea < 4000]

#填充
df_train['Age']=df_train["Age"].fillna(value=df_train["Age"].mean())
df_train.loc[:,'Age']=df_train["Age"].fillna(value=df_train["Age"].mean())
df.loc[:, "BsmtUnfSF"] = df.loc[:, "BsmtUnfSF"].fillna(0)
df.loc[:, "PoolQC"] = df.loc[:, "PoolQC"].fillna("No")
df.loc[:, "SaleCondition"] = df.loc[:, "SaleCondition"].fillna("Normal")

# 对训练集的其他数值型特征进行空缺值填补（中值填补）
# 返回填补后的dataframe，以及每列的中值，用于填补测试集的空缺值
# 数值型特征还要进行数据标准化
from sklearn.preprocessing import StandardScaler


def fillna_numerical_train(df):
    numerical_features = df.select_dtypes(exclude=["object"]).columns

    numerical_features = numerical_features.drop("SalePrice")
    print("Numerical features : " + str(len(numerical_features)))

    df.info()
    df_num = df[numerical_features]
    # df_num.info()

    medians = df_num.median()
    # Handle remaining missing values for numerical features by using median as replacement
    print("NAs for numerical features in df : " + str(df_num.isnull().values.sum()))
    df_num = df_num.fillna(medians)
    print("Remaining NAs for numerical features in df : " + str(df_num.isnull().values.sum()))

    # df_num.info()
    # 分别初始化对特征和目标值的标准化器
    ss_X = StandardScaler()

    # 对训练特征进行标准化处理
    temp = ss_X.fit_transform(df_num)
    df_num = pd.DataFrame(data=temp, columns=numerical_features, index=df_num.index)

    return df_num, medians, ss_X
train_num, medians, ss_X = fillna_numerical_train(train)


# 对测试集的其他数值型特征进行空缺值填补（用训练集中相应列的中值填补）
def fillna_numerical_test(df, medians, ss_X):
    numerical_features = df.select_dtypes(exclude=["object"]).columns
    # numerical_features = numerical_features.drop("SalePrice")  #测试集中没有SalePrice
    print("Numerical features : " + str(len(numerical_features)))

    df_num = df[numerical_features]

    # Handle remaining missing values for numerical features by using median as replacement
    print("NAs for numerical features in df : " + str(df_num.isnull().values.sum()))
    df_num = df_num.fillna(medians)
    print("Remaining NAs for numerical features in df : " + str(df_num.isnull().values.sum()))

    # 对数值特征进行标准化
    temp = ss_X.transform(df_num)
    df_num = pd.DataFrame(data=temp, columns=numerical_features, index=df_num.index)
    return df_num


test_num = fillna_numerical_test(test, medians, ss_X)


## 2.2 特征编码
###2.2.1 数值型特征
#### 2.2.1.1 归一
from sklearn.preprocessing import MinMaxScaler
mm_scaler=MinMaxScaler()
fare_trans=mm_scaler.fit_transform(df_train[['Fare']])
df_train[['Fare']]=fare_trans
# df_train['Fare']=fare_std_trans

#### 2.2.1.2 标准化
from sklearn.preprocessing import StandardScaler
std_scaler=StandardScaler()
fare_std_trans=std_scaler.fit_transform(df_train[['Fare']])

#### 2.2.1.3 正规化
from sklearn.preprocessing import Normalizer
nn=Normalizer()
X_embed[:5]=nn.fit_transform(X_embed[:5])#注意:data中必须都是数值型,否则出错

#### 2.2.1.4 离散化/分箱/分桶
# 等距切分
df_train.loc[:,'fare_cut']=pd.cut(df_train["Fare"],5)
df_train['fare_cut'].unique()# 查看所有可能的取值

# 等频切分
df_train.loc[:,'fare_qcut']=pd.qcut(df_train['Fare'],5)
df_train['fare_qcut'].unique()

# 数值型特征类别化
# Some numerical features are actually really categories
# MSSubClass：建筑类
#MoSold：销售月份
def numberical2cat(df):
    df.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45",
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75",
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120",
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      }, inplace = True)

    return df
train = numberical2cat(train)
test = numberical2cat(test)


#### 2.2.1.5 高次与四则运算
### 还可以通过以下方式创建一些新特征 :
 1. 简化已有特征
 2. 联合已有特征
 3. 现有重要特征（top 10）的多项式

# 取对数变换
log_age=df_train["Age"].apply(lambda x:np.log(x))
df_train.loc[:,'log_age']=log_age# 新增一列数据
# 加法
df_train.loc[:,'family_size']=df_train['SibSp']+df_train['Parch']+1
# 多项式:使用参考 https://blog.csdn.net/tiange_xiao/article/details/79755793
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)#最高次项为二次:0/1/2次都进行计算
poly_fea=poly.fit_transform(df_train[['SibSp','Parch']]
df_train[['SibSp','Parch']].head()

# Create new features
# 1* Simplifications of existing features
# 合并类别
def simplify(df):
    df["SimplOverallQual"] = df.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                    4 : 2, 5 : 2, 6 : 2, # average
                                                    7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                    }, inplace = True)
    df["SimplOverallCond"] = df.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                    4 : 2, 5 : 2, 6 : 2, # average
                                                    7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                    },inplace = True)
    df["SimplPoolQC"] = df.PoolQC.replace({1 : 1, 2 : 1, # average
                                           3 : 2, 4 : 2 # good
                                          },inplace = True)
    df["SimplGarageCond"] = df.GarageCond.replace({1 : 1, # bad
                                                2 : 1, 3 : 1, # average
                                                4 : 2, 5 : 2 # good
                                                        },inplace = True)
    df["SimplGarageQual"] = df.GarageQual.replace({1 : 1, # bad
                                                    2 : 1, 3 : 1, # average
                                                    4 : 2, 5 : 2 # good
                                                    },inplace = True)
    df["SimplFireplaceQu"] = df.FireplaceQu.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          },inplace = True)
    df["SimplFireplaceQu"] = df.FireplaceQu.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          },inplace = True)
    df["SimplFunctional"] = df.Functional.replace({1 : 1, 2 : 1, # bad
                                                         3 : 2, 4 : 2, # major
                                                         5 : 3, 6 : 3, 7 : 3, # minor
                                                         8 : 4 # typical
                                                        },inplace = True)
    df["SimplKitchenQual"] = df.KitchenQual.replace({1 : 1, # bad
                                                           2 : 1, 3 : 1, # average
                                                           4 : 2, 5 : 2 # good
                                                          },inplace = True)
    df["SimplHeatingQC"] = df.HeatingQC.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      },inplace = True)
    df["SimplBsmtFinType1"] = df.BsmtFinType1.replace({1 : 1, # unfinished
                                                             2 : 1, 3 : 1, # rec room
                                                             4 : 2, 5 : 2, 6 : 2 # living quarters
                                                            },inplace = True)
    df["SimplBsmtFinType2"] = df.BsmtFinType2.replace({1 : 1, # unfinished
                                                             2 : 1, 3 : 1, # rec room
                                                             4 : 2, 5 : 2, 6 : 2 # living quarters
                                                            },inplace = True)
    df["SimplBsmtCond"] = df.BsmtCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    },inplace = True)
    df["SimplBsmtQual"] = df.BsmtQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    },inplace = True)
    df["SimplExterCond"] = df.ExterCond.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      },inplace = True)
    df["SimplExterQual"] = df.ExterQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      },inplace = True)
    return df

train = simplify(train)
test = simplify(test)


# 2* Combinations of existing features
def Combine(df):
    # Overall quality of the house
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    # Overall quality of the garage
    df["GarageGrade"] = df["GarageQual"] * df["GarageCond"]
    # Overall quality of the exterior
    df["ExterGrade"] = df["ExterQual"] * df["ExterCond"]
    # Overall kitchen score
    df["KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]
    # Overall fireplace score
    df["FireplaceScore"] = df["Fireplaces"] * df["FireplaceQu"]
    # Overall garage score
    df["GarageScore"] = df["GarageArea"] * df["GarageQual"]
    # Overall pool score
    df["PoolScore"] = df["PoolArea"] * df["PoolQC"]
    # Simplified overall quality of the house
    df["SimplOverallGrade"] = df["SimplOverallQual"] * df["SimplOverallCond"]
    # Simplified overall quality of the exterior
    df["SimplExterGrade"] = df["SimplExterQual"] * df["SimplExterCond"]
    # Simplified overall pool score
    df["SimplPoolScore"] = df["PoolArea"] * df["SimplPoolQC"]
    # Simplified overall garage score
    df["SimplGarageScore"] = df["GarageArea"] * df["SimplGarageQual"]
    # Simplified overall fireplace score
    df["SimplFireplaceScore"] = df["Fireplaces"] * df["SimplFireplaceQu"]
    # Simplified overall kitchen score
    df["SimplKitchenScore"] = df["KitchenAbvGr"] * df["SimplKitchenQual"]
    # Total number of bathrooms
    df["TotalBath"] = df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"]) + \
                      df["FullBath"] + (0.5 * df["HalfBath"])
    # Total SF for house (incl. basement)
    df["AllSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    # Total SF for 1st + 2nd floors
    df["AllFlrsSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
    # Total SF for porch
    df["AllPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + \
                       df["3SsnPorch"] + df["ScreenPorch"]
    # Has masonry veneer or not
    df["HasMasVnr"] = df.MasVnrType.replace({"BrkCmn": 1, "BrkFace": 1, "CBlock": 1,
                                             "Stone": 1, "None": 0})
    # House completed before sale or not
    df["BoughtOffPlan"] = df.SaleCondition.replace({"Abnorml": 0, "Alloca": 0, "AdjLand": 0,
                                                    "Family": 0, "Normal": 0, "Partial": 1})

    return df
# 对训练集和测试集分别进行编码
train = Combine(train)
test = Combine(test)


# Find most important features relative to target
print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
#print(corr.SalePrice)

threshold = corr.SalePrice.iloc[11]  #the first one is SalePrice itself,from 1-11
print threshold
top10_cols = (corr.SalePrice[corr['SalePrice']>threshold]).axes


# Create new features
# 3* Polynomials on the top 10 existing features
def Polynomials_top10(df, top10_cols):
    for i in range(1, 11):
        new_cols_2 = top10_cols[0][i] + '_s' + str(2)
        new_cols_3 = top10_cols[0][i] + '_s' + str(3)
        new_cols_sq = top10_cols[0][i] + '_sq'

        df[new_cols_2] = df[top10_cols[0][i]] ** 2
        df[new_cols_3] = df[top10_cols[0][i]] ** 3
        df[new_cols_sq] = np.sqrt(df[top10_cols[0][i]])

    return df


train = Polynomials_top10(train, top10_cols)
test = Polynomials_top10(test, top10_cols)


### 2.2.2 类别型特征
#### 2.2.2.1 哑编码
embarked_oht=pd.get_dummies(df_train[['Embarked']])
embarked_oht.head()
fare_qcut_oht=pd.get_dummies(df_train[['fare_qcut']])#将我们刚刚分桶区间进行哑编码


def get_dummies_cat(df):
    categorical_features = df.select_dtypes(include=["object"]).columns
    print("Categorical features : " + str(len(categorical_features)))
    df_cat = df[categorical_features]

    # Create dummy features for categorical values via one-hot encoding
    print("NAs for categorical features in df : " + str(df_cat.isnull().values.sum()))
    df_cat = pd.get_dummies(df_cat, dummy_na=True)
    print("Remaining NAs for categorical features in df : " + str(df_cat.isnull().values.sum()))

    return df_cat

# 必须考虑类别型特征的取值范围（训练集和测试的取值范围可能不同）
# train_cat = get_dummies_cat(train)
# test_cat = get_dummies_cat(test)

n_train_samples = train.shape[0]
train_test = pd.concat((train, test), axis=0)
train_test_cat = get_dummies_cat(train_test)

train_cat = train_test_cat.iloc[:n_train_samples, :]
test_cat = train_test_cat.iloc[n_train_samples:, :]


# 类别型特征中含有顺序的含义,可以把类别型数值化
# Encode some categorical features as ordered numbers when there is information in the order
def cat2numberical(df):
    df.replace({"Alley" : {"None":0, "Grvl" : 1, "Pave" : 2},
                "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5,
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                "Street" : {"Grvl" : 1, "Pave" : 2},
                "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}},
                       inplace = True
                     )
    return df

train = cat2numberical(train)
test = cat2numberical(test)

### 2.2.3 日期时间型
# 日期时间如果是类别型,则用pd转换成日期datetime64型,后面处理就很简单了
# car_sales.loc[:,'date']=pd.to_datetime(car_sales['date_t'])

# 取出几月分
# car_sales.loc[:,'month']=car_sales['date'].dt.month#取出datetime中月份
# .dt. tab可以看出能够使用的函数进行调用
# 取出来是几号
# car_sales.loc[:,'dom']=car_sales['date'].dt.day

# 取出来是一年中的第几天
# car_sales.loc[:,'doy']=car_sales['date'].dt.dayofyear

# 生成周末列
# car_sales.loc[:,'is_weekend']=car_sales['dow'].apply(lambda x : 1 if (x==0 or x==6) else )





### 2.2.4 文本型
#### 2.2.4.1 词袋
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
corpus=[
    'This is a very good class',
    'students are very good',
    'This is the third sentence',
    'Is this the last doc'
]
X=vectorizer.fit_transform(corpus)
vectorizer.get_feature_names()#查看词表语料库,这个词表是按照字典的方式进行排序的
X.toarray()

vec=CountVectorizer(ngram_range=(1,3))#构造一个从1到3个组合词的词袋
X_ngram=vec.fit_transform(corpus)
vec.get_feature_names()
X_ngram.toarray()

#### 2.2.4.2 TF-IDF
# TF-IDF是一个带权重的词袋,不仅有词袋的计数器作用,含有词对的重要性大小
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec=TfidfVectorizer()
tfidf_X=tfidf_vec.fit_transform(corpus)
tfidf_vec.get_feature_names()
tfidf_X.toarray()




#### 2.2.5 组合特征
# 借助条件去判断获取组合特征
df_train.loc[:,'alone']=(df_train['SibSp']==0)&(df_train['Parch']==0)
df_train.alone.head()


# Join categorical and numerical features
def joint_num_cat(df_num, df_cat):
    df = pd.concat([df_num, df_cat], axis=1, ignore_index=True)
    print("New number of features : " + str(df.shape[1]))

    return df


FE_train = joint_num_cat(train_num, train_cat)
FE_test = joint_num_cat(test_num, test_cat)

FE_train = pd.concat([FE_train, train['SalePrice']], axis=1)
FE_test = pd.concat([test_id, FE_test], axis=1)

FE_train.to_csv('AmesHouse_FE_train.csv', index=False)
FE_test.to_csv('AmesHouse_FE_test.csv', index=False)




# 3 特征选择

## 3.1 过滤式/filter
from sklearn.feature_selection import  SelectKBest
from sklearn.datasets import  load_iris
iris=load_iris()
X,y=iris.data,iris.target
X_new=SelectKBest(k=2).fit_transform(X,y)# 基于X和y,选出两个特征
X_new.shape
X_new[:5,:]#通过计算,它发现最后两列的相关度是最高的,因此把最后两列的特征保留


## 3.2 包裹型/wrapper
from sklearn.feature_selection import  RFE
from sklearn.ensemble import  RandomForestClassifier
#做递归型特征选择,那么就需要有一个能够判断特征重要度的模型,我们选择随机森林作为判别特征重要度
rf=RandomForestClassifier()#随机森林的分类器
rfe=RFE(estimator=rf,n_features_to_select=2)#基于rf模型做特征选择,保留特征的个数是2个
X_rfe=rfe.fit_transform(X,y)
X_rfe.shape
X_rfe[:5,:]

## 3.3 嵌入式/embeded
from sklearn.feature_selection import SelectFromModel# 通常基于LR+L1或者SVM+L1
from sklearn.svm import LinearSVC
lsvc=LinearSVC(C=0.01,penalty='l1',dual=False).fit(X,y)
model=SelectFromModel(lsvc,prefit=True)
X_embed=model.transform(X)
X_embed.shape
X_embed[:5,:]#这表示:基于SVC和L1正则化,我们选出来的特征是这三列









