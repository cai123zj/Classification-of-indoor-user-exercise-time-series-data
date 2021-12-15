#!/usr/bin/env python
# coding: utf-8

# In[2]:


#最终模型
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR
import matplotlib.pyplot as plt

plt.figure()
train = pd.read_csv('D:/Load/train.csv')
test = pd.read_csv('D:/Load/test.csv')

#分离数据集
X_train_c = train.drop(['ID','CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
#将数据集划分为10个大小相似地互斥子集
nfold = 10
#进行交叉验证
fold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)
result = np.zeros(len(X_test_c))
auc_score = 0
i=0

for train_index, valid_index in fold.split(X_train_c, y_train_c):
    print("\nFold {}".format(i + 1))
    X_train, Y_train = X_train_c[train_index],y_train_c[train_index]
    X_valid, Y_valid = X_train_c[valid_index],y_train_c[valid_index]
    #模型训练
    clf=SVR(kernel='rbf',C=1,gamma='scale')
    clf.fit(X_train,Y_train)
    
    #预测
    x1 = clf.predict(X_valid)#训练集
    auc=roc_auc_score(Y_valid,x1)
    print(auc)
    auc_score+= auc
    i+=1
    
    y1 = clf.predict(X_test_c)#测试集
    test_result+=y1

auc_score=auc_score/fold.n_splits
print("AUC score：",auc_score)
result=result/fold.n_splits
plt.plot(result,color='navy',label='svm')
plt.legend()
plt.show()


# In[ ]:


import pandas
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#数据加载
dataframe = pandas.read_csv("D:/Load/train.csv")
dataframe_test = pandas.read_csv("D:/Load/test.csv")

X = dataframe.iloc[:,1:241].astype(float)
Y = dataframe.iloc[:,241]
X_test = dataframe_test.iloc[:,1:241].astype(float)

#对训练数据重构
data_new = list()
for i in range(len(X)):
    data_new.append(X.loc[i])
data_new = np.array(data_new).reshape(-1,1)
time_id = np.tile(np.array([i for i in range(0,240)]) , len(X)).reshape(-1,1)
id_index = np.array([i for i in range(0,210)]).repeat(240).reshape(-1,1)  
  
data_format = pandas.DataFrame(np.concatenate([id_index,time_id,data_new],axis=1))
data_format.columns = ['id','time','time_series']


#对训练数据进行提取和筛选
from tsfresh import extract_features
extracted_features = extract_features(data_format, column_id="id", column_sort="time")

#特征筛选
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
impute(extracted_features)
features_filtered = select_features(extracted_features, Y)

from tsfresh import feature_extraction
kind_to_fc_parameters = feature_extraction.settings.from_columns(features_filtered)


#测试数据重构和特征筛选
data_new = list()
for i in range(len(X_test)):
    data_new.append(X_test.loc[i])
data_new = np.array(data_new).reshape(-1,1)
time_id = np.tile(np.array([i for i in range(0,240)]) , len(X_test)).reshape(-1,1)
id_index = np.array([i for i in range(0,104)]).repeat(240).reshape(-1,1) 
data_format_test = pandas.DataFrame(np.concatenate([id_index,time_id,data_new],axis=1))
data_format_test.columns = ['id','time','time_series']

features_filtered_test = extract_features(data_format_test, column_id="id", column_sort="time",
                                          kind_to_fc_parameters =kind_to_fc_parameters)

features_filtered_test = features_filtered_test[features_filtered.columns]

#查看已筛选出的特征
features_filtered_test.info()

#修改特征
new_col = ['fea%s'%i for i in range(67)]
print(new_col)
features_filtered_test.columns = new_col
features_filtered.columns = new_col

#对数据集采用10交叉验证法
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

num_folds = 10
folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=2020)
test_result = np.zeros(len(features_filtered_test))
auc_score = 0

#定义lgb参数
params = {'num_leaves': int(16), 
          'objective': 'regression', 
          'max_depth': int(4),
           'min_data_in_leaf': int(5),
           'min_sum_hessian_in_leaf': int(0),
          'learning_rate': 0.18,
          'boosting': 'gbdt',
          'feature_fraction': 0.8,  
          'bagging_freq': int(2),
          'bagging_fraction': 1,
          'bagging_seed': 8,
          'lambda_l1':  0.01,             
          'lambda_l2': 0.01,     
          'metric': 'auc',  ##评价函数选择
          "random_state": 2020, #随机数种子，可以防止每次运行的结果不一致
          }

#模型训练和预测
for fold_, (trn_idx, val_idx) in enumerate(folds.split(features_filtered, Y)):
    print("Fold: ", fold_ + 1)
    X_train, y_train = features_filtered.iloc[trn_idx], Y.iloc[trn_idx]
    X_valid, y_valid = features_filtered.iloc[val_idx], Y.iloc[val_idx]
    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_valid, y_valid, reference=trn_data)
    #模型训练
    clf = lgb.train(params,
                    trn_data,
                    10000,
                    valid_sets=val_data,
                    verbose_eval=50,
                    early_stopping_rounds=50)  
    #模型预测
    y_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)#训练集
    auc = roc_auc_score(y_valid, y_pred)
    print(auc)
    auc_score += auc
    
    preds = clf.predict(features_filtered_test, num_iteration=clf.best_iteration)#测试集
    test_result += preds    
auc_score = auc_score / folds.n_splits
print("AUC score: ", auc_score)
test_result = test_result / folds.n_splits
plt.plot(test_result,color='navy',label='lgb')


# In[ ]:


#模型融合svm+lgb
ans = [0 for i in range(104)]
for i in range(104):
    if result[i]>0.5 and test_result[i]>0.5:
        ans[i] = int(1)
    if result[i]<0.5 and test_result[i]<0.5:
        ans[i] = int(0)
    if result[i]>0.5 and test_result[i]<0.5:
        d1 = result[i]-0.5
        d2 = 0.5 - test_result[i]
        if d1>d2+0.1:
            ans[i] = int(1)
        else:
            ans[i] = int(0)
    if result1[i]<0.5 and test_result[i]>0.5:
        d1 = 0.5 - result[i]
        d2 = test_result[i] - 0.5
        if d2+0.1>d1:
            ans[i] = int(1)
        else:
            ans[i] = int(0)

#文件输出
Id = range(210,314)
df = pd.DataFrame({'ID':Id,'CLASS':ans})
df.to_csv("D:/Load/result.csv",index=False)
plt.plot(ans,color='navy',label='svm+lgb')

