#!/usr/bin/env python
# coding: utf-8

# In[1]:


#其他模型

#svc
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn import svm
import matplotlib.pyplot as plt


train = pd.read_csv('D:/Load/train.csv')
test = pd.read_csv('D:/Load/test.csv')

#分离数据集
X_train_c = train.drop(['ID','CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
#将数据集划分为5个大小相似地互斥子集
nfold = 10
#进行交叉验证
fold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)
test_result = np.zeros(len(X_test_c))
auc_score = 0
i=0

for train_index, valid_index in fold.split(X_train_c, y_train_c):
    print("\nFold {}".format(i + 1))
    X_train, Y_train = X_train_c[train_index],y_train_c[train_index]
    X_valid, Y_valid = X_train_c[valid_index],y_train_c[valid_index]
    #模型训练
    clf=svm.SVC(C=1, kernel='poly', degree=2, gamma='scale', coef0=0.0, shrinking=True, probability=True, 
              tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', 
              break_ties=False, random_state=None)
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
test_result=test_result/fold.n_splits
Y_test=np.round(test_result)
plt.plot(test_result,color='navy',label='svc')
plt.legend()
plt.show()


# In[11]:


#Adaboost+svc
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


train = pd.read_csv('D:/Load/train.csv')
test = pd.read_csv('D:/Load/test.csv')

#分离数据集
X_train_c = train.drop(['ID','CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
#将数据集划分为5个大小相似地互斥子集
nfold = 5
#进行交叉验证
fold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)
test_result = np.zeros(len(X_test_c))
auc_score = 0
i=0

for train_index, valid_index in fold.split(X_train_c, y_train_c):
    print("\nFold {}".format(i + 1))
    X_train, Y_train = X_train_c[train_index],y_train_c[train_index]
    X_valid, Y_valid = X_train_c[valid_index],y_train_c[valid_index]
    #模型训练
    clf=AdaBoostClassifier(base_estimator=svm.SVC(C=2, kernel='poly', degree=2, gamma='scale', coef0=0.0, shrinking=True, probability=True, 
              tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', 
              break_ties=False, random_state=None), 
                         n_estimators=1000, learning_rate=0.7, algorithm='SAMME.R', random_state=None)
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
test_result=test_result/fold.n_splits
Y_test=np.round(test_result)
plt.plot(test_result,color='navy',label='svc+Adaboost')
plt.legend()
plt.show()


# In[3]:


#决策树
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
train = pd.read_csv('D:/Load/train.csv')
test = pd.read_csv('D:/Load/test.csv')

#分离数据集
X_train_c = train.drop(['ID','CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
#将数据集划分为5个大小相似地互斥子集
nfold = 5
#进行交叉验证
fold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)
test_result = np.zeros(len(X_test_c))
auc_score = 0
i=0

for train_index, valid_index in fold.split(X_train_c, y_train_c):
    print("\nFold {}".format(i + 1))
    X_train, Y_train = X_train_c[train_index],y_train_c[train_index]
    X_valid, Y_valid = X_train_c[valid_index],y_train_c[valid_index]
    #模型训练
    clf= clf =DecisionTreeClassifier(criterion='gini', min_samples_leaf=1)
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
test_result=test_result/fold.n_splits
Y_test=np.round(test_result)
plt.plot(test_result,color='navy',label='Tree')
plt.legend()
plt.show()


# In[4]:


#随机森林
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


train = pd.read_csv('D:/Load/train.csv')
test = pd.read_csv('D:/Load/test.csv')

#分离数据集
X_train_c = train.drop(['ID','CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
#将数据集划分为5个大小相似地互斥子集
nfold =5
#进行交叉验证
fold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)
test_result = np.zeros(len(X_test_c))
auc_score = 0
i=0

for train_index, valid_index in fold.split(X_train_c, y_train_c):
    print("\nFold {}".format(i + 1))
    X_train, Y_train = X_train_c[train_index],y_train_c[train_index]
    X_valid, Y_valid = X_train_c[valid_index],y_train_c[valid_index]
    #模型训练
    clf= RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                             min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                             min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, 
                             random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
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
test_result=test_result/fold.n_splits
Y_test=np.round(test_result)
plt.plot(test_result,color='navy',label='RF')
plt.legend()
plt.show()


# In[5]:


#逻辑回归
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

train = pd.read_csv('D:/Load/train.csv')
test = pd.read_csv('D:/Load/test.csv')

#分离数据集
X_train_c = train.drop(['ID','CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
#将数据集划分为5个大小相似地互斥子集
nfold = 5
#进行交叉验证
fold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)
test_result = np.zeros(len(X_test_c))
auc_score = 0
i=0

for train_index, valid_index in fold.split(X_train_c, y_train_c):
    print("\nFold {}".format(i + 1))
    X_train, Y_train = X_train_c[train_index],y_train_c[train_index]
    X_valid, Y_valid = X_train_c[valid_index],y_train_c[valid_index]
    #模型训练
    clf =LogisticRegression(C=100.0,random_state = 1)
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
test_result=test_result/fold.n_splits
Y_test=np.round(test_result)
plt.plot(Y_test,color='navy',label='lr')
plt.legend()
plt.show()


# In[6]:


#线性回归
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn import linear_model 
import matplotlib.pyplot as plt
train = pd.read_csv('D:/Load/train.csv')
test = pd.read_csv('D:/Load/test.csv')

#分离数据集
X_train_c = train.drop(['ID','CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
#将数据集划分为5个大小相似地互斥子集
nfold =5
#进行交叉验证
fold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)
test_result = np.zeros(len(X_test_c))
auc_score = 0
i=0

for train_index, valid_index in fold.split(X_train_c, y_train_c):
    print("\nFold {}".format(i + 1))
    X_train, Y_train = X_train_c[train_index],y_train_c[train_index]
    X_valid, Y_valid = X_train_c[valid_index],y_train_c[valid_index]
    #模型训练
    clf= clf =linear_model.LinearRegression()
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
test_result=test_result/fold.n_splits
Y_test=np.round(test_result)
plt.plot(Y_test,color='navy',label='line')
plt.legend()
plt.show()


# In[10]:


#Adaboost
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


train = pd.read_csv('D:/Load/train.csv')
test = pd.read_csv('D:/Load/test.csv')

#分离数据集
X_train_c = train.drop(['ID','CLASS'], axis=1).values
y_train_c = train['CLASS'].values
X_test_c = test.drop(['ID'], axis=1).values
#将数据集划分为5个大小相似地互斥子集
nfold = 5
#进行交叉验证
fold = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2020)
test_result = np.zeros(len(X_test_c))
auc_score = 0
i=0

for train_index, valid_index in fold.split(X_train_c, y_train_c):
    print("\nFold {}".format(i + 1))
    X_train, Y_train = X_train_c[train_index],y_train_c[train_index]
    X_valid, Y_valid = X_train_c[valid_index],y_train_c[valid_index]
    #模型训练
    clf=AdaBoostClassifier( n_estimators=1000, learning_rate=0.7, algorithm='SAMME.R', random_state=None)
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
test_result=test_result/fold.n_splits
Y_test=np.round(test_result)
plt.plot(Y_test,color='navy',label='Adaboost')
plt.legend()
plt.show()


# In[ ]:




