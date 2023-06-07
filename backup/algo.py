# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import  OneHotEncoder
from scipy.sparse import hstack
from imblearn.over_sampling import RandomOverSampler

data = pd.read_csv('../data/data.csv')
ros = RandomOverSampler(sampling_strategy=0.2,random_state=0)

column = data.columns
X = data.drop('Y', axis=1)
Y = data['Y']
print("Y 中 0 的个数：", sum(data['Y'] == 0))
print("Y 中 1 的个数：", sum(data['Y'] == 1))

# 进行随机过采样
X, Y = ros.fit_resample(X, Y)
# 将过采样后的数据集重新合并
df_resampled = pd.concat([pd.DataFrame(X, columns=X.columns), pd.Series(Y, name='Y')], axis=1)
print("Y 中 0 的个数：", sum(df_resampled['Y'] == 0))
print("Y 中 1 的个数：", sum(df_resampled['Y'] == 1))

X_all, y_all = X,Y
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.3, random_state = 42)

# 定义模型
model = xgb.XGBClassifier(alpha=0.15,
                          nthread=4,     #含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
                          learning_rate=0.1,    #含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                          n_estimators=50,       #含义：总共迭代的次数，即决策树的个数
                          max_depth=4,           #含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
                          gamma=0,               #含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
                          subsample=0.85,       #含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                          )

model.fit(X_train.values, y_train.values)
# 预测及 AUC 评测
y_pred_test = model.predict_proba(X_test.values)
xgb_test_auc = roc_auc_score(pd.get_dummies(y_test), y_pred_test)
print('xgboost test auc: %.5f' % xgb_test_auc)

xgboost = model
# xgboost 编码原有特征
#apply()方法可以获得leaf indices(叶节点索引)
X_train_leaves = xgboost.apply(X_train.values)
#X_train_leaves.shape = (259, 150)
X_test_leaves = xgboost.apply(X_test.values)
#X_test.shape = (112, 4)
#X_test_leaves.shape = (112, 150)
#Return the predicted leaf every tree for each sample.


# 训练样本个数
train_rows = X_train_leaves.shape[0]
# 合并编码后的训练数据和测试数据
X_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
X_leaves = X_leaves.astype(np.int32)
(rows, cols) = X_leaves.shape
# X_leaves.shape = (371, 150)


# 对所有特征进行ont-hot编码
xgbenc = OneHotEncoder()
X_trans = xgbenc.fit_transform(X_leaves)

#fit_transform()的作用就是先拟合数据，然后转化它将其转化为标准形式
#(train_rows, cols) = X_train_leaves.shape

#这里得到的X_trans即为得到的one-hot的新特征
# 定义LR模型
lr = LogisticRegression()
# lr对xgboost特征编码后的样本模型训练
lr.fit(X_trans[:train_rows, :], y_train)
y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])
# 预测及AUC评测
#y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
# y_pred_xgblr1.shape  = (112,)
xgb_lr_auc1 = roc_auc_score(pd.get_dummies(y_test), y_pred_xgblr1)
print('基于Xgb特征编码后的LR AUC: %.5f' % xgb_lr_auc1)


# 将数据分为训练集和测试集进行，用新的特征输入LR进行预测
# 定义LR模型
lr = LogisticRegression(n_jobs=-1, max_iter=50, penalty='l2', C=0.1)
# 组合特征
X_train_ext = hstack([X_trans[:train_rows, :], X_train])
X_test_ext = hstack([X_trans[train_rows:, :], X_test])
# lr对组合特征的样本模型训练
lr.fit(X_train_ext, y_train)
# 预测及AUC评测
y_pred_xgblr2 = lr.predict_proba(X_test_ext)
xgb_lr_auc2 = roc_auc_score(pd.get_dummies(y_test), y_pred_xgblr2)
print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc2)

