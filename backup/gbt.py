import numpy as np
import pandas as pd
import lightgbm as lgb
import re
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, precision_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
import xgboost as xgb

data = pd.read_csv('../data/data.csv')\
    .rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
X = data.drop('Y', axis=1)
y = data['Y']
print("Y 中 0 的个数：", sum(data['Y'] == 0))
print("Y 中 1 的个数：", sum(data['Y'] == 1))

# 进行随机过采样
ros = RandomOverSampler(sampling_strategy=0.2, random_state=0)
X, y = ros.fit_resample(X, y)
# 将过采样后的数据集重新合并
df_resampled = pd.concat([pd.DataFrame(X, columns=X.columns), pd.Series(y, name='Y')], axis=1)
print("Y 中 0 的个数：", sum(df_resampled['Y'] == 0))
print("Y 中 1 的个数：", sum(df_resampled['Y'] == 1))

# 初始化逻辑回归模型
# 定义模型
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 7,
    'num_leaves': 20,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'reg_alpha': 0.5,
    'reg_lambda': 0.5,
    'subsample': 0.88,
    'feature_fraction': 0.9
}

model = lgb.LGBMClassifier(**params)

# model = xgb.XGBClassifier(
#                           reg_alpha=0.15, reg_lambda=0.1,
#                           nthread=4,     #含义：nthread=-1时，使用全部CPU进行并行运算（默认）, nthread=1时，使用1个CPU进行运算。
#                           learning_rate=0.1,    #含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
#                           n_estimators=60,       #含义：总共迭代的次数，即决策树的个数
#                           max_depth=5,           #含义：树的深度，默认值为6，典型值3-10。调参：值越大，越容易过拟合；值越小，越容易欠拟合
#                           gamma=0,               #含义：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
#                           subsample=0.8,       #含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
#                           )

kf = KFold(n_splits=3, shuffle=True, random_state=42)
y_preds = []
y_tests = []
print('进行 k-fold 交叉验证')
i = 0
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    i += 1
    print('k-fold', i)
    # 在训练集上训练模型
    model.fit(X_train, y_train)

    # 在验证集上进行预测
    # y_pred_val = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred_val = (y_pred_proba > 0.35).astype(int)

    # 将预测结果添加到列表中
    y_preds.append(y_pred_val)
    y_tests.append(y_val)

# 将预测结果合并为一个一维数组
y_pred = np.concatenate(y_preds)
y_test = np.concatenate(y_tests)

# single train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model.fit(X_train, y_train)
# y_proba = model.predict_proba(X_test)[:, 1]
# y_pred_binary = (y_proba > 0.5).astype(int)
# print(y_proba)
# print((y_proba > 0.5).astype(int))
# print((y_proba > 0.3).astype(int))
# print(sum(y_pred_binary))

# 将预测值转换为类别标签
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# 打印评估结果
print("Accuracy: {:.4f}".format(accuracy))
print("Recall: {:.4f}".format(recall))
print("AUC: {:.4f}".format(auc))
print("precision: {:.4f}".format(precision))
print("f1: {:.4f}".format(f1))
