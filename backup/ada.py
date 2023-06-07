from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score,f1_score,precision_score
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np

# 加载数据集，假设X为特征矩阵，y为目标变量
data = pd.read_csv('../data/data.csv')
X = data.drop('Y', axis=1)
y = data['Y']
print("Y 中 0 的个数：", sum(data['Y'] == 0))
print("Y 中 1 的个数：", sum(data['Y'] == 1))

# 进行随机过采样
ros = RandomOverSampler(sampling_strategy=0.2,random_state=0)
X, y = ros.fit_resample(X, y)
# 将过采样后的数据集重新合并
df_resampled = pd.concat([pd.DataFrame(X, columns=X.columns), pd.Series(y, name='Y')], axis=1)
print("Y 中 0 的个数：", sum(df_resampled['Y'] == 0))
print("Y 中 1 的个数：", sum(df_resampled['Y'] == 1))

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建弱学习器（基模型），这里使用决策树作为基模型
base_model = DecisionTreeClassifier(max_depth=10, min_samples_split=6, min_samples_leaf=3)

# 创建AdaBoost分类器
adaboost = AdaBoostClassifier(base_model, n_estimators=80, learning_rate=0.1, random_state=42)

# # 训练模型
# adaboost.fit(X_train, y_train)
#
# # 在测试集上进行预测
# y_pred = adaboost.predict(X_test)


# 初始化 k-fold 交叉验证
kf = KFold(n_splits=3, shuffle=True, random_state=42)  # 假设 k=5
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
    adaboost.fit(X_train, y_train)

    # 在验证集上进行预测
    y_pred_val = adaboost.predict(X_val)
    y_pred_proba = adaboost.predict_proba(X_val)[:, 1]
    y_pred_val = (y_pred_proba > 0.35).astype(int)

    # 将预测结果添加到列表中
    y_preds.append(y_pred_val)
    y_tests.append(y_val)

# 将预测结果合并为一个一维数组
y_pred = np.concatenate(y_preds)
y_test = np.concatenate(y_preds)

# 计算准确率、召回率和AUC
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# 打印结果
print("Accuracy:", accuracy)
print("Recall:", recall)
print("AUC:", auc)
print("f1:", f1)
print("precision:", precision)
