import numpy as np
import pandas as pd
import lightgbm as lgb
import re
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from feat import mk_feat, outliers4target, mk_feat_pure
import time

# train_dir = '../data/train_data_cleaned.csv'
# data = pd.read_csv(train_dir)\
#     .rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

df = pd.read_csv('../data/newborn_train.csv', sep=',').rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
test_df, train_df = train_test_split(df, test_size=0.15, random_state=1)
data = mk_feat_pure(train_df)
test_data = mk_feat_pure(test_df)

# test_data, data = train_test_split(data, test_size=0.1, random_state=1)
y_test2 = test_data['newborn_weight']
X_test2 = test_data.drop('newborn_weight', axis=1)

# data = outliers4target(data, 'newborn_weight')
y = data['newborn_weight']
X = data.drop('newborn_weight', axis=1)

# 定义模型
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mape',
    'max_depth': 8,
    'num_leaves': 30,
    'learning_rate': 0.12,
    'n_estimators': 130,
    'reg_alpha': 0.2,
    'reg_lambda': 0.5,
    'subsample': 0.98,
    'colsample_bytree': 0.97
}

kf = KFold(n_splits=4, shuffle=True, random_state=42)
y_preds = []
y_tests = []
params_lst = [1]
model = None
mean_mape = 1
for param in params_lst:
    if len(params_lst) > 1:
        print('\nparam is', param)
    params['reg_lambda'] = param
    model = lgb.LGBMRegressor(**params)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 模型训练
        model.fit(X_train, y_train)

        # 模型预测
        y_pred = model.predict(X_test)
        y_preds.append(y_pred)
        y_tests.append(y_test)

        # MAPE评估
        mape = mean_absolute_percentage_error(y_test, y_pred)*100
        print(f"Fold {fold + 1} MAPE: {mape} %")

    # 所有Fold的MAPE平均值
    mean_mape = np.mean([mean_absolute_percentage_error(y_test, y_pred)
                         for y_test, y_pred in zip(y_tests, y_preds)]) * 100
print(f"Mean MAPE: {mean_mape} %")

model2 = lgb.LGBMRegressor(**params)
model2.fit(X, y)

# X_test2 = mk_feat(is_train=False, is_save=False)
# y_pred_test = model2.predict(X_test2)
# y_test = pd.read_csv('../data/newborn_test_y.csv')

y_pred2 = model2.predict(X_test2)
mape = mean_absolute_percentage_error(y_test2, y_pred2)*100
print('final test mape:%.4f%%' % mape)

"""

"""