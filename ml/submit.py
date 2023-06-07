import numpy as np
import lightgbm as lgb
import re
from feat import mk_feat
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error


data = mk_feat(is_train=True, is_save=False)
data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
X = data.drop('newborn_weight', axis=1)
y = data['newborn_weight']

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mape',
    'max_depth': 8,
    'num_leaves': 30,
    'learning_rate': 0.12,
    'n_estimators': 150,
    'reg_alpha': 0.35,
    'reg_lambda': 1,
    'subsample': 0.98,
    'colsample_bytree': 0.97
}
y_preds = []
y_tests = []
model = lgb.LGBMRegressor(**params)
kf = KFold(n_splits=4, shuffle=True, random_state=42)
for k, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_preds.append(y_pred)
    y_tests.append(y_test)

    mape = mean_absolute_percentage_error(y_test, y_pred)*100
    print(f"Fold {k + 1} MAPE: {mape} %")

mean_mape = np.mean([mean_absolute_percentage_error(y_test, y_pred)
                     for y_test, y_pred in zip(y_tests, y_preds)]) * 100
print(f"Final MAPE: {mean_mape} %")
