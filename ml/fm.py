from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from glance import load_df, check_df
import xlearn as xl
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('../data/train_data_cleaned.csv')
X = df.drop('newborn_weight', axis=1)
y = df['newborn_weight']

best = {'mape': 20, 'param': 0}
param = {'task': 'reg', 'metric': 'mape',
         'lr': 0.1, 'epoch': 150, 'k': 40,
         'reg_lambda': 0, 'reg_alpha': 0.2,
         }

y_preds = []
y_tests = []
model = None
param_name = 'reg_lambda'
param_lst = [0]
kf = KFold(n_splits=4, shuffle=True, random_state=42)
for p in param_lst:
    param[param_name] = p
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx].to_numpy(), X.iloc[test_idx].to_numpy()
        y_train, y_test = np.array(y.iloc[train_idx]), np.array(y.iloc[test_idx])

        model = xl.FMModel(**param)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_preds.append(y_pred)
        y_tests.append(y_test)

        mape = mean_absolute_percentage_error(y_test, y_pred)*100
        print(f"Fold {k + 1} MAPE: {mape} %")

        if mape <= best['mape']:
            best['param'] = p
            best['mape'] = mape

    mean_mape = np.mean([mean_absolute_percentage_error(y_test, y_pred)
                         for y_test, y_pred in zip(y_tests, y_preds)]) * 100
    print(f"Mean MAPE: {mean_mape} %, in param {p}")
print(best['mape'], best['param'])

"""
Fold 4 MAPE: 12.870767279528605 %
Mean MAPE: 13.318774666248167 %
12.487684311945786 0
"""