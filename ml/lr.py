import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from feat import mk_feat


# 读入数据
df = pd.read_csv('../data/train_data_cleaned.csv')
X = df.drop('newborn_weight', axis=1)
y = df['newborn_weight']


y_preds = []
y_tests = []
param = {}
best = {'mape': 25, 'alpha': 0}
model_name = 'SVR'
param_name = 'alpha'
param_lst = [0]
res_lst = []
kf = KFold(n_splits=4, shuffle=True, random_state=42)
for p in param_lst:
    param[param_name] = p
    for k, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx].to_numpy(), X.iloc[test_idx].to_numpy()
        y_train, y_test = np.array(y.iloc[train_idx]), np.array(y.iloc[test_idx])

        if model_name == 'LR':
            model = LinearRegression()
        elif model_name == 'SVM':
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

        else:
            if model_name == 'L1':
                model = Lasso(**param)
            else:
                model = Ridge(**param)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_preds.append(y_pred)
        y_tests.append(y_test)

        mape = mean_absolute_percentage_error(y_test, y_pred)*100
        print(f"Fold {k + 1} MAPE: {mape} %")

    mean_mape = np.mean([mean_absolute_percentage_error(y_test, y_pred)
                         for y_test, y_pred in zip(y_tests, y_preds)]) * 100

    print(f"Mean MAPE: {mean_mape} %, in param {p}")
    if mean_mape <= best['mape']:
        best['param'] = p
        best['mape'] = mean_mape
    res_lst.append(mean_mape)

# if model_name != 'LR':
#     plt.plot(param_lst, res_lst)
#     plt.xlabel('lambda')
#     plt.ylabel('MAPE')
#     plt.title('Curve Plot')
#     plt.show()

model2 = model = Ridge(alpha=10)
model2.fit(X, y)

X_test2 = mk_feat(is_train=False, is_save=False)
y_pred2 = model2.predict(X_test2)

real_y = pd.read_csv('../data/newborn_test_y.csv')
mape = mean_absolute_percentage_error(real_y, y_pred2)*100
print('mape:%.4f%%' % mape)

