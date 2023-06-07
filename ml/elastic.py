import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_percentage_error


# 读入数据
df = pd.read_csv('../data/train_data_cleaned.csv')
train_data, val_data = train_test_split(df, test_size=0.1, random_state=42)
X_train = train_data.drop('newborn_weight', axis=1)
y_train = train_data['newborn_weight']

flag = True
if flag:
    best_parameters = {'alpha': 0.35, 'l1_ratio': 0}
    model = ElasticNet(**best_parameters)
    model.fit(X_train, y_train)
else:
    # 使用5-fold交叉验证搜索最佳的alpha和l1_ratio参数
    param_grid = {
        'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'l1_ratio': np.arange(0, 1.1, 0.1)
    }
    grid_search = GridSearchCV(ElasticNet(),
                               param_grid,
                               cv=5,
                               scoring='mean_absolute_percentage_error')
    grid_search.fit(X_train, y_train)
    print('Best parameters:', grid_search.best_params_)
    model = grid_search.best_estimator_

X_test = val_data.drop('newborn_weight', axis=1)
y_test = val_data['newborn_weight']
y_pred = model.predict(X_test)

mape = mean_absolute_percentage_error(y_test, y_pred)*100
print('MAPE:', mape)
