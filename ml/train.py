from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
import xlearn as xl
import lightgbm as lgb
import pandas as pd
import joblib
import time
import re


def get_x_y():
    df = pd.read_csv('../data/train_data_cleaned.csv')\
        .rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    x = df.drop('newborn_weight', axis=1)
    y = df['newborn_weight']
    return x, y


def fm(x, y):
    begin = time.time()
    param = {'task': 'reg', 'metric': 'mape',
             'lr': 0.1, 'epoch': 150, 'k': 40,
             'reg_lambda': 0.2, 'reg_alpha': 0.2,
             }
    model = xl.FMModel(**param)
    model.fit(x, y)
    print('Time cost of FM:', time.time() - begin)
    joblib.dump(model, '../models/fm_model.pkl')


def gbt(x, y):
    begin = time.time()
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mape',
        'max_depth': 8,
        'num_leaves': 30,
        'learning_rate': 0.12,
        'n_estimators': 150,
        'reg_alpha': 0.5,
        'reg_lambda': 100,
        'subsample': 0.98,
        'colsample_bytree': 0.97
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(x, y)
    print('Time cost of GBT:', time.time() - begin)
    joblib.dump(model, '../models/gbt_model.pkl')


def elastic(x, y):
    begin = time.time()
    params = {'alpha': 0.2, 'l1_ratio': 1.0}
    model = ElasticNet(**params)
    model.fit(x, y)
    print('Time cost of FM:', time.time() - begin)
    joblib.dump(model, '../models/elastic_model.pkl')


def lasso(x, y):
    begin = time.time()
    params = {'alpha': 0.2}
    model = Lasso(**params)
    model.fit(x, y)
    print('Time cost of LASSO:', time.time() - begin)
    joblib.dump(model, '../models/lasso_model.pkl')


def ridge(x, y):
    begin = time.time()
    params = {'alpha': 0.2}
    model = Ridge(**params)
    model.fit(x, y)
    print('Time cost of Ridge:', time.time() - begin)
    joblib.dump(model, '../models/ridge_model.pkl')


def lm(x, y):
    begin = time.time()
    model = LinearRegression()
    model.fit(x, y)
    print('Time cost of LM:', time.time() - begin)
    joblib.dump(model, '../models/LM_model.pkl')


if __name__ == '__main__':
    train_X, train_y = get_x_y()
    gbt(train_X, train_y)
