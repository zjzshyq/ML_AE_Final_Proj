import pandas as pd
import lightgbm as lgb
import re
import time
from feat import mk_feat_pure
from glance import check_df
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import xlearn as xl

print('Loading train data...')
df = pd.read_csv('../data/newborn_train.csv', sep=',')\
    .rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

test_df, train_df = train_test_split(df, test_size=0.1, random_state=1)

print('Feature engineering...')
train = mk_feat_pure(train_df)
test = mk_feat_pure(test_df)

check_df(train, is_pre=False)

train_y = train['newborn_weight']
train_X = train.drop('newborn_weight', axis=1)

test_y = train['newborn_weight']
test_X = train.drop('newborn_weight', axis=1)

print('Training...')
start_time = time.time()
model = LinearRegression()
model.fit(train_X, train_y)
pred_y = model.predict(test_X)
print('LM Train cost:', time.time()-start_time)
mape = mean_absolute_percentage_error(test_y, pred_y)*100
print('LM mape:%.8f%%' % mape)
exit()
start_time = time.time()
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
model = lgb.LGBMRegressor(**params)
model.fit(train_X, train_y)
pred_y = model.predict(test_X)
print('GBT Train cost:', time.time()-start_time)
mape = mean_absolute_percentage_error(test_y, pred_y)*100
print('GBT mape:%.8f%%' % mape)

start_time = time.time()
model = Lasso(alpha=0)
model.fit(train_X, train_y)
pred_y = model.predict(test_X)
print('Lasso Train cost:', time.time()-start_time)
mape = mean_absolute_percentage_error(test_y, pred_y)*100
print('Lasso mape:%.8f%%' % mape)

start_time = time.time()
model = Ridge(alpha=0.35)
model.fit(train_X, train_y)
pred_y = model.predict(test_X)
print('Ridge Train cost:', time.time()-start_time)
mape = mean_absolute_percentage_error(test_y, pred_y)*100
print('Ridge mape:%.8f%%' % mape)

start_time = time.time()
model = ElasticNet(alpha=0.35, l1_ratio=0)
model.fit(train_X, train_y)
pred_y = model.predict(test_X)
print('ElasticNet Train cost:', time.time()-start_time)
mape = mean_absolute_percentage_error(test_y, pred_y)*100
print('ElasticNet mape:%.8f%%' % mape)

print('Training SVR...')
start_time = time.time()
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(train_X, train_y)
pred_y = model.predict(test_X)
print('SVR Train cost:', time.time()-start_time)
mape = mean_absolute_percentage_error(test_y, pred_y)*100
print('SVR mape:%.8f%%' % mape)


start_time = time.time()
params = {'task': 'reg', 'metric': 'mape',
          'lr': 0.1, 'epoch': 150, 'k': 40,
          'reg_lambda': 0, 'reg_alpha': 0.2,
          }
model = xl.FMModel(**params)
model.fit(train_X, train_y)
pred_y = model.predict(test_X)
print('FM Train cost:', time.time()-start_time)
mape = mean_absolute_percentage_error(test_y, pred_y)*100
print('FM mape:%.8f%%' % mape)
