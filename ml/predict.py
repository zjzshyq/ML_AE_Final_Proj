from sklearn.metrics import mean_absolute_percentage_error
from feat import mk_feat
import pandas as pd
import joblib
import numpy as np

print('Making features...')
X_test = mk_feat(is_train=False, is_save=False)

print('Loading the model..')
model = joblib.load('../models/gbt_model.pkl')
y_pred = model.predict(X_test)

check_y_dir = '../data/newborn_test_y.csv'
real_y = np.array(pd.read_csv(check_y_dir)).flatten()
mape = mean_absolute_percentage_error(real_y, y_pred)*100
print('mape:%.4f%%' % mape)
