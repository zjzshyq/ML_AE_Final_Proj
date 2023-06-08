import pandas as pd
from sklearn.model_selection import train_test_split
import re

print('Loading train data...')
train_df = pd.read_csv('../../data/newborn_train.csv', sep=',')\
    .rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

test_df, train_df = train_test_split(train_df, test_size=0.15, random_state=1)
train_df.to_csv('newborn_train.csv', index=False)

test_y = test_df['newborn_weight']
test_X = test_df.drop('newborn_weight', axis=1)
test_X.to_csv('newborn_test.csv', index=False)
test_y.to_csv('newborn_test_y.csv', index=False)
