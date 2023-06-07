import pandas as pd


def parse_non_str(df):
    int_cols = df.select_dtypes(include='int64')
    float_cols = df.select_dtypes(include='float64')
    cols = pd.concat([int_cols, float_cols], axis=1)
    for col in cols:
        num_unique = df[col].nunique()
        if num_unique < 10:
            one_hot = pd.get_dummies(df[col], prefix=col)
            # Remove one-hot encoded columns with NaN or empty values
            one_hot = one_hot.dropna(axis=1, how='all')
            df = pd.concat([df, one_hot], axis=1)
            df.drop(col, axis=1, inplace=True)
    return df


# 创建示例 DataFrame
data = {
    'col1': ['apple', 'banana', 'orange', 'apple', 'banana'],
    'col2': ['2021-01-01', '2021-02-01', '2021-03-01', '2021-01-01', '2021-02-01'],
    'col3': ['red', 'blue', None, 'red', 'blue'],
    'col4': [1, None, 1, 1, 1],
    'col5': [1.1,2.2,3.3,4.4,5.5]
}
df = pd.DataFrame(data)
for col in df.columns:
    print(df[col].dtype)
df = parse_non_str(df)

print(df)