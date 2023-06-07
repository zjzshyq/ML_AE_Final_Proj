# coding=utf-8
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def parse_df_na(df):
    missing_cols = df.columns[df.isnull().sum() > 0]
    print("\nColumns with missing values:", missing_cols)

    for col in missing_cols:
        # Check data type of column
        na_percentage = df[col].isna().mean()
        dtype = df[col].dtype
        if na_percentage < 0.3:
            # Fill missing values based on data type
            if dtype == 'int64':
                df[col].fillna(df[col].median(), inplace=True)
            elif dtype == 'float64':
                df[col].fillna(df[col].mean(), inplace=True)
        elif na_percentage > 0.95:
            # NA占比大于95%，处理为one-hot形式
            if df[col].nunique() > 2:
                df[col + '_not_na'] = np.where(df[col].notna(), 1, 0)
                df[col + '_not_na'] = df[col + '_not_na'].astype(int)
                print(f"列名: {col}, 列类型: {df[col].dtype}, 缺失值占比: {na_percentage:.2f}%, 处理为one-hot形式")
                df.drop(col, axis=1, inplace=True)
        else:
            # NA占比在30%和95%之间，打印列名和部分非NA的值
            non_na_values = df[col].dropna().unique()[:5]
            print(
                f"列名: {col}, 列类型: {df[col].dtype}, 缺失值占比: {na_percentage:.2f}%, 非NA的值: {non_na_values}")
    return df


def parse_str(df):
    str_cols = df.select_dtypes(include='object')
    print('there are', len(df.columns), 'cols')
    for col in str_cols.columns:
        df[col].replace('', pd.NA, inplace=True)
        unique_values = df[col].nunique()
        if unique_values == 1:
            df[col] = df[col].notna().astype(int) if pd.notna(unique_values) else df[col].isna().astype(int)
        elif unique_values == 2:
            # 统计每个值的频次
            value_counts = df[col].value_counts(normalize=True)
            # 获取频次较多和较少的值
            majority_value = value_counts.idxmax()
            minority_value = value_counts.idxmin()
            # 将频次较多的值设置为0，频次较少的值设置为1
            df[col] = df[col].replace({majority_value: 0, minority_value: 1, pd.NA: 0})
            df[col] = df[col].astype(int)
        elif unique_values < 200:
            # For columns with less than 10 unique values, do one-hot encoding
            one_hot = pd.get_dummies(df[col], prefix=col)
            # Remove one-hot encoded columns with NaN or empty values
            one_hot = one_hot.dropna(axis=1, how='all')
            df = pd.concat([df, one_hot], axis=1)
            df.drop(col, axis=1, inplace=True)
        else:
            try:
                df[col] = pd.to_datetime(df[col])
                max_date = df[col].max()
                min_date = df[col].min()
                df[col] = (df[col] - min_date) / (max_date - min_date)
                df[col] = df[col].astype(float)
                df[col].fillna(df[col].mean(), inplace=True)
            except ValueError:
                # 如果转换失败，则不进行处理
                top_values = df[col].dropna().value_counts().nlargest(10)
                print(f'Top 10 values for {col}:')
                print(top_values)
                print()
                df.drop(col, axis=1, inplace=True)
    return df


def check_df(df, is_pre=True):
    if is_pre:
        col_lst = df.columns
    else:
        col_lst = df.columns[df.isna().any()].tolist()
    for col in col_lst:
        # 计算缺失值在该列中的占比
        na_percentage = df[col].isna().mean() * 100
        # 打印列名、列类型和缺失值占比
        print(f"列名: {col}, 列类型: {df[col].dtype}, 缺失值占比: {na_percentage:.2f}%")
    # 检查每一列是否存在缺失值
    na_columns = df.isna().any()
    na_columns = na_columns[na_columns].index.tolist()

    # 输出存在缺失值的列
    if len(na_columns) > 0:
        print('存在缺失值的列:', na_columns)
    else:
        print('DataFrame中没有缺失值')


def df_split(df, target_col):
    # 获取目标列的不同取值
    target_values = df[target_col].unique()

    # 创建空的训练集、测试集和验证集
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)
    valid_df = pd.DataFrame(columns=df.columns)

    # 遍历目标列的不同取值
    for value in target_values:
        # 获取目标列等于当前取值的索引
        indices = df.index[df[target_col] == value].tolist()

        # 划分当前取值的索引为训练集、测试集和验证集
        train_idx, test_valid_idx = train_test_split(indices, test_size=0.2, random_state=42)
        test_idx, valid_idx = train_test_split(test_valid_idx, test_size=0.6, random_state=42)

        # 将当前取值对应的样本添加到训练集、测试集和验证集
        train_df = train_df.append(df.loc[train_idx])
        test_df = test_df.append(df.loc[test_idx])
        valid_df = valid_df.append(df.loc[valid_idx])
    return train_df, test_df, valid_df


def parse_non_str(df):
    int_cols = df.select_dtypes(include='int64')
    float_cols = df.select_dtypes(include='float64')
    cols = pd.concat([int_cols, float_cols], axis=1)
    for col in cols:
        num_unique = df[col].nunique()
        if num_unique < 10:
            print(col)
            one_hot = pd.get_dummies(df[col], prefix=col)
            # Remove one-hot encoded columns with NaN or empty values
            one_hot = one_hot.dropna(axis=1, how='all')
            df = pd.concat([df, one_hot], axis=1)
            df.drop(col, axis=1, inplace=True)
    return df


def show_col_distribution(df, col):
    num_unique = df[col].nunique()
    if df[col].dtype == 'object':
        # 如果是字符串类型，使用柱状图进行可视化
        if num_unique <= 4:
            plot_type = 'pie'
        else:
            plot_type = 'bar'
        df[col].value_counts().plot(kind=plot_type)
        plt.xlabel(col)
        plt.ylabel('Count')
    else:
        if num_unique <= 20:
            df[col].value_counts().plot(kind='bar')
            plt.xlabel(col)
            plt.ylabel('Count')
        else:
            df[col].plot(kind='hist')
            plt.xlabel(col)
            plt.ylabel('frequency')
    print('col name', col)
    plt.title(f'{col} Distribution')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    rt_dir = '~/Downloads/UW/competition/WEC2022_Data'
    bkg_train = pd.read_csv(os.path.join(rt_dir, 'BKG_train.csv'), sep=';')
    bkg_train = bkg_train.sample(frac=0.3, random_state=517)

    bkg_train.drop('UPGRADE_SALES_DATE', axis=1, inplace=True)
    bkg_train.drop('UPGRADE_TYPE', axis=1, inplace=True)
    bkg_train.loc[bkg_train['STAY_LENGTH_D'] < -1000, 'STAY_LENGTH_D'] = pd.NA
    mean_A = bkg_train['STAY_LENGTH_D'].mean()
    bkg_train['STAY_LENGTH_D'].fillna(mean_A, inplace=True)

    # print(len(bkg_train.columns))
    # for i, col in enumerate(bkg_train.columns):
    #     show_col_distribution(bkg_train, col)
    #     print(i)

    check_df(bkg_train, is_pre=True)
    bkg_train = parse_df_na(bkg_train)
    bkg_train = parse_str(bkg_train)

    bkg_train.rename(columns={'UPGRADED_FLAG': 'Y'}, inplace=True)
    bkg_train.drop('BOOKING_ID', axis=1, inplace=True)
    print(bkg_train['BOOKING_DEPARTURE_TIME_UTC'])
    # print(bkg_train.columns)
    # for col in bkg_train.columns:
    #     show_col_distribution(bkg_train, col)

    # check_df(bkg_train, is_pre=False)
    # bkg_train.to_csv('../data/data.csv', header=True)

    # train_df, test_df, valid_df = df_split(bkg_train, 'Y')
    # train_df.to_csv('./data/train.csv', header=True)
    # test_df.to_csv('./data/test.csv', header=True)
    # valid_df.to_csv('./data/valid.csv', header=True)
