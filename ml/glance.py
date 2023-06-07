from sklearn.model_selection import train_test_split
import pandas as pd
import visual


def check_df(df, is_pre=False):
    if is_pre:
        col_lst = df.columns
    else:
        col_lst = df.columns[df.isna().any()].tolist()
    if is_pre:
        ck_df = pd.DataFrame({'col_name': [], 'col_type': [], 'NA_ratio': []})
        for col in col_lst:
            # 计算缺失值在该列中的占比
            na_percentage = df[col].isna().mean() * 100
            # 打印列名、列类型和缺失值占比
            print(f"col_name: {col}, col_type: {df[col].dtype}, NA_ratio: {na_percentage:.2f}%")
            ck_df = ck_df.append({'col_name': col,
                                  'col_type': df[col].dtype,
                                  'NA_ratio': round(na_percentage,5)},
                                 ignore_index=True)
        ck_df = ck_df.sort_values(by='NA_ratio', ascending=False)
        ck_df.to_csv('NA_check.csv', index=False)

    # 检查每一列是否存在缺失值
    na_columns = df.isna().any()
    na_columns = na_columns[na_columns].index.tolist()

    if not is_pre:
        print('checking if col less 0')
        for col in df.columns:
            if (df[col] < 0).any():
                print('col less 0 values', col)

        # 输出存在缺失值的列
        if len(na_columns) > 0:
            print('col_with_NA:', na_columns)
        else:
            print('There is no NA in DataFrame already')


def load_df(csv_dir, seg=0.1, show=False):
    df = pd.read_csv(csv_dir,  sep=',')
    df = df.sample(frac=seg, random_state=517)
    pd.set_option('display.max_columns', None)
    if show:
        print(df.head())
        print(len(df.columns))
        print(df.describe())
    return df


def split_test_train():
    df = pd.read_csv('../data/newborn_train.csv', sep=',')
    test_df, train_df = train_test_split(df, test_size=0.15, random_state=1)

    train_df.to_csv('../data/newborn_train_hyq.csv', index=False)

    test_y = test_df['newborn_weight']
    test_df = test_df.drop('newborn_weight', axis=1)

    test_df.to_csv('../data/newborn_test_X.csv', index=False)
    test_y.to_csv('../data/newborn_test_y.csv', index=False)


if __name__ == '__main__':
    # split_test_train()
    newborn_train = load_df('../data/newborn_train.csv', 1, True)
    # float_lst = ['mother_body_mass_index', 'mother_delivery_weight',
    #              'mother_height', 'mother_weight_gain']
    # for col in float_lst:
    #     print(col)
    #     visual.show_box(newborn_train, col)

    check_df(newborn_train, is_pre=True)
    exit()
    visual.corr_heat(newborn_train)
    for i, c in enumerate(newborn_train.columns):
        visual.show_col_distribution(newborn_train, c)
