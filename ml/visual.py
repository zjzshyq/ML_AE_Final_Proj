import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def show_col_distribution(df: pd.DataFrame, col):
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
    print('column name:', col)
    plt.title(f'{col} Distribution')
    plt.show()
    plt.clf()


def is_cesarean(x):
    if x == "Y":
        return 1
    elif x == "N":
        return 0
    else:
        return None


def corr_heat(df: pd.DataFrame):
    df['newborn_gender'] = df['newborn_gender'].apply(lambda x: 1 if x == 'M' else 0)
    df['previous_cesarean'] = df['previous_cesarean'].apply(is_cesarean)
    df['mother_weight_diff'] = df['mother_delivery_weight'] - df['mother_weight_gain']
    df['mother_weight_height_delivery'] = df['mother_delivery_weight']/df['mother_height']
    df['mother_weight_height_diff'] = (df['mother_delivery_weight'] - df['mother_weight_gain'])/df['mother_height']
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()


def show_box(df, col):
    sns.boxplot(data=df, y=col)
    plt.grid(True)
    plt.show()


def validation_visual():
    col_lst = ['LM', 'Lasso', 'Ridge', 'Elastic', 'SVR', 'FM', 'XGB']
    val_lst = [16.18133, 16.18051, 16.18094, 16.16409,
               16.08132, 16.0366, 15.53767]
    plt.barh(col_lst, val_lst)
    plt.xlim(10, 18)
    for i, v in enumerate(val_lst):
        plt.text(v, i, str(v), ha='left', va='center')

    plt.title('Comparison of MAPE in Validation')
    plt.xlabel('MAPE')
    plt.show()


def test_mape_visual():
    col_lst = ['LM','Lasso', 'Ridge', 'Elastic', 'FM', 'XGB']
    val_lst = [15.84201095, 15.84208675, 15.84210447, 15.84163751,
               15.6736801, 15.41188155]
    plt.barh(col_lst, val_lst)

    plt.xlim(13, 18)
    for i, v in enumerate(val_lst):
        plt.text(v, i, str(v), ha='left', va='center')

    plt.title('Comparison of MAPE in Test')
    plt.xlabel('MAPE')
    plt.show()


def test_test_visual():
    col_lst = ['LM', 'Lasso', 'Ridge', 'Elastic', 'FM', 'XGB']
    time_lst = [1.611, 47.3292, 1.4179, 41.2283, 72.441, 3.5392]
    plt.barh(col_lst, time_lst)

    plt.xlim(1,90)
    for i, v in enumerate(time_lst):
        plt.text(v, i, str(v), ha='left', va='center')

    plt.title('Comparison of Time Cost in Test')
    plt.xlabel('Seconds')
    plt.show()


if __name__ == '__main__':
    validation_visual()
    test_mape_visual()
    test_test_visual()
