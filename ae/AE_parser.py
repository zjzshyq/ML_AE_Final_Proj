import pandas as pd
import matplotlib.pyplot as plt


# 对曝光数做时间序列分析
# # 对AI和非AI分组
def get_time_series_df(df, col='bookmarks'):
    date_like = df[['date', col, 'views', 'rank']]
    date_like['like_rate'] = date_like[col]/date_like['views']
    # date_like['like_rate'] = date_like['views']
    date_like.drop(col,inplace=True, axis=1)
    date_like.drop('views',inplace=True, axis=1)

    date_like = date_like.groupby('date').apply(lambda x: x.nsmallest(50, 'rank'))
    result = date_like.reset_index(drop=True).groupby('date')['like_rate'].mean().reset_index()
    result.columns = ['date', 'mean_like_rate']
    result['date'] = pd.to_datetime(result['date'], format='%Y%m%d').dt.date.astype(str)
    # result.set_index('date', inplace=True)
    return result


def get_tobit(df: pd.DataFrame):
    # y，收藏率和喜欢率分别做为目标（tobit)
    # X，曝光数，是否是AI，评论数，当日排名，创建日期和排名日期时间差，是否带有原神/崩坏等二次元游戏标签
    df['views'] = df['views'].astype(int)
    df['comments'] = df['comments'].astype(int)
    df['like_rate'] = df['likes'].astype(float)/df['views']
    df['mark_rate'] = df['bookmarks'].astype(float) / df['views']

    df['date_format'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.tz_localize('UTC')
    df['create_time_format'] = pd.to_datetime(df['create_time'])
    df['date_diff_day'] = (df['date_format'] - df['create_time_format']).dt.days+1

    # df = df.sort_values(by='date_format', ascending=False)
    # df['top_cnt'] = df.duplicated(subset='pid', keep=False)\
    #     .astype(int)\
    #     .groupby(df['pid'])\
    #     .transform('sum')
    # df = df.drop_duplicates(subset='pid', keep='first')

    df['is_ai'] = df['aiType'].apply(lambda x: 1 if x == 2 else 0)
    df['is_comic'] = df['tags'].str.contains('漫画').astype(int)
    df['is_Honkai'] = df['tags'].str.contains('崩壊').astype(int)
    df['is_Genshin'] = df['tags'].str.contains('原神').astype(int)

    col_lst = ['pid','date', 'like_rate', 'mark_rate',
               'is_comic', 'is_ai', 'is_Genshin', 'is_Honkai',
               'comments','views', 'rank', 'top_cnt','date_diff_day']
    return df[col_lst]


def show_plt(df):
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    plt.plot(df.index, df['mean_likes_origin'], label='man')
    plt.plot(df.index, df['mean_likes_ai'], label='ai')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    top_man = pd.read_csv('../../web_scrapt/project/data/tops.csv')
    top_ai = pd.read_csv('../../web_scrapt/project/data/tops_ai.csv')
    top = pd.concat([top_man, top_ai], ignore_index=True )
    tobit = get_tobit(top)
    # tobit.to_csv('../data/pixiv_tops_lm.csv', index=False)

    top_man = get_time_series_df(top_man)
    top_ai = get_time_series_df(top_ai)
    merged = pd.merge(top_man, top_ai, on='date', how='outer')
    print(merged.head())
    merged = merged.rename(columns={'mean_like_rate_x': 'mean_likes_origin', 'mean_like_rate_y': 'mean_likes_ai'})
    merged.to_csv('../data/pixiv_likes_ts.csv', index=False)
    show_plt(merged)

