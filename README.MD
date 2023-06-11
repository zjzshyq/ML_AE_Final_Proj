ML1 final project
1、特征可视化
2、特征标准话，one-hot，特征制作
3、算法选择比较LM 、lightSVM、KNN、lightGBT、adaBoost、FM
4、参数k-fold
5、L1/L2 可视化

帮我写代码，任务如下：
1、加载csv为df0
2、df1=feat_mk(df0)函数，为聚类做特征处理，这步你不用写。
1、用kmeans聚类df1，k=5，聚类标签作为新的特征并入df0中

自动化特征交叉函数:
def feature_cross(df: pd.DataFrame):
    for col1 in df.columns:
        for col2 in df.columns:
            if sim_cos(col1, col2):
                continue
            else:
               new_feature = f'{feature1}_{feature2}'
               df[new_feature] = df[feature1] * df[feature2]
    return df
sim_cos() 如何两个特征名相似,就不交叉。 判断特征名是否相似的依据是，
两个名字存在一样的单词，比如mother_weight_gain和mother_delivery_weight都有weight， 说明相似。
但是不要考虑mother和father，比如father_age和father_education，不相似。列面中的单词以下划线分割。
列名中有NA，unknown单词的，不参与特征交叉。


这行：df[new_feature] = df[col1] * df[col2]，报警：PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`