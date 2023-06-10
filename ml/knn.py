import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train_data_cleaned4cluster.csv')
X = df.drop('newborn_weight', axis=1)

# 设置不同的聚类个数
cluster_range = range(2, 10)
silhouette_scores = []

# 计算每个聚类个数对应的轮廓系数
for n_clusters in cluster_range:
    print('cluster_range:', n_clusters)
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# 绘制聚类个数与轮廓系数的关系图
plt.plot(cluster_range, silhouette_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.title('K-means clustering')
plt.show()
