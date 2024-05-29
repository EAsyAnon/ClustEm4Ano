from typing import List
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering


def map(labels_1: List[int], labels_2: List[int]) -> List[int]:
    result = []

    for idx, label in enumerate(labels_1):
        result.append(labels_2[label])

    return result


def get_kmeans_cluster(labels: List[int], n_clusters: int, vectors: List):
    kmeans = (KMeans(n_clusters=n_clusters, init='k-means++', algorithm='lloyd', n_init='auto'))
    kmeans.fit(np.array(vectors))
    print(kmeans.labels_)
    labels_categorized = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    labels_categorized = map(labels, labels_categorized)
    labels = labels_categorized
    return labels, cluster_centers


def get_agglomerative_cluster(n_clusters: int, vectors: List):
    agg = (AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean', compute_distances=True))
    agg.fit(np.array(vectors))
    labels_categorized = agg.labels_
    labels = labels_categorized
    return labels, agg


def get_categories(values: List[str], labels: List[int]) -> List[List[str]]:
    num_of_categories = max(labels) + 1
    categories = []
    for i in range(num_of_categories):
        category = [value for value, label_ in zip(values, labels) if label_ == i]
        categories.append(category)
    return categories
