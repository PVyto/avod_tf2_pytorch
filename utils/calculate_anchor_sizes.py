import os
import json
import numpy as np
from sklearn.cluster import KMeans


class AnchorSizeCalculator:
    def __init__(self, labels, classes=[], num_clusters=[], file_name='clusters.json', type='seperate', alg='classic'):
        self.clusters = {}
        self.labels = labels
        self.file = file_name
        self.classes = classes
        self.num_clusters = num_clusters

    def __call__(self, *args, **kwargs):
        return self._calculate_clusters()

    def filter_labels(self, current_class):
        return self.labels.loc[self.labels['class'] == current_class, ['l', 'w', 'h']].to_numpy()

    def _load_clusters(self):
        if os.path.exists(self.file):
            self.clusters = json.load(self.file)

    def _save_clusters(self):
        with open(self.file, 'w') as f:
            json.dump(self.clusters, f)

    def _calculate_clusters(self):
        clusters = []
        for cls, k in zip(self.classes, self.num_clusters):
            current_clusters = []
            labels = self.filter_labels(cls)
            kmeans = KMeans(n_clusters=k).fit(labels)
            self.clusters[cls] = []
            for cluster in range(k):
                cluster_to_append = np.round(kmeans.cluster_centers_[cluster], 3).tolist()
                self.clusters[cls].append(cluster_to_append)
                current_clusters.append(cluster_to_append)
            clusters.append(current_clusters)
        self._save_clusters()
        return clusters
