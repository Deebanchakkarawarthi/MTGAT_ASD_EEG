import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

def cluster_embeddings(embeddings, n_clusters=3, seed=42):
    x = np.asarray(embeddings)
    if len(x) <= n_clusters: raise ValueError("Need more samples than clusters")
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(x)
    return {"labels": labels, "silhouette": float(silhouette_score(x,labels)),
            "davies_bouldin": float(davies_bouldin_score(x,labels))}

def stability_ari(labels_a, labels_b):
    return float(adjusted_rand_score(labels_a, labels_b))
