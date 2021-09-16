from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS

def mds(S, *args, **kwargs):
    return MDS(dissimilarity='precomputed', *args, **kwargs).fit_transform(S)


def agglomerative_clustering(X, return_labels=True, *args, **kwargs):
    if X.shape[0] != X.shape[1]:
        cluster = AgglomerativeClustering(
            n_clusters=10,
            linkage='complete',
            *args,
            **kwargs
        )
    else:
        cluster = AgglomerativeClustering(
            n_clusters=10,
            affinity='precomputed',
            linkage='complete',
            *args,
            **kwargs
        )
    
    if return_labels:
        return cluster.fit_predict(X)
    
    cluster.fit(X)
    return cluster