from dataclasses import dataclass

@dataclass
class ThresholdCandidate:
    dendrogram_threshold: float
    sibling_threshold: float
    n_clusters: int
    silhouette_score: float
    davies_bouldin_score: float
    calinski_harabasz_score: float
    internal_cohesion: float
    cluster_separation: float
    hierarchy_quality: float
    overall_score: float

    def to_dict(self) -> dict:
        return {
            'dendrogram_threshold': self.dendrogram_threshold,
            'sibling_threshold': self.sibling_threshold,
            'n_clusters': self.n_clusters,
            'silhouette_score': self.silhouette_score,
            'davies_bouldin_score': self.davies_bouldin_score,
            'calinski_harabasz_score': self.calinski_harabasz_score,
            'internal_cohesion': self.internal_cohesion,
            'cluster_separation': self.cluster_separation,
            'hierarchy_quality': self.hierarchy_quality,
            'overall_score': self.overall_score
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ThresholdCandidate':
        return cls(
            dendrogram_threshold=data['dendrogram_threshold'],
            sibling_threshold=data['sibling_threshold'],
            n_clusters=data['n_clusters'],
            silhouette_score=data['silhouette_score'],
            davies_bouldin_score=data['davies_bouldin_score'],
            calinski_harabasz_score=data['calinski_harabasz_score'],
            internal_cohesion=data['internal_cohesion'],
            cluster_separation=data['cluster_separation'],
            hierarchy_quality=data['hierarchy_quality'],
            overall_score=data['overall_score']
        )