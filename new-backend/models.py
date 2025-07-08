from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ClusterResult:
    """Represents a cluster result with hierarchical information."""
    id: int
    name: str
    description: str
    features: List[str]
    feature_count: int
    centroid: np.ndarray
    parent_cluster: Optional[int] = None
    child_clusters: List[int] = None
    is_sibling: bool = False
    sibling_distance: float = 0.0


@dataclass
class ProcessingResult:
    """Represents the complete processing result for a clustering operation."""
    reviews_count: int
    features_count: int
    optimal_clusters: int
    silhouette_score: float
    davies_bouldin_score: float
    dendrogram_threshold: float
    sibling_threshold: float
    clusters: List[ClusterResult]