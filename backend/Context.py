from . import Affinity_strategy
from typing import List

class Context():
    def __init__(self, strategy: Affinity_strategy) -> None:
        self.affinity_strategy = strategy
    
    @property
    def strategy(self) -> Affinity_strategy:
        return self.affinity_strategy
    
    @strategy.setter
    def strategy(self, strategy: Affinity_strategy) -> None:
        self.affinity_strategy = strategy

    def use_affinity_algorithm(self,
                               application_name,
                               data: List,
                               linkage,
                               object_weight,
                               verb_weight,
                               distance_threshold):
        return self.affinity_strategy.compute_affinity(application_name,
                                                       data,
                                                       linkage,
                                                       object_weight,
                                                       verb_weight,
                                                       distance_threshold)
    