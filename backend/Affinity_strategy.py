from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

class AffinityStrategy():
    @abstractmethod
    def compute_affinity(self, data: List):
        pass
    
class LevenshteinAffinity(AffinityStrategy):
    def compute_affinity(self, data: List):
        return None
    
class TfIdfAffinity(AffinityStrategy):
    def compute_affinity(self, data: List):
        return None
    
class CosineAffinity(AffinityStrategy):
    def compute_affinity(self, data: List):
        return super().compute_affinity(data)
    