from __future__ import annotations
from abc import abstractmethod
from typing import List
from backend.BertEmbeddingService import BertEmbeddingService


class AffinityStrategy():
    @abstractmethod
    def compute_affinity(self, data: List):
        pass

class BertEmbeddingAffinity(AffinityStrategy):
    def __init__(self, verb_weight=1.0, object_weight=1.0):
        self.bert_service = BertEmbeddingService(verb_weight, object_weight)

    def compute_affinity(self,
                         application_name,
                         labels,
                         linkage,
                         object_weight,
                         verb_weight,
                         distance_threshold,
                         metric):
        return self.bert_service.compute_affinity(application_name,
                                                  labels,
                                                  linkage,
                                                  object_weight,
                                                  verb_weight,
                                                  distance_threshold,
                                                  metric)