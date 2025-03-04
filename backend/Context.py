from . import Affinity_strategy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Context():
    def __init__(self, strategy: Affinity_strategy) -> None:
        self.affinity_strategy = strategy
    
    @property
    def strategy(self) -> Affinity_strategy:
        return self.affinity_strategy
    
    @strategy.setter
    def strategy(self, strategy: Affinity_strategy) -> None:
        self.affinity_strategy = strategy

    def use_affinity_algorithm(self, application_name, data, linkage, object_weight, verb_weight, distance_threshold, metric):
        logger.info(f"Using affinity algorithm with {self.strategy.__class__.__name__}")

        if isinstance(self.strategy, Affinity_strategy.TfidfEmbeddingService):
            logger.info("Calling TF-IDF affinity computation.")

        result = self.strategy.compute_affinity(application_name,
                                                data,
                                                linkage,
                                                object_weight,
                                                verb_weight,
                                                distance_threshold,
                                                metric)
        if result is None:
            logger.error("Affinity algorithm returned None. Possible recursion issue.")
            raise RuntimeError("Affinity algorithm failed.")

        return result
    