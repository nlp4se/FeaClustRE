from . import Affinity_strategy

class Context():
    def __init__(self, strategy: Affinity_strategy) -> None:
        self.affinity_strategy = strategy
    
    @property
    def strategy(self) -> Affinity_strategy:
        return self.affinity_strategy
    
    @strategy.setter
    def strategy(self, strategy: Affinity_strategy) -> None:
        self.affinity_strategy = strategy

    def use_affinity_algorithm(self):
        return self.affinity_strategy.compute_affinity()
    