from abc import ABC, abstractmethod
 
class Model(ABC):


    @abstractmethod
    def build_model(self): 
        pass
    
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X, y): 
        pass

    @abstractmethod
    def cross_validate (self, X, y): 
        pass
