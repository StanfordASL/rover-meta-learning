import numpy as np
import torch

class TerrainType():
    def __init__(self, min_ht = 0.0, max_ht = 0.0):
        self.min_ht = min_ht
        self.max_ht = max_ht

class FlatTerrain(TerrainType):
    def __init__(self, const_ht = 0.0):
        self.const_ht = const_ht
        super().__init__(min_ht = self.const_ht, max_ht = self.const_ht)

    def heightAt(self, x):
        z = self.const_ht
        if np.ndim(x) == 0:
            return z
        if type(x) == torch.Tensor:
            return torch.full_like(x,z)
        return [z for i in x]
    
    def gradient(self, x):
        grad = 0.0
        if np.ndim(x) == 0:
            return grad
        if type(x) == torch.Tensor:
            return torch.full_like(x,grad)
        return [grad for i in x]

class SinusoidTerrain(TerrainType):
    def __init__(self, min_ht = 0.0, max_ht = 1.0, period = 2*np.pi):
        self.min_ht = min_ht
        self.max_ht = max_ht
        self.period = period
        super().__init__(self.min_ht, self.max_ht)

    def heightAt(self, x):
        z = 0.5*(self.max_ht + self.min_ht) + 0.5*(self.max_ht - self.min_ht) * np.sin(2*np.pi*x/self.period)
        return z
    
    def gradient(self, x):
        grad = 0.5*(self.max_ht - self.min_ht) * (2*np.pi/self.period) * np.cos(2*np.pi*x/self.period)
        return grad

class TriangularTerrain(TerrainType):
    def __init__(self, min_ht = 0.0, max_ht = 1.0, period = 2*np.pi):
        self.min_ht = min_ht
        self.max_ht = max_ht
        self.period = period
        super().__init__(self.min_ht, self.max_ht)

    def heightAt(self, x):
        p = self.period 
        exp = np.floor(2*x/p + 1/2)
        z = 0.5*(self.max_ht + self.min_ht) + 0.5*(self.max_ht - self.min_ht) * (4/p) *(x - (p/2)*exp)*(-1)**exp
        return z
    
    def gradient(self, x):
        exp = 2.0*x/self.period
        grad = (-1)**exp
        return grad