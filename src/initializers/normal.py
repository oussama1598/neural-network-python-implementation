import numpy as np

class NormalInitializer:
  def __init__(self):
    pass
    
  def initialize(self, n, m):
    return np.random.randn(n, m)

