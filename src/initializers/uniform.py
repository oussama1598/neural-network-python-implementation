import numpy as np

class UniformInitializer:
  def __init__(self):
    pass
    
  def initialize(self, n, m):
    return np.random.uniform(n, m)

