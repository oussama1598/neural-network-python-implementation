import numpy as np

class HeInitializer:
  def __init__(self):
    pass
    
  def initialize(self, n, m):
    alpha = np.sqrt(6 / n)
    w = np.random.uniform(-alpha, alpha, size = (n, m))
    return w

