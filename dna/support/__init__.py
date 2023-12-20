
def sigmoid(x:float) -> float:
    import numpy as np
    return 1 / (1 + np.exp(-x))