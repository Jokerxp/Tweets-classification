import pytorchtools
import numpy as np
from scipy.misc import derivative

a = np.array([-72, 0]).reshape((1,2))
b = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]]])
print(np.dot(a,b))
