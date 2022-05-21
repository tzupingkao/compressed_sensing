# Ref: http://fab.cba.mit.edu/classes/864.17/people/Yada.Pruksachatkun/CS.html

"""
COMPRESSION SENSING

Python translation of signal recovery from sparsity for one signal that is sparse under 
Discrete Cosine Transform.
TODO: Extend to multiple signals (as per the matlab version)
"""
from sklearn import linear_model
from scipy.fftpack import dct, idct
from scipy.sparse import coo_matrix
from matplotlib.pyplot import plot, show, figure, title
import numpy as np
# np.set_printoptions(threshold=np.nan)

# Initializing constants and signals
# Sine is A is y(t) = amplitude * sin(2 * pi * frequency * time) - time/sampling rate
N = 5000 # Number of time stamps
FS = 4e4 # Sampling rate
M = 200 #Sampling number
f1, f2 = 697, 1336  # Frequencies - number of cycles per second in KHz.
duration = 1. / 8
t = np.linspace(0, duration, int(duration * FS))
f = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
# Pick any two touchtone frequencies
plot(t, f)
title('Original Signal')
show()



f = np.reshape(f, (len(f), 1))
# Randomly sampling the test signal
k = np.random.randint(0, N, (M,)) # get random indices
k = np.sort(k)  # making sure the random samples are monotonic
b = f[k]
plot(t, f, 'b', t[k], b, 'r.')
title('Original Signal with Random Samples')
show()


D = dct(np.eye(N)) # Here, we model using the DCT
A = D[k, :]
lasso = linear_model.Lasso(alpha=0.001)# here, we use lasso to minimize the L1 norm
lasso.fit(A, b.reshape((M,)))
# Plotting the reconstructed coefficients and the signal
# Creates the fourier transform that will most minimize l1 norm
recons = idct(lasso.coef_.reshape((N, 1)), axis=0) # inverse fourier transfomr
figure()
plot(t,recons)
title('Reconstucted Signal')
show()
