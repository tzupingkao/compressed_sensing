import numpy as np
from scipy.fft import fft, fftshift, ifft, ifftshift
import plotly.express as xp
import plotly.graph_objects as go

import matplotlib.pyplot as plt




l = 128
n = 5
sigma = 0.05
np.random.seed(42)

#generate sparse signal
x = np.concatenate( (np.ones(n) / n , np.zeros(l-n)) , axis=0 )
x = np.random.permutation(x)
# add random noise
y = x + sigma * np.random.randn(l)

fig = go.Figure()
fig.add_trace(
    go.Scatter( x=  np.arange(l) , y = x , name='x')
)
fig.add_trace(
    go.Scatter( x=  np.arange(l) , y = y , name='y')
)

fig = go.Figure()
for lam in [0.01,0.05, 0.1, 0.2]:
    fig.add_trace(
        go.Scatter( x=  np.arange(l) , y = 1/(1+lam) * y , name=f"lambda = {str(lam)}")
    )
fig.show()


def soft_thresh(x, lam):
    if ~(isinstance(x[0], complex)):
        return np.zeros(x.shape) + (x + lam) * (x<-lam) + (x - lam) * (x>lam)
    else:
        return np.zeros(x.shape) + ( abs(x) - lam ) / abs(x) * x * (abs(x)>lam)

fig = go.Figure()
for i, lam in enumerate([0.01,0.05, 0.1, 0.2]):
    fig.add_trace(
        go.Scatter( x=  np.arange(l) , y = soft_thresh(y, lam)+i/10 , name=f"lambda = {str(lam)}")
    )
fig.show()

def fftc(x):
    """Computes the centered Fourier transform"""
    return fftshift( fft(x) )

def ifftc(X):
    """Inverses the centered Fourier transform"""
    return ifft( ifftshift(X) )

X = fftc(x)
Y = fftc(y)
fig = go.Figure()
fig.add_trace( go.Scatter(x = np.arange(-l/2,l-1),y=abs(X) , name='X') )
fig.add_trace( go.Scatter(x = np.arange(-l/2,l-1),y=abs(Y) , name='Y') )
fig.show()


#uniformly sampled k-space
Xu = 4 * X
for i in range(1,4):
    Xu[i::4] = 0
#reconstructed signal
xu = ifftc(Xu)

#randomly sampled k-space
Xr = 4 * X * np.random.permutation(np.repeat([1,0,0,0], l/4) )
xr = ifftc( Xr )

#plot the comparison
fig = go.Figure()
fig.add_trace( go.Scatter(y=x*1.5, name='original signal (scaled)'))
fig.add_trace( go.Scatter(y=xu.real, name='uniform sampling'))
fig.add_trace( go.Scatter(y=xr.real, name='random sampling'))
fig.show()


# undersampled noisy signal in k-space and let this be first order Xhat
Y = 4 * fftc(x) * np.random.permutation(np.repeat([1,0,0,0], l/4) )
Xhat = Y.copy()


# Repeat steps 1-4 until change is below a threshold
eps = 1e-4
lam = 0.05

def distance(x,y):
    return abs(sum(x-y))
diff=[]
err = []
itermax = 10000
while True:
    itermax -= 1
    xhat_old = ifftc(Xhat)
    xhat = soft_thresh(xhat_old, lam)
    diff.append(distance(xhat, xhat_old))
    err.append(distance(xhat.real/4,x))
    if ( diff[-1] < eps ) | ( itermax == 0 ):
        break
    Xhat = fftc(xhat)
    Xhat[Y!=0] = Y[Y!=0]

fig = go.Figure()
fig.add_trace( go.Scatter( y = x , name = 'true signal') )
fig.add_trace( go.Scatter( y = ifftc(Y).real, name = 'reconstruction before noise reduction'))
fig.add_trace( go.Scatter( y = xhat.real/4, name = 'reconstructed after noise reduction'))
fig.show()


fig = go.Figure()
fig.add_trace( go.Scatter( y = err) )
fig.update_layout( title = 'Error at each step' )
fig.show()


fig = go.Figure()
fig.add_trace( go.Scatter( y = diff) )
fig.update_layout( title = 'Differential error at each step' )
fig.show()



