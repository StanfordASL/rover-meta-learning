import numpy as np

def barrier_func(x, alpha=1.0, d=0):
    if d == 0:
        return alpha**2*(np.cosh(x / alpha) - 1)
    if d == 1:
        return alpha*(np.sinh(x/alpha))
    if d == 2:
        return np.cosh(x/alpha)

def soft_abs(x, alpha=1.0, d=0):
    z = np.sqrt(alpha**2 + x**2)
    if d == 0:
        return z - alpha
    if d == 1:
        return x/z
    if d == 2:
        return alpha**2 / z**3

def quadratic(x, alpha=1.0, d=0):
    if d == 0:
        return x**2
    if d == 1:
        return 2.*x
    if d == 2:
        return 2. + 0*x

def linear(x, alpha=1.0, d=0):
    if d == 0:
        return x
    if d == 1:
        return 1. + 0*x
    if d == 2:
        return  0*x
