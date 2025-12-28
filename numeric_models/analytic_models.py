from numpy import (sin as np_sin)
from numba import njit

@njit
def ishigami_u3_as_x(u_1, u_2, x, a=7, b=0.1):
        """Define the Ishigami function."""
        return np_sin(u_1) + a * np_sin(u_2)*np_sin(u_2) + b * x*x*x*x * np_sin(u_1)
def ishigami_vectorized_generator(u, constants: list=[7.0, 0.1]):
    u_1, u_2 = u
    a, b = constants
    def f(x):
        return ishigami_u3_as_x(u_1, u_2, x, a, b)
    return f

@njit
def fellmann_function(x, u_1, u_2):
    if x.ndim == 1:
        return -x[0]*x[0]+5*x[1]-u_1+u_2*u_2-1
    else:
        x_1 = x[:,0]
        x_2 = x[:,1]
        return -x_1*x_1+5*x_2-u_1+u_2*u_2-1
def fellmann_function_generator(u):
    u_1, u_2 = u
    def f(x):
        return fellmann_function(x, u_1, u_2)[:, None]
    return f

@njit 
def affine_linear(x, u_1, u_2, u_3, a=1, b=2, c=3):
    return a*u_1+b*u_2+c*u_3+x/2
def affine_linear_generator(u, a=4, b=2, c=4):
    u_1, u_2, u_3 = u
    def f(x):
        return affine_linear(x=x, u_1=u_1, u_2=u_2, u_3=u_3, a=a,b=b,c=c)
    return f