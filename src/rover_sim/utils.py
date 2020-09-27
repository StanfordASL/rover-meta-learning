import numpy as np

def circle(xc,yc,rad):
    th = np.arange(0,2.0*np.pi,0.1)
    x = xc + rad*np.cos(th)
    y = yc + rad*np.sin(th)
    return x,y