import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# parameters characterizing the quench profile from U_ to Uconst

t_turn = 3.0
t0 = 1

def quench_U(t_, U, Uconst):
    if t_turn <= t_ <= t0+t_turn:
        return U + (Uconst-U)*(1/2 - 3/4 * np.cos(np.pi*(t_-t_turn)/t0) + 1/4 * (np.cos(np.pi*(t_-t_turn)/t0))**3)
    elif t_ < t_turn:
        return U
    return Uconst

def integrate_U(t1, t2, U, Uconst):
    return quad(quench_U,t2,t1,args=(U,Uconst))[0]

def bare_prop(t, U, Uconst):
    # bare propagator G_0 as a Matrix for convolution integrals
    # [site, state, time, time]
    G_0 = np.zeros((4, 4, len(t), len(t)), complex)
    U_ = np.zeros((4, len(t)), float)

    for t_ in range(len(t)):
        U_[:, t_] = quench_U(t[t_], U, Uconst)

    # Computation of bare propagators G_0
    if (U == Uconst).all:
        E = np.zeros((4, 4, len(t)), float)

        for site in range(4):
            E[site, 1, :] = -U[site]/2.0
            E[site, 2, :] = -U[site]/2.0

        for i in range(4):
            for t1 in range(len(t)):
                for t2 in range(t1+1):
                    G_0[:, i, t1, t2] = np.exp(-1j * E[:, i, t1-t2] * t[t1-t2])
"bareprop.py" 53L, 1768C                                                                            1,1           Top

