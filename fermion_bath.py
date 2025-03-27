import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
import matplotlib.pyplot as plt
from scipy.integrate import quad

def tdiff(D, t2, t1):
    """
    Create two time object from one time object
    """
    return D[t2 - t1] if t2 >= t1 else np.conj(D[t1 - t2])

def flatBand(w, wC, v, gamma):
    """
    DOS for a flat band with soft cutoff
    """
    return gamma / ((np.exp(v * (w - wC)) + 1) * (np.exp(-v * (w + wC)) + 1))

def fermi_function(w, beta, mu):
    return 1 / (1 + np.exp(beta * (w - mu)))

def genFermionBath(T, mu, tmax, dt, dw, wC):
    """
    Generate Hybridization function for Fermion bath with a wide, flat band
    """
    # additional parameters to generate the flat band
    v = 10
    gamma = 1

    beta = 1.0 / T
    Cut  = np.pi / dt
    N = int(2*Cut/dw)

    t    = np.arange(0, tmax, dt)
    w    = np.arange(-Cut, Cut, dw)

    fermionBath = np.zeros((2, 2, len(t), len(t)), complex)  # indices are gtr/les | spin up/spin down

    # frequency-domain Hybridization function
    Hyb_les = flatBand(w, wC, v, gamma) * fermi_function(w, beta, mu)
    Hyb_gtr = flatBand(w, wC, v, gamma) * (1 - fermi_function(w, beta, mu))
    fHyb_les = ifftshift(fft(fftshift(Hyb_les))) * dw/np.pi
    fHyb_gtr = ifftshift(fft(fftshift(Hyb_gtr))) * dw/np.pi

    # get real times from fft_times
    Hyb_init = np.zeros((2, 2, len(t)), complex)  # greater/lesser | spin up/spin down

    for t1 in range(len(t)):
        Hyb_init[0, :, t1] = fHyb_gtr[int(N / 2) + t1]
        Hyb_init[1, :, t1] = fHyb_les[int(N / 2) + t1]

    for t1 in range(len(t)):
        for t2 in range(len(t)):
            fermionBath[0, 0, t2, t1] = tdiff(Hyb_init[0, 0], t2, t1)
            fermionBath[0, 1, t2, t1] = tdiff(Hyb_init[0, 1], t2, t1)
            fermionBath[1, 0, t1, t2] = tdiff(Hyb_init[1, 0], t2, t1)
            fermionBath[1, 1, t1, t2] = tdiff(Hyb_init[1, 1], t2, t1)

    np.savez_compressed('FermionBath', t=t, F=fermionBath)

