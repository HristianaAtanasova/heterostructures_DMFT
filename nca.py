#!/usr/bin/env python3
  
import numpy as np
from numpy.linalg import solve
import bareprop
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from datetime import datetime

########## Define functions for impurity solver ##########

def delta(f, i):
    return 1 if f == i else 0

def weights(A):
    if len(A) == 1:
        return 0
    if len(A) == 2:
        return 0.5*(A[0]+A[1])
    if len(A) == 3:
        return (A[0]+4*A[1]+A[2])/3
    return np.sum(A) - (A[0]+A[-1])*2/3 + (A[1]+A[-2])*7/24 - (A[2]+A[-3])*1/6 + (A[3]+A[-4])*1/24

def w(A):
    if len(A) == 1:
        return 0
    if len(A) == 2:
        return (1/2)**2
    if len(A) == 3:
        return (1/3)**2
    if len(A) == 4:
        return (3/8)**2
    return (1/3)**2

# Self energy for vertex functions
def SelfEnergy(K, DeltaMatrix, Sigma):
    for f in range(4):
        for i in range(4):
            Sigma[f] += K[i] * DeltaMatrix[i, f]

def fillDeltaMatrix(DeltaMatrix, Delta):
    # fill in DeltaMatrix for all times | first index is gtr/les | second is spin up/spin down

    # initial dot state |0> can only go to |up> or |down> with a Delta_gtr
    DeltaMatrix[0, 1] = Delta[1, 0]
    DeltaMatrix[0, 2] = Delta[1, 1]

    # initial dot state |up> can only go to |0> or |up,down>
    DeltaMatrix[1, 0] = Delta[0, 0]
    DeltaMatrix[1, 3] = Delta[1, 1]

    # initial dot state |down> can only go to |0> or |up,down>
    DeltaMatrix[2, 0] = Delta[0, 1]
    DeltaMatrix[2, 3] = Delta[1, 0]

    # initial dot state |up,down> can only go to |up> or |down> with a Delta_les
    DeltaMatrix[3, 1] = Delta[0, 1]
    DeltaMatrix[3, 2] = Delta[0, 0]

def solve(init_state, site, t, U, T, G_0, phonon, fermion, Lambda, dissBath, output, Delta):
    ########## Computation of bold propagators on separate branches of the contour ##########
    #
    dt = t[1] - t[0]

    # initialize bare propagators
    start = datetime.now()

    # indices are initial and final states
    DeltaMatrix = np.zeros((4, 4, len(t), len(t)), complex)
    fillDeltaMatrix(DeltaMatrix, Delta)

    # indices are initial state, contour times located on the same branch t_n, t_m (propagation from t_m to t_n)
    G     = np.zeros((4, len(t), len(t)), complex)
    Sigma = np.zeros((4, len(t), len(t)), complex)

    # main loop over every pair of times t_n and t_m (located on the same contour-branch), where t_m is the smaller contour time
    # take integral over t1 outside the t_m loop
                                                                   
    for t_n in range(len(t)):
        sum_t1_upper = np.zeros((4, t_n+1), complex)
        for t_m in range(t_n, -1, -1):

            sum_t2_upper = np.zeros(4, complex)
            # sum_t2_lower = np.zeros(4, complex)
            for i in range(4):
                sum_t2_upper[i] = dt**2 * weights(G[i, t_m:t_n+1, t_m] * sum_t1_upper[i, t_m:])

            # Dyson equation for time (t_m, t_n)
            G[:, t_n, t_m] = (G_0[:, t_n, t_m] - sum_t2_upper) / (1 + dt ** 2 * G_0[:, t_n, t_n] * Sigma[:, t_n, t_n] * w(G[i, t_m:t_n+1, t_m]*sum_t1_upper[i, t_m:]))

            # Compute self-energy for time (t_m, t_n)
            Sigma[:, t_n, t_m] = np.sum(G[None, :, t_n, t_m] * DeltaMatrix[:, :, t_n, t_m], 1)
            # Sigma[:, t_m, t_n] = np.sum(G[None, :, t_m, t_n] * DeltaMatrix[:, :, t_m, t_n], 1)

            if phonon == 1:
                Sigma[0, t_n, t_m] += Lambda[t_n, t_m] * (G[0, t_n, t_m] * dissBath[t_n, t_m])

            elif fermion == 1:
                Sigma[0, t_n, t_m] += Lambda[t_n, t_m] * (G[1, t_n, t_m] * dissBath[1, 0, t_n, t_m] + G[2, t_n, t_m] * dissBath[1, 1, t_n, t_m])
                Sigma[1, t_n, t_m] += Lambda[t_n, t_m] * (G[0, t_n, t_m] * dissBath[0, 0, t_n, t_m] + G[3, t_n, t_m] * dissBath[1, 1, t_n, t_m])
                Sigma[2, t_n, t_m] += Lambda[t_n, t_m] * (G[0, t_n, t_m] * dissBath[0, 1, t_n, t_m] + G[3, t_n, t_m] * dissBath[1, 0, t_n, t_m])
                Sigma[3, t_n, t_m] += Lambda[t_n, t_m] * (G[1, t_n, t_m] * dissBath[0, 1, t_n, t_m] + G[2, t_n, t_m] * dissBath[0, 0, t_n, t_m])

            for i in range(4):
                sum_t1_upper[i, t_m] = weights(Sigma[i, t_m:t_n+1, t_m] * G_0[i, t_n, t_m:t_n+1])  # sum[:, t2=t_m]


    # Note that the propagator for the lower branch is calculated explicitly (see Notes)
    for t_m in range(len(t)):
        sum_t1 = np.zeros((4, t_m+1), complex)
        for t_n in range(t_m, -1, -1):
            sum_t2 = np.zeros(4, complex)
            for i in range(4):
                sum_t2[i] = dt**2 * weights(G[i, t_n, t_n:t_m+1] * sum_t1[i, t_n:])

            # Dyson equation for time (t_m, t_n)
            G[:, t_n, t_m] = (G_0[:, t_n, t_m] - sum_t2) / (1 + dt ** 2 * G_0[:, t_n, t_n] * Sigma[:, t_n, t_n] * w(G[i, t_n, t_n:t_m+1]*sum_t1[i, t_n:]))

            # Compute self-energy for time (t_m, t_n)
            Sigma[:, t_n, t_m] = np.sum(G[None, :, t_n, t_m] * DeltaMatrix[:, :, t_n, t_m], 1)

            if phonon == 1:
                Sigma[0, t_n, t_m] += Lambda[t_n, t_m] * (G[0, t_n, t_m] * dissBath[t_n, t_m])
                Sigma[3, t_n, t_m] += Lambda[t_n, t_m] * (G[3, t_n, t_m] * dissBath[t_n, t_m])

            elif fermion == 1:
                Sigma[0, t_n, t_m] += Lambda[t_n, t_m] * (G[1, t_n, t_m] * dissBath[1, 0, t_n, t_m] + G[2, t_n, t_m] * dissBath[1, 1, t_n, t_m])
                Sigma[1, t_n, t_m] += Lambda[t_n, t_m] * (G[0, t_n, t_m] * dissBath[0, 0, t_n, t_m] + G[3, t_n, t_m] * dissBath[1, 1, t_n, t_m])
                Sigma[2, t_n, t_m] += Lambda[t_n, t_m] * (G[0, t_n, t_m] * dissBath[0, 1, t_n, t_m] + G[3, t_n, t_m] * dissBath[1, 0, t_n, t_m])
                Sigma[3, t_n, t_m] += Lambda[t_n, t_m] * (G[1, t_n, t_m] * dissBath[0, 1, t_n, t_m] + G[2, t_n, t_m] * dissBath[0, 0, t_n, t_m])

            for i in range(4):
                sum_t1[i, t_n] = weights(Sigma[i, t_n, t_n:t_m+1] * G_0[i, t_n:t_m+1, t_m])  # sum[:, t2=t_n]

    np.savez_compressed('Prop', t=t, Prop=G)

    print('-'*100)
    print('Finished calculation of bold propagators after', datetime.now() - start)
    print('\n')

    ########## Computation of Vertex Functions including hybridization lines between the upper and lower branch ##########

    # for every initial state i there is a set of 4 coupled equations for K
    for i in init_state:

        # indices are initial state | contour times on lower and upper branch
        K = np.zeros((4, 4, len(t), len(t)), complex)

        K_0 = np.zeros((4, 4, len(t), len(t)), complex)

        # indices are final state | lower and upper branch time
        Sigma = np.zeros((4, len(t), len(t)), complex)

        for t_n in range(len(t)):
            sum_t1 = np.zeros((4, len(t)), complex)
            for t_m in range(len(t)):

                sum_t2 = np.zeros(4, complex)
                M = np.eye(4, dtype=complex)

                for f in range(4):
                    K_0[i, f, t_n, t_m] = delta(i, f) * G[f, 0, None, t_n] * G[f, None, t_m, 0]

                    sum_t2[f] = dt**2 * weights(G[f, t_m, :t_m + 1] * sum_t1[f, :t_m + 1])

                    M[f] -= dt**2 * np.sum(DeltaMatrix[f, :, t_n, t_m] * np.conj(G[:, t_n, t_n]) * G[:, t_m, t_m], 0) * w(G[f, t_m, :t_m + 1] * sum_t1[f, :t_m + 1])

                # Dyson equation for time (t_m, t_n)
                K[i, :, t_n, t_m] = np.linalg.solve(M, K_0[i, :, t_n, t_m] + sum_t2)

                # Compute self-energy for time (t_m, t_n)
                SelfEnergy(K[i, :, t_n, t_m], DeltaMatrix[:, :, t_n, t_m], Sigma[:, t_n, t_m])

                if phonon == 1:
                    Sigma[0, t_n, t_m] += Lambda[t_n, t_m] * (K[i, 0, t_n, t_m] * dissBath[t_n, t_m])
                    Sigma[3, t_n, t_m] += Lambda[t_n, t_m] * (K[i, 3, t_n, t_m] * dissBath[t_n, t_m])

                elif fermion == 1:
                    Sigma[0, t_n, t_m] += Lambda[t_n, t_m] * (K[i, 1, t_n, t_m] * dissBath[1, 0, t_n, t_m] + K[i, 2, t_n, t_m] * dissBath[1, 1, t_n, t_m])
                    Sigma[1, t_n, t_m] += Lambda[t_n, t_m] * (K[i, 0, t_n, t_m] * dissBath[0, 0, t_n, t_m] + K[i, 3, t_n, t_m] * dissBath[1, 0, t_n, t_m])
                    Sigma[2, t_n, t_m] += Lambda[t_n, t_m] * (K[i, 0, t_n, t_m] * dissBath[0, 1, t_n, t_m] + K[i, 3, t_n, t_m] * dissBath[1, 1, t_n, t_m])
                    Sigma[3, t_n, t_m] += Lambda[t_n, t_m] * (K[i, 1, t_n, t_m] * dissBath[0, 1, t_n, t_m] + K[i, 2, t_n, t_m] * dissBath[0, 0, t_n, t_m])

                for f in range(4):
                    sum_t1[f, t_m] = weights(Sigma[f, :t_n+1, t_m] * G[f, :t_n+1, t_n])  # sum[:, t2=t_m]

        ########## Computation of two-times Green's functions ##########

        # Greens function with indices gtr/les | spin up/spin down | initial state | lower and upper branch time
        Green = np.zeros((2, 2, 4, len(t), len(t)), complex)

        for t1 in range(len(t)):
            for t2 in range(len(t)):
                Green[0, 0, i, t1, t2] = (K[i, 0, t1, t2] * G[1, t1, t2] + K[i, 2, t1, t2] * G[3, t1, t2])
                Green[1, 0, i, t1, t2] = (K[i, 1, t1, t2] * G[0, t1, t2] + K[i, 3, t1, t2] * G[2, t1, t2])
                Green[0, 1, i, t1, t2] = (K[i, 0, t1, t2] * G[2, t1, t2] + K[i, 1, t1, t2] * G[3, t1, t2])
                Green[1, 1, i, t1, t2] = (K[i, 2, t1, t2] * G[0, t1, t2] + K[i, 3, t1, t2] * G[1, t1, t2])

        print('Finished calculation of K for initial state', i, 'after', datetime.now() - start)
        err = np.abs(1 - np.real(np.sum(K[i, :, len(t)-1, len(t)-1], 0)))
        print('Error for inital state =', i, 'is', err)
        print('\n')

        print('1 - (Green_gtr + Green_les) for Spin Up site', i, 'is', 1 - np.real(Green[0, 0, i, len(t)-1, len(t)-1] + Green[1, 0, i, len(t)-1, len(t)-1]))
        print('1 - (Green_gtr + Green_les) for Spin Down site', i, 'is', 1 - np.real(Green[0, 1, i, len(t)-1, len(t)-1] + Green[1, 1, i, len(t)-1, len(t)-1]))

        print('\n')

        print('Final population for Spin Up on site', i, 'is', np.real(Green[1, 0, i, len(t) - 1, len(t) - 1]))
        print('Final population for Spin Down on site', i, 'is', np.real(Green[1, 1, i, len(t) - 1, len(t) - 1]))

        print('\n')

        Vertexfunction = 'K_i={}_site={}'
        Greensfunction = 'Green_i={}_site={}'
        np.savez_compressed(Vertexfunction.format(init_state[0], site), t=t, K=K[i])
        np.savez_compressed(Greensfunction.format(init_state[0], site), t=t, Green=Green[:,:,i])

        return Green[:,:,i]

def main():
    parser = argparse.ArgumentParser(description = "run nca impurity solver")
    parser.add_argument("--U",    type=float, default = 2.0)
    args = parser.parse_args()

if __name__ == "__main__":
    main()
