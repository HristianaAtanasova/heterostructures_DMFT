import numpy as np
import matplotlib.pyplot as plt
import argparse
import toml
from datetime import datetime
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift

import nca
import bareprop
import hybridization
import timedep_hybridization
import phonon_bath
import fermion_bath
import coupling_to_diss_bath
import electric_field
import constant_electric_field

def runNCA(init_state, site, U, T, G_0, tmax, dt, Delta, phonon, fermion, Lambda, dissBath, iteration, output):
    t  = np.arange(0, tmax, dt)

    hybsection = "dmft/iterations/{}/delta".format(iteration)
    gfsection  = "dmft/iterations/{}/gf".format(iteration)

    with h5py.File(output, "a") as h5f:
        hdf5.save_green(h5f, hybsection, Delta, (t,t))

    Green = nca.solve(init_state, site, t, U, T, G_0, phonon, fermion, Lambda, dissBath, output, Delta)

    with h5py.File(output, "r") as h5f:
        Green, _ = hdf5.load_green(h5f, gfsection)

    return Green

def runInch(U, tmax, dt, Delta):
    pass

def run_dmft(dim, U1, U2, T, pumpA, probeA, mu, v_0, tmax, dt, dw, tol, solver, phonon, fermion, Lambda, wC, t_diss_end, pumpOmega, t_pump_start, t_pump_end, probeOmega, t_probe_start, t_probe_end, lattice_structure, output, **kwargs):
    t  = np.arange(0, tmax, dt)
    
    msg = 'Starting DMFT loop for dim = {} | U1 = {} | U2={} | phonon = {} | fermion = {} | time = {} | dt = {}'.format(dim, U1, U2, T, phonon, fermion, tmax, dt)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    start = datetime.now()

    # gf indices: [site, gtr/les, up/down, time, time]
    Green     = np.zeros((4, 2, 2, len(t), len(t)), complex)
    Green_old = np.zeros((4, 2, 2, len(t), len(t)), complex)
    Delta = np.zeros((4, 2, 2, len(t), len(t)), complex)

    U = np.zeros((4), float)
    U[:2] = U1
    U[2:] = U2
    # calculate and load bare propagators
    bareprop.bare_prop(t, U, Uconst=U)
    loaded = np.load('barePropagators.npz')
    G_0 = loaded['G_0']

    # delta indices: [gtr/les, up/down, time, time]
    hybridization.genSemicircularHyb(T, mu, v_0, tmax, dt, dw)
    # hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    # timedep_hybridization.genWideBandHyb(T, mu, tmax, dt, dw)
    loaded = np.load('Delta.npz')
    for site in range(4):
        Delta[site] = loaded['D']

    if phonon == 1:
        phonon_bath.genPhononBath(t, mu, T)
        loaded = np.load('PhononBath.npz')
        dissBath = loaded['P']
    elif fermion == 1:
        fermion_bath.genFermionBath(T, mu, tmax, dt, dw, wC)
        loaded = np.load('FermionBath.npz')
        dissBath = loaded['F']
    else:
        dissBath = 0

    # coupling to the dissipation bath can be turned off
    Lambda = coupling_to_diss_bath.gen_timedepLambda(t, t_diss_end, Lambda)

    # option of turning on a pump field and/or a probe field
    v_1 = constant_electric_field.genv(pumpA, v_0, t, t_pump_start, t_pump_end, 0)
    v_2 = constant_electric_field.genv(pumpA, v_0, t, t_pump_start, t_pump_end, 1)
    v = v_0

    # set solver
    if solver == 0:
        Solver = runNCA
    elif solver == 1:
        Solver = runInch
    else:
        raise Exception("solver {:s} not recognized".format(solver))

    # DMFT self-consistency loop
    diff = np.inf
    iteration = 0
    while diff > tol:
        iteration += 1

        Green_old[:] = Green[:]
        spin_inital_state = {'site0': [1] ,'site1': [2], 'site2': [1], 'site3': [2]}
        for site in range(4):
            init_state = spin_inital_state['site{}'.format(site)]
            Green[site] = Solver(init_state, site, U[site], T, G_0[site], tmax, dt, Delta[site], phonon, fermion, Lambda, dissBath, iteration, output)

        diff = np.amax(np.abs(Green_old - Green))

        neighbours = {'site0': [0, 1] ,'site1': [0, 2], 'site2': [1, 3], 'site3': [2, 3]}
        l_coupling = v_2 / (dim * 2.0)
        r_coupling = v_1 / (dim * 2.0)
        if dim == 3:
            layer_coupling = (v_1 + v_2) * (1.0 - 2.0 / dim) + v_0 / (dim)
        else:
            layer_coupling = (v_1 + v_2) * (1.0 - 1.0 / dim)

        # antiferromagnetic self-consistency
        for site in range(4):
            N = neighbours['site{}'.format(site)]
            if site == 0:
                Delta[site, :, 0] = l_coupling * Green[N[0], :, 1] + layer_coupling * Green[site, :, 1] + r_coupling * Green[N[1], :, 0]
                Delta[site, :, 1] = l_coupling * Green[N[0], :, 0] + layer_coupling * Green[site, :, 0] + r_coupling * Green[N[1], :, 1]
            elif site == 3:
                Delta[site, :, 0] = l_coupling * Green[N[0], :, 0] + layer_coupling * Green[site, :, 1] + r_coupling * Green[N[1], :, 1]
                Delta[site, :, 1] = l_coupling * Green[N[0], :, 1] + layer_coupling * Green[site, :, 0] + r_coupling * Green[N[1], :, 0]
            else:
                Delta[site, :, 0] = l_coupling * Green[N[0], :, 0] + layer_coupling * Green[site, :, 1] + r_coupling * Green[N[1], :, 0]
                Delta[site, :, 1] = l_coupling * Green[N[0], :, 1] + layer_coupling * Green[site, :, 0] + r_coupling * Green[N[1], :, 1]
        msg = 'U = {}, iteration {}: diff = {} (elapsed time = {})'
        print(msg.format(U, iteration, diff, datetime.now() - start))

    msg = 'Computation finished after {} iterations and {} seconds'.format(iteration, datetime.now() - start)
    print('-'*len(msg))
    print(msg)
    print('-'*len(msg))

    return Green
def main():
    parser = argparse.ArgumentParser(description = "run dmft")
    parser.add_argument("--output",   default = "savetxt")
    parser.add_argument("--params",   default = "run.toml")
    parser.add_argument("--savetxt",  action  = "store_true")
    args = parser.parse_args()

    with open(args.params, "r") as f:
        params = toml.load(f)

    params.update(vars(args))
    Green = run_dmft(**params)

if __name__ == "__main__":
    main()
