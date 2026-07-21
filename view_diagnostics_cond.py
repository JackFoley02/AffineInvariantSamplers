#File to make and package arrays of sampler parameters by dimension.
import numpy as np
import json
import os


#some constants
COND = [1, 3.162, 10, 31.62, 100, 316.2, 1000]
SEEDS = ['00000', '00001', '00002', '00003', '00004']
DISTRIBUTION = 'Rosenbrock'


#initializing empty arrays for storage

ESS_med = np.empty((4, 7))
ESS_top = np.empty((4, 7))
ESS_bot = np.empty((4, 7))

#main loop
for idx, c in enumerate(COND):
    print(f"Storing data for {c} condition...")
    dim_results_NUTS = []
    dim_results_HWM = []
    dim_results_SM = []
    dim_results_HMC = []
    for seed in SEEDS:
        file = f'{DISTRIBUTION}ResultsC_agnostic/{c}c/seeds/seed_{seed}/results_summary.json'
        with open(file, 'r') as f:
            data = json.load(f)
            print(f"Loaded {file}")
            dim_results_NUTS.append(data['summary']['Dense-mass NUTS']['ess_per_eval'])
            dim_results_HWM.append(data['summary']['Hamiltonian Walk Move']['ess_per_eval'])
            dim_results_SM.append(data['summary']['Stretch Move']['ess_per_eval'])
            dim_results_HMC.append(data['summary']['HMC']['ess_per_eval'])

    ESS_med[0, idx] = np.nanmedian(dim_results_NUTS)
    ESS_top[0, idx] = np.nanmax(dim_results_NUTS)
    ESS_bot[0, idx] = np.nanmin(dim_results_NUTS)
    ESS_med[1, idx] = np.nanmedian(dim_results_HWM)
    ESS_top[1, idx] = np.nanmax(dim_results_HWM)
    ESS_bot[1, idx] = np.nanmin(dim_results_HWM)
    ESS_med[2, idx] = np.nanmedian(dim_results_SM)
    ESS_top[2, idx] = np.nanmax(dim_results_SM)
    ESS_bot[2, idx] = np.nanmin(dim_results_SM)
    ESS_med[3, idx] = np.nanmedian(dim_results_HMC)
    ESS_top[3, idx] = np.nanmax(dim_results_HMC)
    ESS_bot[3, idx] = np.nanmin(dim_results_HMC)

    
arrays = {'ESS_med':ESS_med, 'ESS_top':ESS_top, 'ESS_bot':ESS_bot}
np.savez(f'{DISTRIBUTION}_cond_zipped_results.npz', **arrays)
print('Saving Complete! Please transfer the arrays to your machine.')


DISTRIBUTION = 'Gaussian'


#initializing empty arrays for storage

ESS_med = np.empty((4, 7))
ESS_top = np.empty((4, 7))
ESS_bot = np.empty((4, 7))

#main loop
for idx, c in enumerate(COND):
    print(f"Storing data for {c} condition...")
    dim_results_NUTS = []
    dim_results_HWM = []
    dim_results_SM = []
    dim_results_HMC = []
    for seed in SEEDS:
        file = f'{DISTRIBUTION}ResultsC/{c}c/seeds/seed_{seed}/results_summary.json'
        with open(file, 'r') as f:
            data = json.load(f)
            print(f"Loaded {file}")
            dim_results_NUTS.append(data['summary']['Dense-mass NUTS']['ess_per_eval'])
            dim_results_HWM.append(data['summary']['Hamiltonian Walk Move']['ess_per_eval'])
            dim_results_SM.append(data['summary']['Stretch Move']['ess_per_eval'])
            dim_results_HMC.append(data['summary']['HMC']['ess_per_eval'])

    ESS_med[0, idx] = np.nanmedian(dim_results_NUTS)
    ESS_top[0, idx] = np.nanmax(dim_results_NUTS)
    ESS_bot[0, idx] = np.nanmin(dim_results_NUTS)
    ESS_med[1, idx] = np.nanmedian(dim_results_HWM)
    ESS_top[1, idx] = np.nanmax(dim_results_HWM)
    ESS_bot[1, idx] = np.nanmin(dim_results_HWM)
    ESS_med[2, idx] = np.nanmedian(dim_results_SM)
    ESS_top[2, idx] = np.nanmax(dim_results_SM)
    ESS_bot[2, idx] = np.nanmin(dim_results_SM)
    ESS_med[3, idx] = np.nanmedian(dim_results_HMC)
    ESS_top[3, idx] = np.nanmax(dim_results_HMC)
    ESS_bot[3, idx] = np.nanmin(dim_results_HMC)

    
arrays = {'ESS_med':ESS_med, 'ESS_top':ESS_top, 'ESS_bot':ESS_bot}
np.savez(f'{DISTRIBUTION}_cond_zipped_results.npz', **arrays)
print('Saving Complete! Please transfer the arrays to your machine.')



            

        
        
        