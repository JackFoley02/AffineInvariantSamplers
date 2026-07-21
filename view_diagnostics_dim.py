#File to make and package arrays of sampler parameters by dimension.
import numpy as np
import json
import os


#some constants
DIMS = [2, 4, 8, 16, 32, 64, 128]
SEEDS = ['00000', '00001', '00002', '00003', '00004']
DISTRIBUTION = 'Rosenbrock'
agnosticism = '_agnostic'

#initializing empty arrays for storage

ESS_med = np.empty((4, 7))
ESS_top = np.empty((4, 7))
ESS_bot = np.empty((4, 7))

#main loop
for idx, d in enumerate(DIMS):
    print(f"Storing data for {d} dimensions...")
    dim_results_NUTS = []
    dim_results_HWM = []
    dim_results_SM = []
    dim_results_HMC = []
    for seed in SEEDS:
        file = f'{DISTRIBUTION}ResultsM{agnosticism}/{d}d/seeds/seed_{seed}/results_summary.json'
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
np.savez(f'{DISTRIBUTION}_dimension_zipped_results.npz', **arrays)
print('Saving Complete! Please transfer the arrays to your machine.')


DISTRIBUTION = 'Gaussian'
agnosticism = ''

#initializing empty arrays for storage

ESS_med = np.empty((4, 7))
ESS_top = np.empty((4, 7))
ESS_bot = np.empty((4, 7))

#main loop
for idx, d in enumerate(DIMS):
    print(f"Storing data for {d} dimensions...")
    dim_results_NUTS = []
    dim_results_HWM = []
    dim_results_SM = []
    dim_results_HMC = []
    for seed in SEEDS:
        file = f'{DISTRIBUTION}ResultsM{agnosticism}/{d}d/seeds/seed_{seed}/results_summary.json'
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
np.savez(f'{DISTRIBUTION}_dimension_zipped_results.npz', **arrays)
print('Saving Complete! Please transfer the arrays to your machine.')

            

        
        
        