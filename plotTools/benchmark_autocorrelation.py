#!/usr/bin/env python3
#imports
import matplotlib.pyplot as plt
import numpy as np
import os


def benchmark_autocorrelation(results: dict, outdir:str, distrib:str = ''):
    """
    Wrapper function to generate autocorrelation functions for all different samplers

    Arguments:
    ----------
    results:
        results dictionary from benchmark functions
    outdir:
        output directory path
    distrib:
        string identifier for the distribution this is applied to.
    """

    os.makedirs(outdir, exist_ok = True) #make output directory if it doesn't already exist

    samplers = list(results.keys())

    D = results[samplers[0]]['series'].shape[-1]

    cmap = plt.get_cmap('tab10')
    linestyles = ['-', '--', '-.', ':']

    plt.figure(figsize=(12, 6))
    plt.xlim(right = min(300 * D, 1000))

    endpoints = [] #storing endpoints

    for i, name in enumerate(samplers): #iterating over the samplers
        acf = results[name]["autocorrelation"]
        max_lag = min([300 * D, len(acf), 1000])
        plt.plot(np.arange(max_lag), acf[:max_lag], label=name, color = cmap(i%10), linestyle = linestyles[i % 4])

        end_x, namey, end_y = max_lag-1, name, acf[max_lag-1]
        endpoints.append([end_x, namey, end_y, cmap(i%10)])

    #sort the endpoints
    endpoints.sort(key=lambda e: e[2])

    minsep = 0.04 #minimum separation between two labels.

    for i in range(1, len(endpoints)):
        prev = endpoints[i-1][2]
        curr = endpoints[i][2]

        if curr <= prev + minsep:
            endpoints[i][2] = prev + minsep 

    for x, n, y, c in endpoints:
        plt.text(x + 5, y, n, color = c, va='center', fontsize = 10)


    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    outfile = os.path.join(outdir, f'autocorrelation_{distrib}.pdf')
    plt.savefig(outfile, dpi = 200)

    print('Saved', outfile)




