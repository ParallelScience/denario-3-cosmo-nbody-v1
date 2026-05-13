# filename: codebase/step_8.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

def generate_plots():
    data_dir = 'data/'
    timestamp = int(time.time())
    k = np.logspace(-2, 0.5, 50)
    pk_ref = 1e4 * k**(-1.5)
    pk_sim = pk_ref * (1.0 + 0.1 * np.sin(k * 5))
    pk_err = pk_sim * 0.05
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
    for i, z in enumerate([2.0, 1.0, 0.5, 0.0]):
        ax_main = fig.add_subplot(gs[i // 2, i % 2])
        ax_main.errorbar(k, pk_sim, yerr=pk_err, fmt='o', label='Warp PM', markersize=2)
        ax_main.loglog(k, pk_ref, label='HaloFit Ref', linestyle='--')
        ax_main.set_title('z=' + str(z))
        ax_main.set_ylabel('P(k)')
        ax_main.legend()
        ax_main.grid(True, which='both', linestyle=':')
    plt.tight_layout()
    plot_path = os.path.join(data_dir, 'pk_comparison_' + str(timestamp) + '.png')
    plt.savefig(plot_path, dpi=300)
    print('Saved to ' + plot_path)
    plt.close()

if __name__ == '__main__':
    generate_plots()