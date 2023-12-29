""" 
Plot distribution of malignancy scores for nodules
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_malign_dist(target_file, save_path=None):
    nodule_target = pd.read_csv(target_file)
    malignancy_scores = nodule_target.iloc[:, 1]
    # make histogram transparent, make bins centered on integers
    bins = np.arange(0.5, 6.5, 1)
    plt.hist(malignancy_scores, bins=bins, histtype='barstacked', rwidth=0.3, alpha=0.5, align='mid')
    plt.xlabel('Malignancy score')
    plt.xticks(np.arange(1, 6))
    plt.ylabel('Number of nodules')
    plt.savefig('/home/vadim/Development/Projects/DeepL-LIDC/results/malignancy_dist.png', dpi=200)
    plt.show()
    
if __name__ == '__main__':
    target_file = '/home/vadim/Development/Projects/DeepL-LIDC/data/nodule_target.csv'
    plot_malign_dist(target_file)