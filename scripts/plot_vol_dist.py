"""
Plot the distribution of volumes.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('/home/vadim/Development/Projects/DeepL-LIDC/data/merged.csv')
#TODO: rename ' volume' column to 'volume'
#make mean volume array. group by patient, then by nodule. find mean volume for each nodule
mean_volume = df.groupby(['patient_id', 'nodule'])[' volume'].mean()
n_nodules = len(mean_volume)
#number of nodules with volume < 5000
n_5000 = len(mean_volume[mean_volume < 5000])
#number of nodules with volume < 2000
n_2000 = len(mean_volume[mean_volume < 2000])
#add text in the upper right corner with the number of nodules
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))

# Plot the full distribution
ax1.hist(mean_volume, density=False, bins=100)
ax1.set_xlabel('Volume (mm^3)')
ax1.text(0.95, 0.95, 'n = ' + str(n_nodules), fontsize=12, transform=ax1.transAxes, ha='right', va='top')

# Plot the distribution in range [0, 5000]
ax2.hist(mean_volume, density=False, bins=100, range=(0, 5000))
ax2.set_xlabel('Volume (mm^3)')
ax2.text(0.95, 0.95, 'n = ' + str(n_5000), fontsize=12, transform=ax2.transAxes, ha='right', va='top')

# Plot the distribution in range [0, 2000]
ax3.hist(mean_volume, density=False, bins=100, range=(0, 2000))
ax3.set_xlabel('Volume (mm^3)')
ax3.text(0.95, 0.95, 'n = ' + str(n_2000), fontsize=12, transform=ax3.transAxes, ha='right', va='top')

# Set the title
fig.suptitle('Distribution of nodule volumes')

# Save the figure
plt.savefig('/home/vadim/Development/Projects/DeepL-LIDC/results/volume_distribution.png', dpi=200)

# Show the plots
plt.show()
