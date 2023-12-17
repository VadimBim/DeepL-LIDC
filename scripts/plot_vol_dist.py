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
#add text in the upper right corner with the number of nodules
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the full distribution
ax1.hist(mean_volume, density=False, bins=100)
ax1.set_xlabel('Volume (mm^3)')
ax1.set_title('Full Distribution of Nodule Volumes')
ax1.text(0.7, 0.8, 'n = ' + str(n_nodules), fontsize=12, transform=plt.gcf().transFigure)

# Plot the distribution in range [0, 5000]
ax2.hist(mean_volume, density=False, bins=100, range=(0, 5000))
ax2.set_xlabel('Volume (mm^3)')
ax2.set_title('Distribution of Nodule Volumes (0-5000)')

# # Hide y-axis values for both plots
# ax1.axes.get_yaxis().set_visible(False)
# ax2.axes.get_yaxis().set_visible(False)

# Save the figure
plt.savefig('/home/vadim/Development/Projects/DeepL-LIDC/results/volume_distribution.png', dpi=200)

# Show the plots
plt.show()
