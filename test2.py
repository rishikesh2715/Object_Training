import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

data = pd.read_csv('data_log.csv')  # chnage the name to the csv name


values = data['values']

# Calculate the mean and standard deviation of the data
mean = np.mean(values)
std_dev = np.std(values)

x = np.linspace(min(values), max(values), 100)

#Gaussian distribution
gaussian = norm.pdf(x, mean, std_dev)

# plot the histogram
plt.hist(values, bins=30, density=True, alpha=0.6, color='g')

# Gaussian distribution
plt.plot(x, gaussian, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mean, std_dev)
plt.title(title)

# Show the plot
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.show()
