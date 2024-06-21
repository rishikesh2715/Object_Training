import numpy as np
import scipy.stats as stats
import pandas as pd

data = pd.read_csv('data_log.csv')

# Concatenate all values into a single string and then into a list of bits (integers)
bitstream = ''.join(data['values']).strip()
bitstream = [int(bit) for bit in bitstream]

# Print the last element of the bitstream
print(bitstream[-1])

# Frequency Test
num_ones = np.sum(bitstream)
num_zeros = len(bitstream) - num_ones
print(f'Frequency Test: Ones: {num_ones}, Zeros: {num_zeros}')

# Shannon Entropy
prob_ones = num_ones / len(bitstream)
prob_zeros = num_zeros / len(bitstream)
entropy = - (prob_ones * np.log2(prob_ones) + prob_zeros * np.log2(prob_zeros))
print(f'Shannon Entropy: {entropy:.4f} bits per bit')

# Chi-Square Test
observed_frequencies = [num_zeros, num_ones]
expected_frequencies = [len(bitstream) / 2, len(bitstream) / 2]
chi_square_stat, p_value = stats.chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)
print(f'Chi-Square Test: Chi-square stat: {chi_square_stat:.4f}, p-value: {p_value:.4f}')
