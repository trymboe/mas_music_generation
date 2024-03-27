from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# Set the number of successes and the number of observations for each group
percent = np.array(
    [35.465175051386595, 33.524278060450854]
)  # number of successes (correct predictions)
nobs = np.array([29677, 29677])  # number of observations (test subjects)
count = percent * nobs / 100

count = np.array([4410, 4730])
nobs = np.array([10000, 10000])

# Perform the two-proportion z-test
z_stat, p_value = proportions_ztest(count, nobs)

print(z_stat, p_value)
