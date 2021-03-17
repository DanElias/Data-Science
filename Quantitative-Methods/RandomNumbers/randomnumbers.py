# -*- coding: utf-8 -*-
"""RandomNumbers.ipynb

# Random Numbers
- Generate pseudorandom uniformly distributed numbers with the Linear Congruential Method
- Build a frequency table and the histogram of the frequency of the generated numbers.
- Use the Chi-Squared Test to check if the generated numbers are uniformly distributed
- Use the Run Test to check the randomness of  the generated numbers

## Import python libraries
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

"""## Load Excel with random numbers generated with the *Linear Congruential Method*"""

from google.colab import drive

drive.mount("/content/gdrive")  
!pwd

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/gdrive/My Drive/IntelligentSystems/RandomNumbers"
!ls

random_df = pd.read_excel("random_numbers.xlsx", usecols=["xi","R"])
random_df.head(12)

random_df.shape

constants_df = pd.read_excel("random_numbers.xlsx", usecols=["m","a", "c", "x0"])
constants_df = constants_df.iloc[0]
constants_df.head()

"""## Make Classes Frequency Table and Histogram"""

N = random_df.shape[0]
N

DECIMALS = 6
DECIMALS

MAX = round(random_df["R"].max(), DECIMALS)
MAX

MIN = round(random_df["R"].min(), DECIMALS)
MIN

C = math.ceil(1 + 3.3 * math.log10(N))
C

W = (MAX-MIN)/C
W = round(W, DECIMALS)
W

SUB_W = 0.000001
SUB_W

"""### Frequency Table"""

classes_df = pd.DataFrame(columns=["Class Number", "Lower Limit", "Upper Limit", "m", "f"])
classes_df["Class Number"] = np.arange(1, C + 2, 1)
classes_df = classes_df.set_index("Class Number")

classes_df["Lower Limit"] = np.arange(MIN, MAX + W, W + SUB_W)
classes_df["Lower Limit"] = classes_df["Lower Limit"] - SUB_W
classes_df["Lower Limit"].loc[1] = classes_df["Lower Limit"].loc[1] - SUB_W

classes_df["Upper Limit"] = classes_df["Lower Limit"] + W
classes_df["Upper Limit"].loc[1] = classes_df["Upper Limit"].loc[1] - SUB_W

classes_df = classes_df.drop(classes_df.index[C])

classes_df["m"] = (classes_df["Lower Limit"] + classes_df["Upper Limit"]) / 2

for i in range(1, C + 1, 1):
  classes_df["f"][i] = len(
      random_df[
        (random_df["R"] >= classes_df.loc[i]["Lower Limit"]) &
        (random_df["R"] <= classes_df.loc[i]["Upper Limit"])]
  )

classes_df

"""
# Matplotlib Histogram
### Only takes two lines of code and no creation of table
"""

fig, axs = plt.subplots()
axs.hist(random_df["R"], bins=8, rwidth=0.4)
plt.show()

"""
# Chi-squared Test
"""

CHI2_CLASSES = 10
ALPHA = 0.05
DEGREES_OF_FREEDOM = CHI2_CLASSES - 1
H0 = "Ri ~ Uniform[0,1]"
H1 = "Ri !~ Uniform[0,1]"

# Define our DataFrame
chi2_df = pd.DataFrame(columns=["Class Number", "Lower Limit", "Upper Limit", "O", "E", "O-E", "(O-E)^2", "(O-E)^2/E"])

# There are always 10 classes
chi2_df["Class Number"] = np.arange(1, CHI2_CLASSES + 1, 1)

# Set the dataframe index to the class number
chi2_df = chi2_df.set_index("Class Number")

# Define Lower Limits for each class
chi2_df["Lower Limit"] = np.arange(0, 1, 0.1)

# Difine Upper Limits for each class
chi2_df["Upper Limit"] = chi2_df["Lower Limit"] + 0.1

# COUNTIF - Count the number of random numbers frequency for each class
for i in range(1, CHI2_CLASSES + 1, 1):
  chi2_df["O"][i] = len(
      random_df[
        (random_df["R"] >= chi2_df.loc[i]["Lower Limit"]) &
        (random_df["R"] <= chi2_df.loc[i]["Upper Limit"])]
)

# Establish E =  N / Classes 
chi2_df["E"] = math.ceil(N/CHI2_CLASSES)

# Simple O-E operation
chi2_df["O-E"] = (chi2_df["O"] - chi2_df["E"])

# Simple (O-E)^2
chi2_df["(O-E)^2"] = chi2_df["O-E"] * chi2_df["O-E"]

# (O-E)^2 / E
chi2_df["(O-E)^2/E"] = chi2_df["(O-E)^2"] / chi2_df["E"]

chi2_df

## Make sure the frequencies sum up to 100
print(chi2_df["O"].sum())

## Get x^2
X2 = chi2_df["(O-E)^2/E"].sum()
print(X2)

X2_ALPHA = 16.9190
print(X2_ALPHA)

if X2 < X2_ALPHA:
  print(H0)
else:
  print(H1)

"""# Run Test for Randomness"""

R_NUMBER_OF_RUNS = 24
H0_R = "Ri ~ Random[0,1]"
H1_R = "Ri !~ Random[0,1]"
NUMBER_OF_SIGNS = random_df.shape[0] - 1
print(NUMBER_OF_SIGNS)

#First we obtain the signs
i = 1
signs = []
while i < random_df.shape[0]:
  if random_df["R"][i - 1] < random_df["R"][i]:
    signs.append('+')
  else:
    signs.append('-')
  i += 1
signs = np.array(signs)
print("*** SIGNS ***")
print(signs)
print(np.array(random_df["R"]))

MIU_R = ((2*NUMBER_OF_SIGNS) - 1) / 3
print(MIU_R)

SIGMA_R2 = ((16*NUMBER_OF_SIGNS) - 29) / 90
print(SIGMA_R2)

SIGMA_R = math.sqrt(SIGMA_R2)
print(SIGMA_R)

ZETA_R = (R_NUMBER_OF_RUNS - MIU_R) / SIGMA_R
print(ZETA_R)

ZETA_ALPHA2 = 1.96 
if abs(ZETA_R) > ZETA_ALPHA2:
  print(H1_R)
else:
  print(H0_R)

"""# Save Results"""

results_df = pd.DataFrame(columns=["MIU_R", "SIGMA_R", "R", "ZETA_R", "SIGNS"])
results_df["MIU_R"] = [MIU_R]
results_df["SIGMA_R"] = [SIGMA_R]
results_df["R"] =  [R_NUMBER_OF_RUNS]
results_df["ZETA_R"] = [ZETA_R]
results_df["SIGNS"] = [''.join([sign for sign in signs ])]
results_df.index.name = "Row"
results_df.head()

tfile = open('results.txt', 'w')
tfile.write(results_df.to_string())
tfile.close()