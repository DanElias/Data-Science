"""
Author: Daniel Elias Becerra
Date: 03/2021
Random Number generation with the Linear Congruential Method
Uses Matplotlib and a frquency table to plot the histogram for the numbers
Chi Squared Test + Run Test for Randomness
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Random_Number_Generator:
  def __init__(self):
    """
    Constructor for Random_Number_Generator Class
    """
    # Read Excel
    self.random_df = pd.read_excel("random_numbers.xlsx", usecols=["TRUNC"])
    self.random_df["R"] = self.random_df["TRUNC"]
    self.constants_df = pd.read_excel("random_numbers.xlsx", usecols=["m","a", "c", "x0"])
    self.constants_df = self.constants_df.iloc[0]
    self.N = self.random_df.shape[0]

    #Constants for frequency table creation
    self.DECIMALS = 6
    self.MAX = round(self.random_df["R"].max(), self.DECIMALS)
    self.MIN = round(self.random_df["R"].min(), self.DECIMALS)
    self.C = math.ceil(1 + 3.3 * math.log10(self.N))
    self.W = round((self.MAX - self.MIN) / self.C, self.DECIMALS)
    self.SUB_W = 0.000001

    # Plot histogram
    self.classes_df = self.plot_histogram()
    # Get results for run test
    self.results_df = self.run_test()
    # Get results for chi 2 test
    self.results_df["CHI2_RESULT"] = self.chi2()
    # Save results in txt
    self.save_results()
  
  def plot_histogram(self):
    """
    Plots an histogram of frequencies for different classes/bins
    Created with the random numbers
    """
    # Frequency table generation
    classes_df = pd.DataFrame(columns=["Class Number", "Lower Limit", "Upper Limit", "m", "f"])
    classes_df["Class Number"] = np.arange(1, self.C + 2, 1)
    classes_df = classes_df.set_index("Class Number")
    classes_df["Lower Limit"] = np.arange(self.MIN, self.MAX + self.W, self.W + self.SUB_W)
    classes_df["Lower Limit"] = classes_df["Lower Limit"] - self.SUB_W
    classes_df["Lower Limit"].loc[1] = classes_df["Lower Limit"].loc[1] - self.SUB_W
    classes_df["Upper Limit"] = classes_df["Lower Limit"] + self.W
    classes_df["Upper Limit"].loc[1] = classes_df["Upper Limit"].loc[1] - self.SUB_W
    classes_df = classes_df.drop(classes_df.index[self.C])
    classes_df["m"] = (classes_df["Lower Limit"] + classes_df["Upper Limit"]) / 2
    for i in range(1, self.C + 1, 1):
      classes_df["f"][i] = len(
          self.random_df[
            (self.random_df["R"] >= classes_df.loc[i]["Lower Limit"]) &
            (self.random_df["R"] <= classes_df.loc[i]["Upper Limit"])]
    )
    #Plot the histogram using Matplotlib = same as the last lines
    fig, axs = plt.subplots()
    axs.hist(self.random_df["R"], bins=8, rwidth=0.4)
    plt.show()
    return classes_df

  def chi2(self, ALPHA = 0.05):
    """
    Performs the chi2 test for uniformity on the random numbers
    """
    CHI2_CLASSES = 10
    X2_ALPHA = 16.9190
    CHI2_RESULT = ""
    N = self.random_df.shape[0]
    H0 = "With an alpha of " + str(ALPHA) + " Ri ~ Uniform[0,1], H0 is not rejected"
    H1 =  "With an alpha of " + str(ALPHA) + " Ri !~ Uniform[0,1], H0 is rejected"
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
          self.random_df[
            (self.random_df["R"] >= chi2_df.loc[i]["Lower Limit"]) &
            (self.random_df["R"] <= chi2_df.loc[i]["Upper Limit"])]
    )
    # Establish E =  N / Classes 
    chi2_df["E"] = math.ceil(N/CHI2_CLASSES)
    # Simple O-E operation
    chi2_df["O-E"] = (chi2_df["O"] - chi2_df["E"])
    # Simple (O-E)^2
    chi2_df["(O-E)^2"] = chi2_df["O-E"] * chi2_df["O-E"]
    # (O-E)^2 / E
    chi2_df["(O-E)^2/E"] = chi2_df["(O-E)^2"] / chi2_df["E"]
    ## Get x^2
    X2 = chi2_df["(O-E)^2/E"].sum()
    if X2 < X2_ALPHA:
      print(H0)
      CHI2_RESULT = H0
    else:
      print(H1)
      CHI2_RESULT = H1
    return CHI2_RESULT

  def run_test(self, ALPHA = 0.05):
    """
    Performs the Run Test fro Randomness on the the random numbers
    """
    # Define hypothesis
    H0_R = "With an alpha of " + str(ALPHA) + " Ri ~ Random[0,1], H0 is not rejected"
    H1_R = "With an alpha of " + str(ALPHA) + "Ri !~ Random[0,1]"
    NUMBER_OF_SIGNS = self.random_df.shape[0] - 1
    #First we obtain the signs
    i = 1
    signs = []
    while i < self.random_df.shape[0]:
      if self.random_df["R"][i - 1] < self.random_df["R"][i]:
        signs.append('+')
      else:
        signs.append('-')
      i += 1
    signs = np.array(signs)
    print("*** SIGNS ***")
    print(signs)
    print(np.array(self.random_df["R"]))
    # Calculate R Number of Runs
    last_sign = signs[0]
    R = 1
    if len(signs) == 0:
      R = 0
    for sign in signs:
      if sign != last_sign:
        R += 1
        last_sign = sign
    MIU_R = ((2*NUMBER_OF_SIGNS) - 1) / 3
    SIGMA_R2 = ((16*NUMBER_OF_SIGNS) - 29) / 90
    SIGMA_R = math.sqrt(SIGMA_R2)
    ZETA_R = (R - MIU_R) / SIGMA_R
    ZETA_ALPHA2 = 1.96 
    RUN_TEST_RESULT = ""
    if abs(ZETA_R) > ZETA_ALPHA2:
      print(H1_R)
      RUN_TEST_RESULT = H1_R
    else:
      print(H0_R)
      RUN_TEST_RESULT = H0_R
    #  Save run test results
    results_df = pd.DataFrame(columns=["MIU_R", "SIGMA_R", "R", "ZETA_R", "RUN_TEST_RESULT", "SIGNS"])
    results_df["MIU_R"] = [MIU_R]
    results_df["SIGMA_R"] = [SIGMA_R]
    results_df["R"] =  [R]
    results_df["ZETA_R"] = [ZETA_R]
    results_df["RUN_TEST_RESULT"] = [RUN_TEST_RESULT]
    results_df["SIGNS"] = [''.join([sign for sign in signs ])]
    results_df.index.name = "Row"
    return results_df

  def save_results(self):
    """
    Saves chi2 and run test results on a txt file
    """
    tfile = open('results.txt', 'w')
    tfile.write(self.results_df.to_string())
    tfile.close()

x = Random_Number_Generator()