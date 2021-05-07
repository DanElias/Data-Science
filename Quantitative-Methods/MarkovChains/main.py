"""
Author: Daniel Elias
Date: May 2021
Stochastic Matrices and Markov Chains

User can:
1) Create his/her own N X N matrix
2) Check that the created matrix is a stochastic matrix (if not, the user must enter the matrix again)
3) Calculate the probability from going to one state to another in n steps. (Your program must show every power of the matrix until it reaches the number of steps)
4) Calculate the long-term state (steady) of the matrix. Indicate the minimum needed value for the power (if there is no steady state notify the user)
5) Identify if the matrix is regular or not.
6) Create automatically a 3 X 3 random stochastic matrix and let the user use steps 3, 4 and 5. (Each time you execute your program, the matrix must be different).
"""

import math
import pandas as pd
import numpy as np
import warnings
from numpy.linalg import matrix_power
warnings.filterwarnings('ignore')

class MarkovChain:
    def __init__(self):
        """
        Constructor
        """
        self.matrix = []
        self._run()
    
    def _run(self):
        """
        Displays the main menu and asks user to select from:
        a) Inputing their on matrix or
        b) Generating a random 3 x 3 matrix
        """
        close = False
        while not close:
            print("\nDo you want to:")
            print("a) Input your own matrix")
            print("b) Use a random 3 x 3 matrix")
            print("c) Exit the program")
            choice = input("Type the letter of your choice:")
            if choice == 'a':
                self._load_matrix()
                print("Now checking your matrix ...")
                self._check_matrix()
                print("Great! Your matrix is Stochastic ...")
                input("\nPress ENTER to continue")
                self._menu()
            elif choice == 'b':
                print("\nNow creating a 3 x 3 Random Stochastic Matrix")
                self._generate_random_stochastic_matrix()
                print("Checking the matrix ...")
                self._check_matrix()
                print("Great! The matrix is Stochastic ...")
                input("\nPress ENTER to continue")
                self._menu()
            elif choice == 'c':
                close = True
                print("\nGoodbye!")
                exit()
            else:
                print("Invalid option, choose an option")
                input("Press ENTER to continue")

    def _menu(self):
        """
        Displays the analysis that can be applied to the matrix
        """
        close = False
        while not close:
            print("\n--- MENU ---")
            print("Now you can select different operations you can do with your matrix:")
            print("\na) Calculate the probability from going to one state to another in n steps.")
            print("b) Calculate the long-term state (steady) of the matrix.")
            print("c) Identify if the matrix is regular or not.")
            print("d) Choose another matrix")
            print("e) Exit the program")
            choice = input("Type the letter of your choice:")
            if choice == 'a':
                self._state_transition()
            elif choice == 'b':
                self._steady_state()
            elif choice == 'c':
                self._is_regular()
            elif choice == 'd':
                close = True
            elif choice == 'e':
                close = True
                print("\nGoodbye!")
                exit()
            else:
                print("Invalid option, choose an option")
                input("Press ENTER to continue")
    
    def _generate_random_stochastic_matrix(self):
        """
        Generates a random 3 x 3 stochastic matrix
        """
        matrix = np.random.rand(3,3)
        self.matrix = matrix / matrix.sum(axis=1)[:,None]

    def _state_transition(self):
        """
        Calculate the probability from going to one state to another in n steps.
        The program shows every power of the matrix until it reaches the number of steps.
        """
        print("\n**State transition probability**")

        origin = input("Type the number of the origin state. (states start at 0): ")
        origin = int(origin)

        destiny = input("Type the number of the destiny state: ")
        destiny = int(destiny)

        steps = input("Type the steps/generations. (e.j. grandparent - son = 2): ")
        steps = int(steps)

        print("Calculating...")

        power_matrix = self.matrix
        i = 0
        print(f'\nMatrix at step {i + 1} is:')
        print(power_matrix)
        i += 1
        while i < steps:
            power_matrix = np.matmul(power_matrix, self.matrix)
            # Trunc each number in the matrix
            myfunc_vec = np.vectorize(self._trunc)
            power_matrix = myfunc_vec(power_matrix)
            print(f'\nMatrix at step {i + 1} is:')
            print(power_matrix)
            i += 1
        print(f'The probability of going from state {origin} to state {destiny} in {steps} generations is:')
        try:
            print(power_matrix[origin][destiny])
        except:
            print("Something went wrong... maybe you gave an unexisting state number, remember states start at 0")
        input("Press ENTER to continue")
        
    def _steady_state(self, steps = 100):
        """
        Calculate the long-term state (steady) of the matrix.
        Indicates the minimum needed value for the power
        If there is no steady state the user is notified
        """
        print("\n**Steady state**")
        print("Calculating...")
        power_matrix = self.matrix
        i = 0
        myfunc_vec = np.vectorize(self._round_steady)
        power_matrix = myfunc_vec(power_matrix)
        print(f'\nMatrix at step {i + 1} is:')
        print(power_matrix)
        i += 1
        past_matrix = power_matrix
        steady = False
        while i < steps and not steady:
            power_matrix = power_matrix.astype(np.float64)
            power_matrix = np.matmul(power_matrix, self.matrix)

            power_matrix = myfunc_vec(power_matrix)
            print(f'\nMatrix at step {i + 1} is:')
            print(power_matrix)

            str_float_vect = np.vectorize(self._precise_float_str)
            precise_current = str_float_vect(power_matrix)
            precise_past = str_float_vect(past_matrix)

            comparison = precise_current == precise_past
            if comparison.all():
                steady = True
                first_row =  precise_current[0]
                j = 1
                while j < len(precise_current):
                    row_comparison = first_row ==  precise_current[j]
                    if not row_comparison.all():
                        steady = False
                    j += 1
                if steady:
                    print("\n*Attention: The matrix reached a steady state")
                    print(f"This state was reached in step: {i + 1}")
                    print("The steady vector is:")
                    print(power_matrix[0])
            past_matrix = power_matrix
            i += 1
        if not steady:
            print(f"\n*Attention: The matrix never reached a steady state before {steps} steps")
        input("Press ENTER to continue")
    
    def _get_negative_idxs(self, matrix):
        """
        Returns in a list the x,y position in the matrix of the elements that
        are negatives/zeroes
        """
        n = len(matrix)
        zeroes_idxs = list()
        for i in range(n):
            for j in range(n):
                if matrix[i][j] <= 0:
                   zeroes_idxs.append([i,j])
        return np.array(zeroes_idxs)

    def _is_regular(self, steps = 100):
        """
        Identifies if the matrix is regular or not by:
        Checking if negatives/zeroes positions don't change on the next generation
        or if the max number of steps is reached and the matrix still has
        negatives/zeroes
        """
        print("\n**Is your matrix regular?**")
        print("Calculating...")

        power_matrix = self.matrix

        i = 0
        myfunc_vec = np.vectorize(self._round_steady)
        power_matrix = myfunc_vec(power_matrix)

        print(f'\nMatrix at step {i + 1} is:')
        print(power_matrix)

        i += 1
        past_matrix = power_matrix
        past_negative_idxs = self._get_negative_idxs(power_matrix)

        while i < steps:
            power_matrix = power_matrix.astype(np.float64)
            power_matrix = np.matmul(power_matrix, self.matrix)

            power_matrix = myfunc_vec(power_matrix)
            print(f'\nMatrix at step {i + 1} is:')
            print(power_matrix)

            curr_negative_idxs = self._get_negative_idxs(power_matrix)
            row_comparison = past_negative_idxs == curr_negative_idxs

            if len(curr_negative_idxs) == 0:
                print("\n*Attention: The matrix is Regular!")
                print(f"\n*After {i + 1} steps the matrix was found to be Regular!")
                input("Press ENTER to continue")
                return
            elif type(row_comparison) != bool:
                if row_comparison.all():
                    print("\n*Attention: The matrix is not regular")
                    print(f"\n*After {i + 1} steps the matrix was found to be not regular")
                    print("\n*The negative numbers/zeroes positions didn't change in the following iteration")
                    input("Press ENTER to continue")
                    return

            past_negative_idxs = curr_negative_idxs
            past_matrix = power_matrix
            i += 1
        curr_negative_idxs = self._get_negative_idxs(power_matrix)
        if len(curr_negative_idxs) > 0:
            print("\n*Attention: The matrix is not regular")
            print(f"\n*After {steps} steps the matrix stil has negatives/zeroes")
        input("Press ENTER to continue")
        
    def _load_matrix(self):
        """
        Helper function that loads the user given matrix in the matrix.txt file
        """
        print("\nInput a matrix on the matrix.txt file")
        print("\nEach number should be separated by a comma and each row should be in a new line")
        print("\nFor example:")
        print("0.5, 0.5, 0")
        print("0, 0.5, 0.5")
        print("0.5, 0, 0.5")
        input("\nPress enter when you have done this ...")
        print("\nNow reading your matrix ...")
        try:
            self.matrix = np.loadtxt("matrix.txt", dtype='f', delimiter=',')
        except:
            print("An error courred by reading your matrix, make sure to use\n decimals, the correct comma delimiter and that your matrix is of size n x n")
            print("\nNow exiting program ...")
            exit()
        
    def _check_matrix(self):
        """
        Helper function
        Checks if the matrix is stochastic and has a size of n x n
        """
        n = self.matrix.shape[0]
        m = self.matrix.shape[1]
        if n != m:
            print("Your matrix is not a square matrix. Matrix must of size n x n")
            print("\nNow exiting program ...")
            exit()

        # Trunc each number in the matrix
        myfunc_vec = np.vectorize(self._trunc)
        self.matrix = myfunc_vec(self.matrix)
        print(self.matrix)

        # Get sums of each row, must be 1 for each row
        sums = self.matrix.sum(axis=1)
        sums = myfunc_vec(sums)

        # Round numbers near 1 like 0.999999
        myfunc_vec = np.vectorize(self._round)
        sums = myfunc_vec(sums)

        for num in sums:
            if num != 1:
                print("Your matrix is not stochastic. Each row should add up to 1")
                print("\nNow exiting program ...")
                exit()
        
    def _trunc(self, x):
        """
        Helper function
        Truncs decimals in matrix to only have 6 decimals
        """
        # Trunc to 6 decimals
        x = np.around(x, decimals=6)
        x = math.floor(np.floor(x * 10**6)) / 10 ** 6
        return x

    def _round(self, x):
        """
        Helper function
        Rounds decimals in matrix to only have 6 decimals
        """
        # Round number near 1
        if x > 0.999991 and x < 1.000009:
            return 1
        return x

    def _round_steady(self, x):
        """
        Helper function
        Rounds decimals in matrix to only have 6 decimals
        Better precision
        """
        x = np.around(x, decimals=6)
        x = math.floor(np.floor(x * 10**6)) / 10 ** 6
        if x > 0.999991 and x < 1.000009:
            return 1
        return x

    def _precise_float_str(self, x):
        """
        Helper function
        Converts float to str
        """
        x = format(x, '.5f')
        return x

# Runds the program
mc = MarkovChain()