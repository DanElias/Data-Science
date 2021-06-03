"""
Author: Daniel Elias
Date: June 2021
Simulation of a battle between antagonist groups using Markov Chains and the Monte Carlo method.

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
import random
from numpy.linalg import matrix_power
warnings.filterwarnings('ignore')

class WarSimulation:
    def __init__(self):
        """
        Constructor
        """
        self.matrix = []
        self.number_of_factions = 0
        self.factions = []
        self.total_warriors = 0
        self.intervals = []
        self.factions_positions = dict()
        self.result = ""
        
        self._run()
    
    def _run(self):
        """
        Displays the main menu and asks user to select from:
        a) Inputing their on matrix or
        b) Generating a random 3 x 3 matrix
        """
        close = False
        while not close:
            print("\nLet's begin our war simulation... Do you want to:")
            print("a) Input your own matrix")
            print("b) Use a random n x n matrix")
            print("c) Exit the program")
            choice = input("Type the letter of your choice:")
            if choice == 'a':
                print("Remember your matrix should be squared and the rows/cols equal the number of factions")
                self._load_matrix()
                print("Now checking your matrix ...")
                self._check_matrix()
                input("\nPress ENTER to continue")
                self._menu()
            elif choice == 'b':
                n = input("Give the width of the matrix which is the number of factions:  ")
                print("\nNow creating a n x n Random Stochastic Matrix")
                self._generate_random_stochastic_matrix(int(n))
                print("Checking the matrix ...")
                self._check_matrix()
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
        self._initialize_factions()
        self._initialize_intervals()
        while not close:
            print("\n--- MENU ---")
            print("Now we can start the war simulation!")
            print("a) Unleash the war!")
            print("b) Choose another matrix")
            print("c) Exit the program")
            choice = input("Type the letter of your choice:")
            if choice == 'a':
                self._war()
                with open ("output.txt", "w") as output:
                    output.write(self.result)
                self.result = ""
                self.factions_positions = dict()
                input("Press ENTER to continue")
                close = True
            elif choice == 'b':
                close = True
            elif choice == 'c':
                close = True
                print("\nGoodbye!")
                exit()
            else:
                print("Invalid option, choose an option")
                input("Press ENTER to continue")

    def _print_factions(self):
        for i in range(len(self.factions_positions)):
            if -1 == self.factions_positions[i]:
                print("Faction " + str(i) + " has 0 warriors")
                self.result += "\nFaction " + str(i) + " has 0 warriors"
            else:
                print("Faction " + str(i) + " has " + str(self.factions[self.factions_positions[i]]) + " warriors")
                self.result += "\nFaction " + str(i) + " has " + str(self.factions[self.factions_positions[i]]) + " warriors"

    def _war(self):
        print(self.matrix)
        self.result += "\n" + str(self.matrix)
        self._print_factions()
        while self.number_of_factions > 1:
            attacker = random.randint(0, self.number_of_factions - 1)
            attacked = 0
            attacker_intervals = self.intervals[attacker]
            random_probability = random.random()
            for index, row in attacker_intervals.iterrows():

                # if the probability is in the interval (monte carlo)
                if random_probability >= row["inferior"] and random_probability < row["superior"]:
                    attacked = index

                    if self.factions[attacked] > 0:

                        # perform the attack
                        self.factions[attacked] -= 1
                        for i in range(len(self.factions_positions)):
                            if attacked == self.factions_positions[i]:
                                print("Faction " + str(i) + " was attacked")
                                self.result += "\nFaction " + str(i) + " was attacked"
                            if attacker == self.factions_positions[i]:
                                print("Faction " + str(i) + " attacked")
                                self.result += "\nFaction " + str(i) + " attacked"
                        # print the remaining warriors for each faction
                        self._print_factions()
                        
                        # Check if attacked faction died
                        if self.factions[attacked] == 0:

                            #reconfigure the markov chain and the intervals (monte carlo)
                            print("*A faction was defeated*")
                            self.result += "\n*A faction was defeated*"
                            
                            # reduce number of factions
                            self.number_of_factions -= 1

                            for i in range(len(self.factions_positions)):
                                # Remove attacked from factions list
                                if attacked == self.factions_positions[i]:
                                    print("Faction " + str(i) + " was defeated!!!")
                                    self.result += "\nFaction " + str(i) + " was defeated!!!"
                                    self.factions.pop(attacked)
                                    self.factions_positions[i] = -1
                                # move the other remaining factions one space in the factions list
                                if attacked < self.factions_positions[i]:
                                    self.factions_positions[i] -= 1
                            
                            # See if only one faction is remaining
                            if self.number_of_factions == 1:
                                # Print who was the winner
                                for i in range(len(self.factions_positions)):
                                    if 0 == self.factions_positions[i]:
                                        print("Faction " + str(i) + " has " + str(self.factions[0]) + " remaining warriors and is the WINNER!!!")
                                        self.result += "\nFaction " + str(i) + " has " + str(self.factions[0]) + " remaining warriors and is the WINNER!!!"
                                        i = len(self.factions_positions)
                                print("The war has ended... Peace & Love!")
                                self.result += "\n\nThe war has ended... Peace & Love!"
                                # reconfigure stochastic matrix rows/cols to the number of remaining factions
                            else:
                                print("*Reconfiguring the war...*")
                                self.result += "\n\n*Reconfiguring the war...*"
                                self._reconfigure()
                                self.result += "\n" + str(self.matrix)
                                
                        
                            
            
    def _reconfigure(self):
        print("Reconfiguring the matrix...")
        self._generate_random_stochastic_matrix(self.number_of_factions)
        print(self.matrix)
        print("Checking the matrix...")
        self._check_matrix()
        self._initialize_intervals()
        print("Continuing with the war...")
    
    def _generate_random_stochastic_matrix(self, n):
        """
        Generates a random n x n stochastic matrix
        """
        matrix = np.random.rand(n,n)
        self.matrix = matrix / matrix.sum(axis=1)[:,None]
        # Set diagonal to zeroes and add the diagonal num to rest of columns in row to keep stochastic
        for i in range(len(self.matrix)):
            divided_diagonal = self.matrix[i][i] / (len(self.matrix) - 1)
            for j in range(len(self.matrix)):
                if i == j:
                    self.matrix[i][j] = 0
                else:
                    self.matrix[i][j] += divided_diagonal


    def _initialize_factions(self):
        self.factions = []
        close = False
        while not close:
            print("How do you want to set the number of individuals per faction?")
            print("a) Input the numbers per faction")
            print("b) Randomly generate the numbers per faction")
            choice = input("Type the letter of your choice:")
            if choice == 'a':
                self._input_faction_numbers()
                print("This are the number of individuals per faction:")
                print(self.factions)
                input("Press ENTER to continue")
                close = True
            elif choice == 'b':
                self._random_faction_numbers()
                print("This are the number of individuals per faction:")
                print(self.factions)
                input("Press ENTER to continue")
                close = True
            else:
                print("Invalid option, choose an option")
                input("Press ENTER to continue")

    def _initialize_intervals(self):
        self.intervals = []
        columns = ['index','inferior','superior']
        index = np.arange(0,self.number_of_factions)
        for i in range(self.number_of_factions):
            inferior = []
            superior = []
            aux = 0
            for j in range(self.number_of_factions):
                inferior.append(aux)
                if j + 1 < self.number_of_factions:
                    aux = aux + self.matrix[i][j]
                    superior.append(aux)
                else:
                    superior.append(1)
            new_df = pd.DataFrame(index=index, columns=columns)
            new_df["index"] = index
            new_df["inferior"] = inferior
            new_df["superior"] = superior
            self.intervals.append(new_df)

    def _input_faction_numbers(self):
        self.number_of_factions = len(self.matrix)
        for i in range(self.number_of_factions):
            self.factions_positions[i] = i
            number_per_faction = input("Input number of individuals for faction " + str((i + 1)) + ": ")
            self.total_warriors += int(number_per_faction)
            self.factions.append(number_per_faction)

    def _random_faction_numbers(self):
        self.number_of_factions = len(self.matrix)
        for i in range(self.number_of_factions):
            self.factions_positions[i] = i
            number_per_faction = random.randint(10, 100)
            self.factions.append(number_per_faction)
            self.total_warriors += number_per_faction
            

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
        print("Great! Your matrix is Stochastic ...")
        
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
mc = WarSimulation()