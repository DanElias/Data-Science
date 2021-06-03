import random
import math

class PIApproximation:
    def __init__(self):
        self.n = input("Give N: ")
        self.n = int(self.n)
        self.inside = 0
        self.outside = 0
        self.pi = 0
        self.approximate()

    def approximate(self):
        for count in range(self.n):
            x = random.random()
            y = random.random()
            r = math.sqrt(math.pow(x,2) + math.pow(y,2))
            if r < 1:
                self.inside+=1
            else:
                self.outside+=1
        self.pi = 4 * (self.inside / self.n)
        print("PI Approximation is: " + str(self.pi))
    


piapx = PIApproximation()
