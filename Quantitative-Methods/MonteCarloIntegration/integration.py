import random
import math

class PIApproximation:
    def __init__(self):
        self.n = input("Give N: ")
        self.n = int(self.n)
        self.inside = 0
        self.outside = 0
        self.integral = 0
        self.dx = 0.1
        self.a = 2
        self.b = 11
        self.maxy = 0
        self.approximate()

    def function(self, x):
        return math.pow(math.e, math.cos(x))

    def approximate(self):
        for count in range(self.n):
            self.maxy = self.find_max()
            x = random.uniform(self.a, self.b)
            y = random.uniform(0, self.maxy)
            yr = self.function(x)
            if y < yr:
                self.inside += 1
            else:
                self.outside += 1
        self.integral = ((self.b - self.a) * self.maxy)  * (self.inside / self.n)
        print("Integral is: " + str(self.integral))

    def find_max(self):
        i = self.a
        maxyr = self.maxy;
        while i < self.b:
            y = self.function(i)
            if y > self.maxy:
                maxyr = i
            i += self.dx
        return maxyr

    


piapx = PIApproximation()
