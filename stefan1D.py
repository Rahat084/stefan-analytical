import numpy as np
import scipy as sp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

class Stefan1D :
    """
    Analytical Solution to 1D Stefan problem
    x* (x = 0, t = 0) = 0 -> initial position of the interface
    """
    def __init__(self, value = None):
       self.ste_ = value

    @classmethod
    def fromDict(cls, **kwargs):
        """
        Calculate stefan number from given parameters
        """
        Tmax = kwargs["Tmax"]
        Tmin = kwargs["Tmin"]
        Tsol = kwargs["Tsol"]
        Tliq = kwargs["Tliq"]
        Cps = kwargs["Cps"]
        Cpl = kwargs["Cpl"]
        Lheat = kwargs["Lheat"]
        ste = (Cps*(Tsol - Tmin) + Cpl*(Tmax - Tliq)) / Lheat # stefan number
        self.ste_ = ste
        return cls(value = ste)

    def lmdCalc_(self):
        """
        Solve for lambda value
        """
        lfunc = lambda lmbda : self.ste_*lmbda - 1/np.sqrt(np.pi) * (np.exp(-lmbda**2))/sp.special.erf(lmbda)
        self.lmbda_ = fsolve(lfunc, 1)

    def xStar(self, t):
        """
        Calculate interface position for given time
        """
        if self.ste_:
            self.lmdCalc_()
            return 2*self.lmbda_*np.sqrt(t)
        else:
            print("Define stefan number")

    def T(self, x, t):
        """
        Calculate temperature for given position and  time
        """
        if self.ste_:
            self.lmdCalc_()
            return 1 - sp.special.erf(x/(2*np.sqrt(t)))/sp.special.erf(self.lmbda_)
        else:
            print("Define stefan number")


