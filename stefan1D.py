import numpy as np
import scipy as sp
from scipy.optimize import fsolve

class Stefan1D :
    """
    Analytical Solution to 1D Stefan problem for melting case
    x* (x = 0, t = 0) = 0 -> initial position of the interface
    Boundary condition,
    T(x=0,t) = Tliquid0
    T(x=L,t) = Tsolid0
    Initial Condition
    T(x,t=0) = Tsolid0, x> 0

    """
    def __init__(self, Params):

        """
        define stefan 1D stefan problem with  parameters as value to the following
        dictionary key
        {
        "TS0" : , -> Initial temperature of solid
        "TL0" : , -> Initial temperature of liquid
        "Tmelt" :, -> Melting temperature
        "Lheat" : , -> Latent heat of phase change
        "lmdaS" : , -> Thermal conductivity of Solid
        "lmdaL" : , -> Thermal conductivity of Liquid
        "rhoS" : , -> Denisty of Solid
        "rhoL : , -> Denisty of Liquid
        "cpS" : , -> Heat capacity of Solid
        "cpL": , -> Heat capacity of Liquid
        }

        """
        # Parse Parameters
        self.TS0 = Params["TS0"]
        self.TL0 = Params["TL0"]
        self.Tmelt = Params["Tmelt"]
        self.Lheat = Params["Lheat"]
        self.lamdaS = Params["lamdaS"]
        self.lamdaL = Params["lamdaL"]
        self.rhoS =  Params["rhoS"]
        self.rhoL =  Params["rhoL"]
        self.cpS = Params["cpS"]
        self.cpL = Params["cpL"]

        # Calculate Constants
        self.alphaS = self.lamdaS/(self.cpS*self.rhoS)
        self.alphaL = self.lamdaL/(self.cpL*self.rhoL)
        self.Ste =  self.cpL*(self.TL0 -self.Tmelt)/self.Lheat

        phiFunc = lambda phi : (self.Ste/np.sqrt(np.pi)) - sp.special.erf(phi)*(phi*np.exp(phi**2) -
        (self.cpS*(self.Tmelt- self.TS0)*np.exp((phi**2)*(1 - self.alphaL/self.alphaS))*np.sqrt(self.lamdaS*self.rhoS*self.cpS))/
        (self.Lheat*np.sqrt(np.pi)*sp.special.erfc(phi*np.sqrt(self.alphaL/self.alphaS))*np.sqrt(self.lamdaL*self.rhoL*self.cpL)))

        self.phi = fsolve(phiFunc, 1)


    def getSte( self) -> np.float64 :
        """
        Get Calculated Stefan Number(Ste)
        """
        return self.Ste[0]

    def xStar(self, t : np.ndarray) -> np.ndarray:
        """
        Calculate interface position for given discretized time
        """
        return 2*self.phi*np.sqrt(self.alphaL*t)

    def T(self, x : np.ndarray, t : np.ndarray) -> np.ndarray:
        """
        Calculate temperature for given discretized domain and  time
        """
        T = np.zeros((x.shape[0],t.shape[0]), dtype = "float64")
        xstar = self.xStar(t)
        for i,ti in enumerate(t):
            for j,xi in enumerate(x):
                if (xi <= xstar[i]):
                     temp  = self.TL0 + (self.Tmelt - self.TL0)*sp.special.erf(xi/(2*np.sqrt(self.alphaL*ti)))/sp.special.erf(self.phi)
                     T[j,i] = temp[0]

                else:
                     temp = self.TS0 + (self.Tmelt - self.TS0)*sp.special.erfc(xi/(2*np.sqrt(self.alphaS*ti)))/sp.special.erfc(self.phi*np.sqrt(self.alphaL/self.alphaS))
                     T[j,i] = temp[0]
        return T

#---------------------------------------------USAGE------------------------------------#
if __name__ == "__main__":
    # or by giving parameters
    water_params = { 
              "TS0" : 272.5 ,
              "TL0" :  300 ,
              "Tmelt" : 273 ,
              "Lheat":336e3,
              "lamdaS" : 2.22 ,
              "lamdaL" : 0.54 ,
              "rhoS": 0.0009,
              "rhoL": 0.001,
              "cpS": 2090,
              "cpL": 4200
                 }
    ice_melt = Stefan1D(water_params)
    # Calculate Melt interface position at first 100s
    print("interface position = ", ice_melt.xStar(np.linspace(0,100,10)))
    # Calculate Temperature at given location at given time
    x = np.array([0.5])
    t = np.array([10, 20])
    print("Temperature =\n ", ice_melt.T(x, t))

