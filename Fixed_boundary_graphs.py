import freegs
import numpy as np
import matplotlib.pyplot as plt

#The following function is used to find the O- and X-points in the psi grid, which must be found for this project.
from freegs.critical import find_critical

#Boundary conditions
from freegs import boundary

def equlibriumGSsolution(Rmin, Rmax, Zmin, Zmax, nx, ny, maxPressure):
    eq = freegs.Equilibrium(
        Rmin=Rmin,
        Rmax=Rmax,
        Zmin=Zmin,
        Zmax=Zmax,
        nx=nx,
        ny=ny,
        boundary=boundary.fixedBoundary,
    )

    #We set the conditions for the plasma profiles, which are used to solve the Grad-Shafranov equation.
    #Our variable that is changed, is the maxPressure.

    profiles = freegs.jtor.ConstrainPaxisIp(
        eq,
        maxPressure,  # Plasma pressure on axis [Pascals]
        1e5,  # Plasma current [Amps]
        1.0,
    )  # fvac = R*Bt

    #Nonlinear solver for Grad-Shafranov equation
    freegs.solve(
        eq,  # The equilibrium to adjust
        profiles,  # The toroidal current profile function
        show=True,
    )

    #The r and x values of the O point need to be found, which is done using the find_critical function.
    R_values = eq.R #radial values of our grid(cylindrical coordinates)
    Z_values = eq.Z #z values of our grid(cylindrical coordinates)
    psi_values = eq.psi() #the psi values of our grid, which is the solution to the Grad-Shafranov equation(Poloidal flux).

    opoints, xpoints = find_critical(R_values, Z_values, psi_values, discard_xpoints=True)
    if len(opoints) == 0:
        raise RuntimeError("No O-points found in psi grid")
    r_axis, z_axis, psi_axis = opoints[0]

    R_geo = (Rmax+Rmin)/2   #We also need to determine the polodial beta, which can be retrived from the our solution.
    poloidal_beta = eq.poloidalBeta()

    return (r_axis-R_geo), z_axis, poloidal_beta

pressure_values = np.linspace(1e3, 1e4, 5) #We want to vary the pressure on axis from 1e4 to 1e5 Pascals,
#which is done using the linspace function.

#We define the values required for our profiles in the GD solver.
R_min = 0.1
R_max = 1.0
Z_min = -0.5
Z_max = 0.5
nx = 65
ny = 65


opointRvalues = []
Betavalues = []


#We loop through different max pressure values, that as a result changes the poloidal beta
for pressure in pressure_values:
    R,Z,Beta = equlibriumGSsolution(R_min, R_max, Z_min, Z_max, nx, ny, pressure)

    #Append retrieved values to lists of \deltaR and \beta values.
    opointRvalues.append(R)
    Betavalues.append(Beta)

    print(f"Pressure on axis: {pressure} Pa, Poloidal beta: {Beta}")


#We plot the values of \deltaR and \beta.
plt.plot(opointRvalues, Betavalues, marker='o')
plt.xlabel(r'Poloidal beta/ $\beta $')
plt.ylabel(r' $\Delta$R/m')
plt.title(r'$\Delta$R as function of Poloidal beta/ $\beta $')
plt.show()








