import os
# Load required standard modules
import numpy as np
import math
import pyproj
import itertools
import matplotlib
from matplotlib import colormaps
from matplotlib import pyplot as plt
from combis import Combis
from combis import normd
from combis import obspr
from full_estimation import FullEstimation
from tudatpy.interface import spice

## Define a set of radar positions for grid search
nposit = 5 # number of positions
nradars = 1 # number of radars per iteration
ran_cadence = 1.0 #0.1 #1.0 #10
elevationangle = 20.

# Load spice kernels
spice.load_standard_kernels()
kernels = ['/home/neumwl/TudatProjects/de438.bsp', '/home/neumwl/TudatProjects/sat427.bsp',
           '/home/neumwl/TudatProjects/enceladus_ssd_230702_v1.tpc']
#                      '/home/neumwl/TudatProjects/pck00010.tpc']
spice.load_standard_kernels(kernels)

print(os.environ['CONDA_DEFAULT_ENV'])

## Print the number of all combinations of radars over given positions
print("\nNumber of all possible combinations of", nradars, "radars in", nposit, "positions:", len(Combis.Combislist(nposit,nradars)))

        #listco1 = [90.0, 141.93951659864038, 0.0, -43.49818395038546, -90.0]
        #listco2 = [0.0, -35.80240753106147, 84.98447189992432, -43.41505576764974, 0.0]
        #r = {radar_names[0]: [0.0, np.deg2rad(-90.), np.deg2rad(0.)]}
        #r = {radar_names[0]: [0.0, np.deg2rad(listco1[l]),np.deg2rad(listco2[l])]}

lenloop=Combis.Combislist(nposit,nradars)
r=Combis.Coordiarray(Combis.Combislist(nposit,nradars))

l=0
while l < len(lenloop):
    print('\nl =',l)
    if nradars == 1:
        print('r', r[:,l])
        FullEstimation.estim(l,r[:,l],nradars,ran_cadence,elevationangle)
        print('normd minimum',min(normd))
        print('normd maximum',max(normd))
    else:
        print(len(r))
        print('r',r)
        rr = np.array(r)
        print('rr',rr[:,l*nradars:(l+1)*nradars])
        FullEstimation.estim(l, rr[:,l*nradars:(l+1)*nradars], nradars, ran_cadence, elevationangle)
        print('normd minimum',min(normd))
        print('normd maximum',max(normd))
    l += 1

"""l=0
while l < 1:#len(Combis.Combislist(nposit,nradars)):
    print('\nl =',l)
    FullEstimation.estim(l,Combis.Coordiarray(Combis.Combislist(nposit,nradars)),nradars,ran_cadence,elevationangle)
    print('normd minimum',min(normd))
    print('normd maximum',max(normd))
    l += 1"""

print(normd)
