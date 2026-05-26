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

for element in normd:
    if element[0]<0.:
        element[0]=max(normd)[0]
print(normd)

maxi=max(normd).copy()
maxiarg=np.argmax(normd).copy()

modnor=[]
modnor=normd.copy()
selection=10

print('\nMinima and their arguments')
for i in range(0,selection):
    globals()['mini' + str(i+1)] = min(modnor).copy()
    globals()['miniarg' + str(i+1)] = np.argmin(modnor).copy()
    globals()['miniarg' + str(i+1)] = globals()['miniarg' + str(i+1)].astype(int).copy()
    modnor[globals()['miniarg' + str(i+1)]] = [1000.]
    print(globals()['mini' + str(i+1)], globals()['miniarg' + str(i+1)])

print('\nMaximum and its argument')
print(max(normd), np.argmax(normd))

#np.savetxt(str(nradars) + 'rad_' + str(nposit) + 'pos_cad' + str(ran_cadence) + '_ele' + str(elevationangle) + '_allpar_formal.out', [])

radcoor1=[]
for i in range(len(normd)):
    radcoor1=normd[i].copy()
    for element in Combis.Combislist(nposit,nradars)[i]:
        for elt in element:
            radcoor1.append(elt)
    with open(str(nradars) + 'rad_' + str(nposit) + 'pos_cad' + str(ran_cadence) + '_ele' + str(elevationangle) + '_allpar_formal.out', "ab") as f:
        np.savetxt(f, [radcoor1])

with open(str(nradars) + 'rad_' + str(nposit) + 'pos_cad' + str(ran_cadence) + '_ele' + str(elevationangle) + '_allpar_formal.out', "ab") as f:
    f.write(b"\n")
    for i in range(0, selection):
        radcoor2 = globals()['mini' + str(i + 1)].copy() #radcoor1=normd[i].copy()
        for element in Combis.Combislist(nposit, nradars)[globals()['miniarg' + str(i+1)]]:
            for elt in element:
                radcoor2.append(elt)
        np.savetxt(f, np.column_stack(radcoor2))
    maxx = max(normd).copy()
    maxxarg = np.argmax(normd).copy()
    for element in Combis.Combislist(nposit, nradars)[maxxarg]:
        for elt in element:
            maxx.append(elt)
    np.savetxt(f, np.column_stack(maxx))

#print('\nAll radar coordinate combinations (Combis.Combislist(nposit,nradars))')
#print(Combis.Combislist(nposit,nradars))
print('\nMinima radar coordinate combinations')
print(Combis.Combislist(nposit,nradars)[miniarg1])
print(Combis.Combislist(nposit,nradars)[miniarg2])
print(Combis.Combislist(nposit,nradars)[miniarg3])
print(Combis.Combislist(nposit,nradars)[miniarg4])
print(Combis.Combislist(nposit,nradars)[miniarg5])
print(Combis.Combislist(nposit,nradars)[miniarg6])
print(Combis.Combislist(nposit,nradars)[miniarg7])
print(Combis.Combislist(nposit,nradars)[miniarg8])
print(Combis.Combislist(nposit,nradars)[miniarg9])
print(Combis.Combislist(nposit,nradars)[miniarg10])

gtdata = np.loadtxt('K2_ground_track.out', delimiter=',')
xx = gtdata[:,0].copy()
yy = gtdata[:,1].copy()

radcoor=[]
for element in Combis.Coordiarray(Combis.Combislist(nposit,nradars)):#Combis.Combislist(nposit,nradars):
    for elt in element:
        #print('element', elt)
        radcoor.append(elt)
        #for e in elt:
        #    radcoor.append(e)

if nradars == 1:
    zz = radcoor[0:nposit].copy()
    ww = radcoor[nposit:2*nposit].copy()
    print(zz)
    print(ww)
else:
    radcoor3 = []
    cc = []
    print(Combis.Combislist(nposit, nradars)[globals()['miniarg' + str(1)]])
    for element in Combis.Combislist(nposit, nradars)[globals()['miniarg' + str(1)]]:
        for elt in element:
            radcoor3.append(elt)
    print(radcoor3)
    zzz = radcoor3[0::2].copy()
    www = radcoor3[1::2].copy()

# coordinates of the sample positions nposit
longi = Combis.fibonacci_sphere(nposit)[0::2].copy()
latit = Combis.fibonacci_sphere(nposit)[1::2].copy()

minimumofnorm=np.array(min(normd))

for i in min(obspr):
    minobspr = float(i)

plt.figure(figsize=(9, 5), dpi=400)
plt.title("Ground track of Orbiter")
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xticks(np.arange(-150, 200, step=50))
plt.xlim([min(xx), max(xx)])
plt.yticks(np.arange(-80, 100, step=20))
plt.ylim([-90, 90])
plt.grid()
hex_list = ['#d62728', 'fde725', '#a0da39', '#4ac16d', '#1fa187', '#277f8e', '#365c8d', '#46327e', '#440154']
if nradars == 1:
    print('nradars=1')
    sc1 = plt.scatter(longi, latit, marker='.', s=10, c='blue', label="")
    sc = plt.scatter(zz, ww, marker='.', s=60, c=normd, cmap=Combis.get_continuous_cmap(hex_list, float_list=[0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.95, 1]))#Combis.get_continuous_cmap(hex_list, float_list=[0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 1]))#Combis.get_continuous_cmap(hex_list, float_list=[0, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1]))#
    plt.colorbar(sc)
else:
    sc1 = plt.scatter(longi, latit, marker='.', s=60, c='blue', label="")
    sc = plt.scatter(zzz, www, marker='.', s=60, c='red', label="")#Combis.get_continuous_cmap(hex_list, float_list=[0, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1]))
textstr = '\n'.join((
    r'Number of radar transponders = $%d$' % (int(nradars), ),
    r'Number of sample positions = $%d$' % (int(nposit), ),
    r'Norm minimum [m] = $%.5f$' % (minimumofnorm[0], ),))
props = dict(boxstyle='square', facecolor='white', edgecolor='white', alpha=0.)
plt.text(-177.0, 87.0, textstr, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=props)
textstr = '\n'.join((
    r'Range cadence [s] = $%.1f$' % (ran_cadence, ),
    r'Elevation angle [deg] = $%d$' % (elevationangle, ),))
props = dict(boxstyle='square', facecolor='white', edgecolor='white', alpha=0.)
plt.text(70.0, 87.0, textstr, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=props)
plt.scatter(xx, yy, marker='.', s=0.05, c='black')
plt.tight_layout()
plt.savefig(str(nradars) + 'rad_' + str(nposit) + 'pos_cad' + str(ran_cadence) + '_ele' + str(elevationangle) + '_allpar_formal.png')
plt.show()

"""plt.figure(figsize=(9, 5), dpi=400)
plt.title("Ground track of Orbiter")
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xticks(np.arange(-150, 200, step=50))
plt.xlim([min(xx), max(xx)])
plt.yticks(np.arange(-80, 100, step=20))
plt.ylim([-90, 90])
plt.grid()
hex_list = ['#d62728', 'fde725', '#a0da39', '#4ac16d', '#1fa187', '#277f8e', '#365c8d', '#46327e', '#440154']
if nradars == 1:
    print('nradars=1')
    sc1 = plt.scatter(longi, latit, marker='.', s=10, c='blue', label="")
    sc = plt.scatter(zz, ww, marker='.', s=60, c=obspr, cmap=Combis.get_continuous_cmap(hex_list, float_list=[0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.95, 1]))#Combis.get_continuous_cmap(hex_list, float_list=[0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 1]))#Combis.get_continuous_cmap(hex_list, float_list=[0, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1]))#
    plt.colorbar(sc)
#else:
#    sc1 = plt.scatter(longi, latit, marker='.', s=60, c='blue', label="")
#    sc = plt.scatter(zzz, www, marker='.', s=60, c=minobspr, cmap=Combis.get_continuous_cmap(hex_list, float_list=[0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.95, 1]))#Combis.get_continuous_cmap(hex_list, float_list=[0, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1]))
#textstr = '\n'.join((
#    r'Number of radar transponders = $%d$' % (int(nradars), ),
#    r'Number of sample positions = $%d$' % (int(nposit), ),
#    r'Norm minimum [m] = $%.5f$' % (minimumofnorm[0], ),))
props = dict(boxstyle='square', facecolor='white', edgecolor='white', alpha=0.)
plt.text(-177.0, 87.0, textstr, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=props)
textstr = '\n'.join((
    r'Range cadence [s] = $%d$' % (ran_cadence, ),
    r'Elevation angle [deg] = $%d$' % (elevationangle, ),))
props = dict(boxstyle='square', facecolor='white', edgecolor='white', alpha=0.)
plt.text(70.0, 87.0, textstr, fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=props)
plt.scatter(xx, yy, marker='.', s=0.05, c='black')
plt.tight_layout()
plt.savefig(str(nradars) + 'rad_' + str(nposit) + 'pos_cad' + str(ran_cadence) + '_ele' + str(elevationangle) + '_allpar_formal.png')
plt.show()"""

"""tuples=[]
tuples=[Combis.Combislist(nposit,nradars)[miniarg1], Combis.Combislist(nposit,nradars)[miniarg2], Combis.Combislist(nposit,nradars)[miniarg3], Combis.Combislist(nposit,nradars)[miniarg4], Combis.Combislist(nposit,nradars)[miniarg5], Combis.Combislist(nposit,nradars)[miniarg6], Combis.Combislist(nposit,nradars)[miniarg7], Combis.Combislist(nposit,nradars)[miniarg8], Combis.Combislist(nposit,nradars)[miniarg9], Combis.Combislist(nposit,nradars)[miniarg10]]

mincoor=[]
for element in tuples:
    for e in element:
        mincoor.append(e)

print('mincoor =',mincoor)

nradars=2*nradars
nposit=selection
print("\nNumber of all possible combinations of", nradars, "radars in", nposit, "positions:", len(Combis.Combislist2(nposit,nradars,mincoor)))
print(Combis.Combislist2(nposit,nradars,mincoor), len(Combis.Combislist2(nposit,nradars,mincoor)))

l=0
while l < len(Combis.Combislist2(nposit,nradars,mincoor)):
    print('\nl =',l)
    FullEstimation.estim(l,Combis.Coordiarray(Combis.Combislist2(nposit,nradars,mincoor)),nradars)
    print('\nNorm difference', normd)
    print(min(normd))
    l += 1
"""