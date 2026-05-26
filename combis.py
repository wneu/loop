# Load required standard modules
import numpy as np
import math
import pyproj
import itertools
import matplotlib.colors as mcolors

normd = []
obspr = []

class Combis:

    # Function for converting cartesian coordinates to geodetic ones
    def cartesian_to_geodetic(x, y, z): # not used
        enc_rad = 256600.0
        alt = enc_rad
        # Define the coordinate systems
        ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
        # Perform the coordinate transformation
        lon, lat, alt = pyproj.transform(ecef, lla, x, y, z, radians=False)
        return lat, lon, alt

    # Function for defining equidistant radar positions set along a Fibonacci spiral
    def fibonacci_sphere(samples):
        pointsgeo = []
        phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2 # y goes from 1 to -1
            radius = math.sqrt(1 - y * y)  # radius at y
            theta = phi * i  # golden angle increment
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            longi = np.rad2deg(math.atan2(y, x)) #* 360. / 2 * math.pi
            latit = np.rad2deg(math.asin(z / 1.)) #* 360 / 2 * math.pi
            pointsgeo.append(longi)
            pointsgeo.append(latit)
        return pointsgeo

    def Combislist(numberpositions, numberradars):
        # Geodetic coordinates of the radar positions set
        longi=Combis.fibonacci_sphere(numberpositions)[0::2]
        latit=Combis.fibonacci_sphere(numberpositions)[1::2]
        radars=[]
        for i in range(len(longi)):
            radars.append((longi[i],latit[i]))
        radars_combis = itertools.combinations(radars, numberradars)
        ## Convert the iterator to a list to display all combinations
        combis_list = list(radars_combis)
        return combis_list

    def Combislist2(numberpositions, numberradars, minco):
        # Geodetic coordinates of the radar positions set
        #print(Combis.Combislist(nposit, nradars)[miniarg1])
        #longi=Combis.fibonacci_sphere(numberpositions)[0::2]
        #latit=Combis.fibonacci_sphere(numberpositions)[1::2]
        radars=minco
        radars_combis = itertools.combinations(radars, numberradars)
        ## Convert the iterator to a list to display all combinations
        combis_list = list(radars_combis)
        return combis_list

    def Coordiarray(combinationlist):
        #helpcoordiarray=combinationlist[0]
        coordiarray=[0,0] #helpcoordiarray[0]
        for m in range(len(combinationlist)):
           #print(f'\n{m}. ')
           #print(combinationlist[m])
           coordi = combinationlist[m]
           #coordiarray = coordi[0]
           #print(coordiarray)
           for n in range(0,len(coordi)):
              if m==0 and n==0:
                 coordiarray=coordi[n]
              else:
                 #print(coordi[n])
                 coordiarray = np.column_stack((coordiarray,coordi[n]))
                 #coordiarray.append(coordi[n], axis=0)
           #print(coordiarray)
           #print(coordiarray[0,-1])
           #print(coordiarray[1,-1])
           #normd.append(normdiff)
           #os.system('python full_estimation.py')
        #print(coordiarray)
        #print('\nNorm difference',normd)
        return coordiarray

    def get_continuous_cmap(hex_list, float_list=None):
        ''' creates and returns a color map that can be used in heat map figures.
            If float_list is not provided, colour map graduates linearly between each color in hex_list.
            If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

            Parameters
            ----------
            hex_list: list of hex code strings
            float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

            Returns
            ----------
            colour map'''
        rgb_list = [Combis.rgb_to_dec(Combis.hex_to_rgb(i)) for i in hex_list]
        if float_list:
            pass
        else:
            float_list = list(np.linspace(0, 1, len(rgb_list)))

        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp

    def hex_to_rgb(value):
        '''
        Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values'''
        value = value.strip("#")  # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    def rgb_to_dec(value):
        '''
        Converts rgb to decimal colours (i.e. divides each value by 256)
        value: list (length 3) of RGB values
        Returns: list (length 3) of decimal values'''
        return [v / 256 for v in value]