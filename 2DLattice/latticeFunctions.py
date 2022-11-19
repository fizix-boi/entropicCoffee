import random
import math
from math import sqrt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def generatePnL(dim, cloudL, N):
    '''
    generatePnL (generate points 'n lattice)
    
    function inputs:
        -dim: the idealized desired length of the box (box
            length will be made odd to make a well-defined origin)
        -cloudL: the length of the cloud of particles in the system
        -N: the number of particles in the system
        
    general algorithm:
        1. Redefine lattice such that the 0 point is well defined
            (making the side length odd)
        2. Calculate the extent of the cloud (with some given cloud
            diameter as an input)
        3. Iterate through and generate an array of points within the
            extent of the cloud
    '''
    
    side = 2 * np.round(dim / 2)
    lattice = side + 1
    
    extent = (cloudL - 1) / 2
    points = np.zeros((N, 2))
    for it in range(N):
        points[it] = [-extent + (it % cloudL), -extent + (int(it / cloudL) % cloudL)]
    
    return(points, lattice)



def plotLattice(P, L, titlr, dpid, filr):
    '''
    plotLattice
    
    function inputs:
        -P: the points of the system
        -L: the lattice of the system
        -dpid: the desired dpi of the photo
        -titlr: the deisred title for the plot
    
    general algorithm:
        1. Cast x and y values to separate arrays
        2. Plot a scatter of these arrays
    '''
    
    xVals = np.zeros(len(P))
    yVals = np.zeros(len(P))
    for it in range(len(P)):
        xVals[it] = P[it][0]
        yVals[it] = P[it][1]
    
    bound = int((L - 1) / 2)
    
    fig, ax = plt.subplots()
    plt.scatter(xVals, yVals, s = 2)
    plt.xlim(-bound - 1, bound + 1)
    plt.ylim(-bound - 1, bound + 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titlr)
    plt.savefig(filr, dpi = dpid)
    plt.close()
    
    
    
def pLTitlrIte(ite):
    '''
    pLTitlrIte
    
    function inputs:
        -ite: current iteration of system
    '''
    
    return('Orientation of Box at Iteration #10^' + str(np.round(math.log10(ite), 3)))
    


def stepDiff(P, L):
    '''
    stepDiff
    
    function inputs:
        -P: the points of the system
        -L: the lattice of the system
    
    general algorithm:
        1. Define the bounds of the box
        2. For all the particles in the systems
            1. Generate a random number for selecting the next
                direction of movement
            2. Step in that random direction
                1. If the new step has taken the particle out of the
                    box, force it back in as if it ricocheted off the
                    wall of the system
            3. Save the new location of each particle once it is calculated
                for an corrected
    '''
    
    bound = int((L - 1) / 2)
    N = len(P)
    
    for sel in range(len(P)):
        locSel = P[sel]

        dirs = ( (1, 0), (-1, 0), (0, 1), (0, -1) )
        r = random.random()
        if(r <= 0.25): 
            locSel += dirs[0]
        elif(r <= 0.5):
            locSel += dirs[1]
        elif(r <= 0.75):
            locSel += dirs[2]
        else:
            locSel += dirs[3]

        if(abs(locSel[0]) == bound + 1 or abs(locSel[1]) == bound + 1):
            if(locSel[0] == (bound + 1)):
                locSel[0] -= 2
            if(locSel[0] == -(bound + 1)):
                locSel[0] += 2
            if(locSel[1] == (bound + 1)):
                locSel[1] -= 2
            if(locSel[1] == -(bound + 1)):
                locSel[1] += 2

        P[sel] = locSel
    
    return(P)



def stepDiffHole(P, L, H):
    '''
    stepDiff
    
    function inputs:
        -P: the points of the system
        -L: the lattice of the system
        -H: width of hole in side of system
    
    general algorithm:
        1. Define the bounds of the box
        2. For all the particles in the systems
            1. Generate a random number for selecting the next
                direction of movement
            2. Step in that random direction
                1. If the new step has taken the particle out of the
                    box, force it back in as if it ricocheted off the
                    wall of the system
                2. If the new stop has taken the particle through the hole,
                    instead remove the particle from the points array
            3. Save the new location of each particle once it is calculated
                for an corrected
    '''
    
    bound = int((L - 1) / 2)
    outtr = int((H - 1) / 2)
    N = len(P)
    
    for sel in range(len(P)):
        if(sel >= len(P)):
            break
        locSel = P[sel]
        
        dirs = ( (1, 0), (-1, 0), (0, 1), (0, -1) )
        r = random.random()
        if(r <= 0.25): 
            locSel += dirs[0]
        elif(r <= 0.5):
            locSel += dirs[1]
        elif(r <= 0.75):
            locSel += dirs[2]
        else:
            locSel += dirs[3]

        skp = False #escapee

        if(abs(locSel[0]) == bound + 1 or abs(locSel[1]) == bound + 1):
            if(locSel[0] == (bound + 1)):
                locSel[0] -= 2
            elif(locSel[0] == -(bound + 1)):
                locSel[0] += 2
            elif(locSel[1] == (bound + 1)):
                locSel[1] -= 2
            elif(locSel[1] == -(bound + 1)):
                if(abs(locSel[0]) <= outtr):
                    skp = True
                else:
                    locSel[1] += 2

        if(not skp):
            P[sel] = locSel
        else:
            P = np.delete(P, sel, 0)
    
    return(P)



def calcEnt(P, L):
    '''
    calcEnt
    
    function inputs:
        -P: the points of the system
        -L: the lattice of the system
    
    general algorithm:
        1. Define the bounds of a new grid that will be used
            to calculate the entropy
        2. Translate the positions of the particles into coordinates
            in this grid
        3. Calculate the number of points in each grid space in the
            entropy grid
        4. Turn this counts into the probabilities of finding a particle
            in a given subsection of the lattice for a given orientation
        5. Use the equation $S=-\Sigma_i p_i\log p_i over all the
            different portions in the entropy grid to find the total
            entropy of the system
    '''

    N = len(P)
    
    numBinSide = 20
    numBin = numBinSide ** 2
    gridL = L / numBinSide
    count = np.zeros((numBinSide, numBinSide))
    
    normP = np.floor(np.divide(P, gridL))
    indP = np.matrix(np.add(normP, numBinSide / 2), dtype = int)
    
    for it in range(N):
        count[indP[it, 0]][indP[it, 1]] += 1
    
    p_i = np.divide(count, N)
    
    loggers = np.zeros((numBinSide, numBinSide))
    for it1 in range(numBinSide):
        for it2 in range(numBinSide):
            if(count[it1][it2] != 0):
                loggers[it1][it2] = math.log(p_i[it1][it2])
            
    S_i = -1 * np.multiply(p_i, loggers)
    S = sum(sum(S_i))
    
    return(S)