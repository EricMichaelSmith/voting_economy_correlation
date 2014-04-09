# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:56:38 2014

@author: Eric Smith

Determines whether a correlation exists between 2008/2012 voting shifts and unemployment shifts

2014-04-08: Make everything modular. Make a function to scale a colormap appropriately and then use that to color a shape plot; maybe add a colorbar?
"""

from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp

import config
reload(config)
import election2008
reload(election2008)
import election2012
reload(election2012)
import fips
reload(fips)
import unemployment
reload(unemployment)



def main():
    
    # Reading in county and state names
    fipsDF = fips.main()
    
    # Reading in the 2008 election file
    (election2008_DF, shapeIndexL, shapeL) = election2008.main()
    
    # Reading in the 2012 election file
    election2012_DF = election2012.main()

    # Reading in the unemployment files
    fileNameS_L = ["laucnty08.txt", "laucnty09.txt", "laucnty10.txt", "laucnty11.txt",
                  "laucnty12.txt"]
    yearL = range(2008, 2013)
    unemploymentDF_L = list()
    for fileNameS, year in zip(fileNameS_L, yearL):
      unemploymentDF_L.append(unemployment.main(fileNameS, year))
    
    # Merging DataFrames
    fullDF = fipsDF.join(election2008_DF, how='inner')
    fullDF = fullDF.join(election2012_DF, how='inner')
    for unemploymentDF in unemploymentDF_L:
        fullDF = fullDF.join(unemploymentDF, how='inner')
    
    # Save to csv file
    #fullDF.to_csv(os.path.join(config.outputPathS, 'fullDF.csv'))
    
    
    ## [[[testing plotting]]]
    fullDF.loc[:, 'DemIsHigher2012'] = (fullDF.loc[:, 'Election2012Dem'] >
                                        fullDF.loc[:, 'Election2012Rep'])
    shapeFig = plt.figure()
    ax = shapeFig.add_subplot(1, 1, 1)
    shapeBoundsAllShapesL = [float('inf'), float('inf'), float('-inf'), float('-inf')]
    for lFIPS in fullDF.index:
        if fullDF.loc[lFIPS, 'DemIsHigher2012']:
            shapeColorT = (0, 0, 1)
        else:
            shapeColorT = (1, 0, 0)
        
        iShapeL = [i for i,j in enumerate(shapeIndexL) if j==lFIPS]
        for iShape in iShapeL:            
            shapeBoundsThisShapeL = shapeL[iShape].bbox
            shapeBoundsAllShapesL[0] = \
                min(shapeBoundsThisShapeL[0], shapeBoundsAllShapesL[0])
            shapeBoundsAllShapesL[1] = \
                min(shapeBoundsThisShapeL[1], shapeBoundsAllShapesL[1])
            shapeBoundsAllShapesL[2] = \
                max(shapeBoundsThisShapeL[2], shapeBoundsAllShapesL[2])
            shapeBoundsAllShapesL[3] = \
                max(shapeBoundsThisShapeL[3], shapeBoundsAllShapesL[3])
            
            thisShapesPatches = []
            pointsA = np.array(shapeL[iShape].points)
            shapeFileParts = shapeL[iShape].parts
            allPartsL = list(shapeFileParts) + [pointsA.shape[0]]
            for lPart in xrange(len(shapeFileParts)):
                thisShapesPatches.append(patches.Polygon(
                    pointsA[allPartsL[lPart]:allPartsL[lPart+1]]))
            ax.add_collection(PatchCollection(thisShapesPatches,
                                              color=shapeColorT))
    ax.set_xlim(-127, -65)
    ax.set_ylim(23, 50)
#    ax.set_xlim(shapeBoundsAllShapesL[0], shapeBoundsAllShapesL[2])
#    ax.set_ylim(shapeBoundsAllShapesL[1], shapeBoundsAllShapesL[3])
    

    ## (1) Shape plot of vote shift
    # {{{import BrBG colormap (see http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps)}}}
    # {{{find conversion factor from range you want to color to colormap}}}
    # {{{}}}

    ## (2) Shape plot of jobs shift
    # {{{}}}
    
    
    ## (3) Make a basic plot of unemployment shift vs. election shift
    percentDem2008_SR = fullDF.Election2008Dem / fullDF.Election2008Total
    percentDem2012_SR = fullDF.Election2012Dem / fullDF.Election2012Total
    demShiftSR = percentDem2012_SR - percentDem2008_SR
    uRateShiftSR = fullDF.URate2012 - fullDF.URate2008
    scatterFig = plt.figure()
    ax = scatterFig.add_subplot(1, 1, 1)
    plt.scatter(uRateShiftSR, 100*demShiftSR, edgecolors='none')
    ax.set_xlim(-5, 10)
    ax.set_ylim(-25, 15)
    print(sp.stats.pearsonr(uRateShiftSR, demShiftSR))
    
    
    ## (4) {{{}}}
    # {{{}}}
    
    
    ## (5) Quickly plot all of the counties in the bottom right clump
    inClumpB_SR = (uRateShiftSR > 2.57) & (100*demShiftSR < -11.6)
    
    clumpScatterFig = plt.figure()
    ax = clumpScatterFig.add_subplot(1, 1, 1)
    plt.scatter(uRateShiftSR[~inClumpB_SR], 100*demShiftSR[~inClumpB_SR],
                c='k', edgecolors='none')
    plt.scatter(uRateShiftSR[inClumpB_SR], 100*demShiftSR[inClumpB_SR],
                c='g', edgecolors='none')
    ax.set_xlim(-5, 10)
    ax.set_ylim(-25, 15)
    
    clumpShapeFig = plt.figure()
    ax = clumpShapeFig.add_subplot(1, 1, 1)
    for lFIPS in fullDF.index:
        if inClumpB_SR.loc[lFIPS]:
            shapeColorT = (0, 1, 0)
        else:
            shapeColorT = (0, 0, 0)
        iShapeL = [i for i,j in enumerate(shapeIndexL) if j==lFIPS]
        for iShape in iShapeL:
            thisShapesPatches = []
            pointsA = np.array(shapeL[iShape].points)
            shapeFileParts = shapeL[iShape].parts
            allPartsL = list(shapeFileParts) + [pointsA.shape[0]]
            for lPart in xrange(len(shapeFileParts)):
                thisShapesPatches.append(patches.Polygon(
                    pointsA[allPartsL[lPart]:allPartsL[lPart+1]]))
            ax.add_collection(PatchCollection(thisShapesPatches,
                                              color=shapeColorT))
    ax.set_xlim(-127, -65)
    ax.set_ylim(23, 50)
  
    return fullDF