# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:56:38 2014

@author: Eric Smith

Determines whether a correlation exists between 2008/2012 voting shifts and unemployment shifts

2014-04-13: Now, create all of the shape plots you need to by appropriate calls to make_shape_plot. Then, make scatter plots modular as with shape plots, and create all needed scatter plots. Remember to save all plots.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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

    
    ## Create the full DataFrame
    
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
    
    
    ## Plotting
    
    ## [[[testing plotting: delete this]]]
#    fullDF.loc[:, 'DemIsHigher2012'] = (fullDF.loc[:, 'Election2012Dem'] >
#                                        fullDF.loc[:, 'Election2012Rep'])
#    shapeFig = plt.figure()
#    ax = shapeFig.add_subplot(1, 1, 1)
#    shapeBoundsAllShapesL = [float('inf'), float('inf'), float('-inf'), float('-inf')]
#    for lFIPS in fullDF.index:
#        if fullDF.loc[lFIPS, 'DemIsHigher2012']:
#            shapeColorT = (0, 0, 1)
#        else:
#            shapeColorT = (1, 0, 0)
#        
#        iShapeL = [i for i,j in enumerate(shapeIndexL) if j==lFIPS]
#        for iShape in iShapeL:            
#            shapeBoundsThisShapeL = shapeL[iShape].bbox
#            shapeBoundsAllShapesL[0] = \
#                min(shapeBoundsThisShapeL[0], shapeBoundsAllShapesL[0])
#            shapeBoundsAllShapesL[1] = \
#                min(shapeBoundsThisShapeL[1], shapeBoundsAllShapesL[1])
#            shapeBoundsAllShapesL[2] = \
#                max(shapeBoundsThisShapeL[2], shapeBoundsAllShapesL[2])
#            shapeBoundsAllShapesL[3] = \
#                max(shapeBoundsThisShapeL[3], shapeBoundsAllShapesL[3])
#            
#            thisShapesPatches = []
#            pointsA = np.array(shapeL[iShape].points)
#            shapeFileParts = shapeL[iShape].parts
#            allPartsL = list(shapeFileParts) + [pointsA.shape[0]]
#            for lPart in xrange(len(shapeFileParts)):
#                thisShapesPatches.append(mpl.patches.Polygon(
#                    pointsA[allPartsL[lPart]:allPartsL[lPart+1]]))
#            ax.add_collection(mpl.collections.PatchCollection(thisShapesPatches,
#                                              color=shapeColorT))
#    ax.set_xlim(-127, -65)
#    ax.set_ylim(23, 50)
##    ax.set_xlim(shapeBoundsAllShapesL[0], shapeBoundsAllShapesL[2])
##    ax.set_ylim(shapeBoundsAllShapesL[1], shapeBoundsAllShapesL[3])

    # (1) Shape plot of vote shift
    # {{{import BrBG colormap (see http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps)}}}
    # {{{find conversion factor from range you want to color to colormap}}}
    # {{{}}}

    # (2) Shape plot of jobs shift
    # {{{}}}
    
    # (3) Make a basic plot of unemployment shift vs. election shift
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
    
    # (4) {{{}}}
    # {{{}}}
    
    # (5) Quickly plot all of the counties in the bottom right clump
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
                thisShapesPatches.append(mpl.patches.Polygon(
                    pointsA[allPartsL[lPart]:allPartsL[lPart+1]]))
            ax.add_collection(mpl.collections.PatchCollection(thisShapesPatches,
                                              color=shapeColorT))
    ax.set_xlim(-127, -65)
    ax.set_ylim(23, 50)
  
    return fullDF
    
    

def define_boolean_color(inputDF, booleanColumnS, colorT):
    """
    colorT should be two tuples of length 3: one if the boolean value is
    true and one if it is false
    """
    
    colorColumnT = ('red', 'green', 'blue')
    colorDF = pd.DataFrame(np.ndarray((len(inputDF), 3)), columns=colorColumnT)

    # Set columns one by one
    for lColumn, columnS in enumerate(colorColumnT):
        colorDF.ix[inputDF[booleanColumnS], columnS] = colorT[0][lColumn]
        colorDF.ix[~inputDF[booleanColumnS], columnS] = colorT[1][lColumn]
    return zip(colorDF.red, colorDF.green, colorDF.blue)
    
    
    
def define_gradient_color(inputDF, gradient_column_s, colorT):
    """ 
    colorT should be three tuples of length 3: one if the value is maximally
    negative, one if it is zero, and one if it is maximally positive.
    Intermediate values will be interpolated.
    """
    
    # Find the maximum-magnitude value in the column of interest: this will be
    # represented with the brightest color.
    maxMagnitude = max([abs(i) for i in inputDF['gradient_column_s']])

    # For each index in inputDF, interpolate between the values of colorT to
    # find the approprate color of the index
    for index in inputDF.index:
        yield interpolate_gradient_color(colorT,
                                         inputDF.loc[index, gradient_column_s],
                                         maxMagnitude)


                                                      
def interpolate_gradient_color(colorT, value, maxMagnitude):
    """
    Returns a tuple containing the interpolated color for the input value
    """
    
    normalizedMagnitude = abs(value)/maxMagnitude
    
    # The higher the magnitude, the closer the color to farColorT; the lower
    # the magnitude, the closter the color to nearColorT
    nearColorT = colorT[1]
    if value < 0:
        farColorT = colorT[0]
    else:
        farColorT = colorT[2]
    interpolatedColorA = (normalizedMagnitude * np.array(farColorT) +
                          (1-normalizedMagnitude) * np.array(nearColorT))
    return tuple(interpolatedColorA)            
                                                      
    
    
def make_shape_plot(df, shapeIndexL, shapeL, colorType):
    shapeFig = plt.figure()
    ax = shapeFig.add_subplot(1, 1, 1)
    shapeBoundsAllShapesL = [float('inf'), float('inf'), float('-inf'), float('-inf')]    

    colorTypesD = {'boolean': lambda x: define_boolean_color(x),
                     'gradient': lambda x: define_gradient_color(x)}
    colorDF = colorTypesD(colorType)
        
    for lFIPS in df.index:
        thisCountiesColorT = tuple(colorDF.loc[lFIPS])        
        
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
                thisShapesPatches.append(mpl.patches.Polygon(
                    pointsA[allPartsL[lPart]:allPartsL[lPart+1]]))
            ax.add_collection(mpl.collections.PatchCollection(thisShapesPatches,
                                              color=thisCountiesColorT))
    ax.set_xlim(-127, -65)
    ax.set_ylim(23, 50)