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
    fullDF.to_csv(os.path.join(config.outputPathS, 'full_df.csv'))
    
    
    ## Plotting

    # (1) Shape plot of vote shift
    fullDF.PercentDem2008 = fullDF.Election2008Dem / fullDF.Election2008Total
    fullDF.PercentDem2012 = fullDF.Election2012Dem / fullDF.Election2012Total
    fullDF.DemShift = fullDF.PercentDem2012 - fullDF.PercentDem2008
    make_shape_plot(fullDF, shapeIndexL, shapeL,
                    plotColumnS='DemShift',
                    colorType='gradient',
                    colorT_T=((1,0,0), (0,0,1)))
    plt.savefig(os.path.join(config.outputPathS, 'shape_plot_vote_shift.png'))

    # (2) Shape plot of unemployment rate shift
    fullDF.URateShift = fullDF.URate2012 - fullDF.URate2008   
    make_shape_plot(fullDF, shapeIndexL, shapeL,
                    plotColumnS='URateShift',
                    colorType='gradient',
                    colorT_T=((1,0,0), (0,0,1)))
    plt.savefig(os.path.join(config.outputPathS, 'shape_plot_unemployment_rate_shift.png'))
    
    # (3) Scatter plot of unemployment shift vs. election shift
    xSR_T = (fullDF.URateShift)
    ySR_T = (100*fullDF.DemShift)
    colorT_T = ((0,0,0))
    make_scatter_plot(xSR_T, ySR_T, colorT_T)
    plt.savefig(os.path.join(config.outputPathS, 'scatter_plot_basic.png'))

    # (4) Scatter plot of unemployment shift vs. election shift, highlighting all
    # counties with zscore(unemployment shift) > 2 and zscore(election shift) <
    # -2
    fullDF.DemShiftZScore = sp.stats.mstats.zscore(fullDF.DemShift)
    fullDF.URateShiftZScore = sp.stats.mstats.zscore(fullDF.URateShift)
    # {{{}}}

    # (5)
    # {{{}}}
    
    # Print stats
    print(sp.stats.pearsonr(fullDF.URateShift, fullDF.DemShift))

    
    ## {{{}}}    
    
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
    
    

def define_boolean_color(inputDF, booleanColumnS, colorT_T):
    """
    colorT_T should be two tuples of length 3: one if the boolean value is
    true and one if it is false
    """
    
    colorColumnT = ('red', 'green', 'blue')
    colorDF = pd.DataFrame(np.ndarray((len(inputDF), 3)), columns=colorColumnT)

    # Set columns one by one
    for lColumn, columnS in enumerate(colorColumnT):
        colorDF.ix[inputDF[booleanColumnS], columnS] = colorT_T[0][lColumn]
        colorDF.ix[~inputDF[booleanColumnS], columnS] = colorT_T[1][lColumn]
    return zip(colorDF.red, colorDF.green, colorDF.blue)
    
    
    
def define_gradient_color(inputDF, gradientColumnS, colorT_T):
    """ 
    colorT_T should be three tuples of length 3: one if the value is maximally
    negative, one if it is zero, and one if it is maximally positive.
    Intermediate values will be interpolated.
    """
    
    # Find the maximum-magnitude value in the column of interest: this will be
    # represented with the brightest color.
    maxMagnitude = max([abs(i) for i in inputDF[gradientColumnS]])

    # For each index in inputDF, interpolate between the values of colorT_T to
    # find the approprate color of the index
    for index in inputDF.index:
        yield interpolate_gradient_color(colorT_T,
                                         inputDF.loc[index, gradientColumnS],
                                         maxMagnitude)


                                                      
def interpolate_gradient_color(colorT_T, value, maxMagnitude):
    """
    Returns a tuple containing the interpolated color for the input value
    """
    
    normalizedMagnitude = abs(value)/maxMagnitude
    
    # The higher the magnitude, the closer the color to farColorT; the lower
    # the magnitude, the closter the color to nearColorT
    nearColorT = colorT_T[1]
    if value < 0:
        farColorT = colorT_T[0]
    else:
        farColorT = colorT_T[2]
    interpolatedColorA = (normalizedMagnitude * np.array(farColorT) +
                          (1-normalizedMagnitude) * np.array(nearColorT))
    return tuple(interpolatedColorA)
    
    
    
def make_scatter_plot(xSR_T, ySR_T, colorT_T):
    """
    Creates a scatter plot. xSR_T and ySR_T are length-n tuples containing the n
    Series to be plotted; colors of plot points are given by the length-n tuple
    colorT_T.
    """
    
    scatterFig = plt.figure()
    ax = scatterFig.add_subplot(1, 1, 1)
    for lSeries in xrange(len(xSR_T)):
        plt.scatter(xSR_T[lSeries], ySR_T[lSeries],
                    c=colorT_T[lSeries],
                    edgecolors='none')
    ax.set_xlim(-5, 10)
    ax.set_ylim(-25, 15)
                                                      
    
    
def make_shape_plot(DF, shapeIndexL, shapeL, plotColumnS, colorType, colorT_T):
    """
    Creates a shape plot. DF is the DataFrame containing the data to be plotted;
    shapeIndexL indexes the shapes to plot by FIPS code; shapeL contains the
    shapes to plot; plotColumnS defines which column to plot the data of;
    colorType defines whether the plot will be shaded according to a binary or a
    gradient; colorT_T defines the colors to shade with.
    """
    
    shapeFig = plt.figure()
    ax = shapeFig.add_subplot(1, 1, 1)
    shapeBoundsAllShapesL = [float('inf'), float('inf'), float('-inf'), float('-inf')]    

    colorTypesD = {'boolean': lambda: define_boolean_color(DF, plotColumnS, colorT_T),
                   'gradient': lambda: define_gradient_color(DF, plotColumnS, colorT_T)}
    colorDF = colorTypesD[colorType]
        
    for lFIPS in DF.index:
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