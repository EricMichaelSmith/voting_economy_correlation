# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 07:56:38 2014

@author: Eric Smith

Determines whether a correlation exists between 2008/2012 voting shifts and unemployment shifts

Suffixes at the end of variable names:
A: numpy array
D: dictionary
DF: pandas DataFrame
L: list
S: string
SR: pandas Series
T: tuple
Underscores indicate chaining: for instance, "fooT_T" is a tuple of tuples

[[[2014-05-01: Make sure all of the plots are being generated well. So you don't want to take the time to make colorbars, do you?]]]
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    
    
    ## Calculations
    
    # Calculating shifts in data between 2008 and 2012
    fullDF.loc[:, 'PercentDem2008'] = fullDF.Election2008Dem / fullDF.Election2008Total
    fullDF.loc[:, 'PercentDem2012'] = fullDF.Election2012Dem / fullDF.Election2012Total
    fullDF.loc[:, 'DemShift'] = fullDF.PercentDem2012 - fullDF.PercentDem2008
    fullDF.loc[:, 'URateShift'] = fullDF.URate2012 - fullDF.URate2008
    
    # Finding anomalous values
    fullDF.loc[:, 'DemShiftZScore'] = sp.stats.mstats.zscore(fullDF.DemShift)
    fullDF.loc[:, 'URateShiftZScore'] = sp.stats.mstats.zscore(fullDF.URateShift)
    fullDF.loc[:, 'CoAnomalous'] = ((fullDF.DemShiftZScore < -2) &
                                    (fullDF.URateShiftZScore > 2))
    
    
    ## Plotting
    
    # Scatter plot axes
    xLabelS = 'Percent change in unemployment between 2008 and 2012'
    yLabelS = 'Percent change in Obama vote share between 2008 and 2012'

    # (1) Shape plot of vote shift
    make_shape_plot(fullDF.DemShift, shapeIndexL, shapeL,
                    colorTypeS='gradient',
                    colorT_T=((1,0,0),
                              (1,1,1),
                              (0,0,1)))
    plt.savefig(os.path.join(config.outputPathS, 'shape_plot_vote_shift.png'))

    # (2) Shape plot of unemployment rate shift
    make_shape_plot(fullDF.URateShift, shapeIndexL, shapeL,
                    colorTypeS='gradient',
                    colorT_T=((0,0,1),
                              (1,1,1),
                              (1,0,0)))
    plt.savefig(os.path.join(config.outputPathS, 'shape_plot_unemployment_rate_shift.png'))
    
    # (3) Scatter plot of unemployment shift vs. election shift
    xSR_T = (fullDF.URateShift,)
    ySR_T = (100*fullDF.DemShift,)
    colorT_T = ((0,0,1),)
    ax = make_scatter_plot(xSR_T, ySR_T, colorT_T)
    ax.set_xlabel(xLabelS)
    ax.set_ylabel(yLabelS)
    plt.savefig(os.path.join(config.outputPathS, 'scatter_plot_basic.png'))

    # (4) Scatter plot of unemployment shift vs. election shift, highlighting
    # all counties with zscore(unemployment shift) > 2 and zscore(election
    # shift) < -2
    xSR_T = (fullDF.loc[fullDF.CoAnomalous, 'URateShift'],
             fullDF.loc[~fullDF.CoAnomalous, 'URateShift'])
    ySR_T = (100*fullDF.loc[fullDF.CoAnomalous, 'DemShift'],
             100*fullDF.loc[~fullDF.CoAnomalous, 'DemShift'])
    colorT_T = ((0,1,0),
                (0,0,0))
    ax = make_scatter_plot(xSR_T, ySR_T, colorT_T)
    ax.set_xlabel(xLabelS)
    ax.set_ylabel(yLabelS)
    plt.savefig(os.path.join(config.outputPathS, 'scatter_plot_highlight_anomalous.png'))
    
    # (5) Highlight anomalous points on the shape plot
    make_shape_plot(fullDF.CoAnomalous, shapeIndexL, shapeL,
                    colorTypeS='boolean',
                    colorT_T=((0,1,0),
                              (0,0,0)))
    plt.savefig(os.path.join(config.outputPathS, 'shape_plot_highlight_anomalous.png'))
    
    
    ### (6) Running clustering algorithms on unemployment vs. election scatter
    # plots. I don't get separation of the bottom right clump with any of the
    # three methods I've tried, AffinityPropagation, DBSCAN, or KMeans. With
    # AffinityPropagation, the preference values I tried and the number of
    # clusters I got are as follows:
    #   no preference: 809 clusters
    #   preference=-50: 1793 clusters
    #   preference=-200: 2818 clusters
    #   preference=-10: 700 clusters
    #   preference=-1: 263 clusters
    #   preference=-0.1: 343 clusters
    #   preference=-4: 968 clusters
    # With DBSCAN, I either get one massive cluster in the center or, if I shrink
    # eps down way low, a bunch of mini-clusters in the central clump.
    scatterA = fullDF.loc[:, ['URateShift', 'DemShift']].values
    scatterA[:, 1] *= 100
    # I want to plot DemShift in percentage points
    standardizedScatterA = StandardScaler().fit_transform(scatterA)
    
    # Algorithms
    algorithmD = {'AffinityPropagation': 
                      {'estimator': AffinityPropagation(preference=-4),
                       'plot_name': 'scatter_plot_affinity_propagation.png'},
                  'DBSCAN':
                      {'estimator': DBSCAN(eps=0.1, min_samples=10),
                       'plot_name': 'scatter_plot_dbscan.png'},
                  'KMeans':
                      {'estimator': KMeans(n_clusters=2),
                       'plot_name': 'scatter_plot_kmeans.png'}}
    # Each value is a tuple of the estimator and the filename to save the plot as
    for algorithmS, paramD in algorithmD.iteritems():
        estimator = paramD['estimator'].fit(standardizedScatterA)
        ax = make_cluster_scatter_plot(estimator, scatterA, algorithmS)
        ax.set_xlabel(xLabelS)
        ax.set_ylabel(yLabelS)
        plt.savefig(os.path.join(config.outputPathS, paramD['plot_name']))
    

    ## Miscellaneous
    
    # Save to csv file
    fullDF.to_csv(os.path.join(config.outputPathS, 'full_df.csv'))
    
    # Print stats
    print(sp.stats.pearsonr(fullDF.URateShift, fullDF.DemShift))
  
    return fullDF
    
    

def define_boolean_color(booleanSR, colorT_T):
    """
    booleanSR is the boolean-valued series to define the color from. colorT_T
    should be two tuples of length 3: one if the boolean value is
    true and one if it is false.
    """
    
    colorColumnT = ('red', 'green', 'blue')
    colorDF = pd.DataFrame(np.ndarray((len(booleanSR.index), 3)), 
                           index=booleanSR.index,
                           columns=colorColumnT)

    # Set columns one by one
    for lColumn, columnS in enumerate(colorColumnT):
        colorDF.loc[booleanSR, columnS] = colorT_T[0][lColumn]
        colorDF.loc[~booleanSR, columnS] = colorT_T[1][lColumn]
    return (colorDF, -1)
    # The second entry of the returned tuple specifies that there is no maximum
    # magnitude, like we'd have if this were define_gradient_color
    
    
    
def define_gradient_color(valueSR, colorT_T):
    """ 
    valueSR is the series to define the color from. colorT_T should be three
    tuples of length 3: one if the value is maximally negative, one if it is
    zero, and one if it is maximally positive. Intermediate values will be
    interpolated.
    """
    
    # Find the maximum-magnitude value in the column of interest: this will be
    # represented with the brightest color.
    maxMagnitude = max([abs(i) for i in valueSR])

    # For each index in inputDF, interpolate between the values of colorT_T to
    # find the approprate color of the index
    gradientDF = pd.DataFrame(np.ndarray((len(valueSR.index), 3)),
                              index=valueSR.index,
                              columns=['red', 'green', 'blue'])
    for index in valueSR.index:
        gradientDF.loc[index] = \
        interpolate_gradient_color(colorT_T,
                                   valueSR[index],
                                   maxMagnitude)
    return (gradientDF, maxMagnitude)


                                                      
def interpolate_gradient_color(colorT_T, value, maxMagnitude):
    """
    colorT_T is three tuples of length 3: one if the value is maximally negative,
    one if it is zero, and one if it is maximally positive. maxMagnitude sets the
    intensity of the interpolated color. The function returns a tuple containing
    the interpolated color for the input value.
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



def make_colorbar(fig, maxMagnitude, colorT_T):
    """
    Creates a colorbar with the given figure handle fig; the colors are defined
    according to colorT_T and the values are mapped from -maxMagnitude to
    +maxMagnitude.
    """
    
    # {{{}}}
    
    

def make_cluster_scatter_plot(algorithm, scatterA, titleS):
    """
    Plots the result of fitting a clustering algorithm to the 2-column XY data
    contained in scatterA. Use is made of the scikit-learn tutorials at
    http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
    and http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#example-cluster-plot-dbscan-py.
    """
    
    # Plot all data
    scatterFig = plt.figure()
    ax = scatterFig.add_subplot(1, 1, 1)
    labelA = algorithm.labels_
    uniqueLabelSet = set(labelA)
    numClusters = len(uniqueLabelSet) - (1 if -1 in uniqueLabelSet else 0)
    ax.set_title('%s (estimated number of clusters: %d)' % (titleS, numClusters))
    colorA = mpl.cm.Spectral(np.linspace(0, 1, len(uniqueLabelSet)))
    for label, color in zip(uniqueLabelSet, colorA):
        if label == -1:
            color = 'k'
        classMemberL = [index[0] for index in np.argwhere(labelA == label)]
        for index in classMemberL:
            coordL = scatterA[index]
            plt.scatter(coordL[0], coordL[1], c=color,
                                              edgecolors='none')
    
    # Plot x=0 and y=0 lines
    plot_axes_at_zero(ax)
    
    return ax
    
    
    
def make_scatter_plot(xSR_T, ySR_T, colorT_T):
    """
    Creates a scatter plot. xSR_T and ySR_T are length-n tuples containing the n
    Series to be plotted; colors of plot points are given by the length-n tuple
    colorT_T.
    """
    
    # Plot all data
    scatterFig = plt.figure()
    ax = scatterFig.add_subplot(1, 1, 1)
    for lSeries in xrange(len(xSR_T)):
        plt.scatter(xSR_T[lSeries], ySR_T[lSeries],
                    c=colorT_T[lSeries],
                    edgecolors='none')
                    
    # Plot x=0 and y=0 lines
    plot_axes_at_zero(ax)
    
    return ax
                                                      
    
    
def make_shape_plot(valueSR, shapeIndexL, shapeL, colorTypeS, colorT_T):
    """
    Creates a shape plot. valueSR is the Series containing the data to be
    plotted; shapeIndexL indexes the shapes to plot by FIPS code; shapeL contains
    the shapes to plot; colorTypeS defines whether the plot will be shaded
    according to a binary or a gradient; colorT_T defines the colors to shade
    with.
    """
    
    shapeFig = plt.figure(figsize=(11,6))
    ax = shapeFig.add_subplot(1, 1, 1)
    shapeBoundsAllShapesL = [float('inf'), float('inf'), float('-inf'), float('-inf')]    

    colorTypesD = {'boolean': lambda: define_boolean_color(valueSR, colorT_T),
                   'gradient': lambda: define_gradient_color(valueSR, colorT_T)}
    colorT = colorTypesD[colorTypeS]()
    colorDF = colorT[0]
    maxMagnitude = colorT[1]
        
    for lFIPS in valueSR.index:
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
    ax.set_axis_off()
    
    if maxMagnitude != -1:
        make_colorbar(shapeFig, maxMagnitude, colorT_T)
    
    return ax
    
    
    
def plot_axes_at_zero(ax):
    """
    Plots x=0 and y=0 lines on the current plot. The stretching of the lines and
    resetting of the axis limits is kind of a hack.
    """
    
    axisLimitsT = ax.axis()
    plt.plot([2*axisLimitsT[0], 2*axisLimitsT[1]], [0, 0], 'k')
    plt.plot([0, 0], [2*axisLimitsT[2], 2*axisLimitsT[3]], 'k')
    ax.set_xlim(axisLimitsT[0], axisLimitsT[1])
    ax.set_ylim(axisLimitsT[2], axisLimitsT[3])
    return ax