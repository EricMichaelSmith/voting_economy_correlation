# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 12:38:16 2014

@author: Eric

Reads in FIPS codes from https://github.com/hadley/data-counties/blob/master/county-fips.csv (Hadley Wickham)

Suffixes at the end of variable names:
A: numpy array
B: boolean
D: dictionary
DF: pandas DataFrame
L: list
S: string
SR: pandas Series
T: tuple
Underscores indicate chaining: for instance, "fooT_T" is a tuple of tuples
"""

import os
import pandas as pd

import config
reload(config)



def main():
    
    # Read in FIPS data
    filePathS = os.path.join(config.rawDataPathS, 'fips_codes', 'county-fips.csv')
    conversionD = {'msa': (lambda x: str(x)),
                   'pmsa': (lambda x: str(x)),
                   'county': (lambda x: str(x)),
                   'state': (lambda x: str(x)),
                   'county_fips': (lambda x: str('%03d' % int(x))),
                   'state_fips': (lambda x: str(x))}
    fullDF = pd.read_csv(filePathS,
                         converters=conversionD)
    fullDF.loc[:, 'FIPS'] = (fullDF.state_fips + fullDF.county_fips).astype(int)
    
    # Select relevant columns and set index
    finalDF = fullDF.loc[:, ['FIPS', 'county', 'state']]
    finalDF = finalDF.sort(columns='FIPS')
    finalDF = finalDF.set_index('FIPS')
    
    # Using the now-current FIPS code for Miami-Dade County, FL
    finalDF = finalDF.rename(index={12025: 12086})
    finalDF.loc[12086, 'county'] = 'Miami-Dade'

    return finalDF