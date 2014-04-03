# -*- coding: utf-8 -*-
"""
@author: Eric Smith
Created 2014-03-03

Reads in 2012 US election results, from http://www.theguardian.com/news/datablog/2012/nov/07/us-2012-election-county-results-download#data
"""

import numpy as np
import os
import pandas as pd

import config
reload(config)



def main():
  
    # Read in 2012 election data
    filePathS = os.path.join(config.basePathS, 'election_statistics',
                             'US_elect_county__2012.csv')
    fullDF = pd.read_csv(filePathS,
                              low_memory=False)
    fullDF = fullDF.convert_objects(convert_numeric=True)
    
    # Remove entries that correspond to the voting records of the entire state
    validRowsLC = fullDF.loc[:, 'FIPS Code'].astype(bool)
    countyDF = fullDF.loc[validRowsLC, :]
    
    # Extract the correct information for each row
    countyDF.loc[:, 'numDemVotes'] = extract_votes_all_rows(countyDF, 'Dem')
    countyDF.loc[:, 'numGOPVotes'] = extract_votes_all_rows(countyDF, 'GOP')

    # Extract the important fields for each row: State Postal, FIPS Code, County Name, TOTAL VOTES CAST, numDemVotes, numGOPVotes
    desiredColumnsL = ['FIPS Code', 'TOTAL VOTES CAST',
                       'numDemVotes', 'numGOPVotes']
    partialDF = countyDF.reindex(columns=desiredColumnsL)
    partialDF.columns = ['FIPS', 'Election2012Total',
                       'Election2012Dem', 'Election2012Rep']
                       
    # Sum all entries with the same FIPS code (since the New England states report vote totals by municipality instead of by city)
    groupedPartialDF = partialDF.groupby('FIPS')
    finalDF = groupedPartialDF.aggregate(np.sum)
    
    return finalDF



def extract_votes_all_rows(countyDF, partyS):
    numCounties = countyDF.shape[0]
    partyVotesSR = pd.Series(index=countyDF.index)
    numExtractedVotes = 0
    
    # Set votes from 'Party' column
    isCorrectPartyB_SR = countyDF.loc[:, 'Party'] == partyS
    partyVotesSR.loc[isCorrectPartyB_SR] = \
        countyDF.loc[isCorrectPartyB_SR, 'Votes']
    numExtractedVotes += np.sum(isCorrectPartyB_SR)
    
    # Set votes from 'Party.1', 'Party.2', etc. columns
    iParty = 0
    while numExtractedVotes != numCounties:
        iParty += 1
        isCorrectPartyB_SR = countyDF.loc[:, 'Party.' + str(iParty)] == partyS
        partyVotesSR.loc[isCorrectPartyB_SR] = \
            countyDF.loc[isCorrectPartyB_SR, 'Votes.' + str(iParty)]
        numExtractedVotes += np.sum(isCorrectPartyB_SR)
        
    return partyVotesSR