# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:15:23 2024

@author: piercetf
"""

# Based on the R script that was provided as a starting file for this project


import pandas
import numpy
from pathlib import Path

def load_candidates():
    CACHED = r"C:\Users\piercetf\Downloads\CachedCETSAData\Candidates.tsv"
    CANONICAL = r"T:\TGB\LSS\TGU\Users\Tomas\2024_CETSA_MS\Monocyte_CETSA_Statistical_Analysis\CETSA_ind_temp_analysis_starting_files\Candidates.tsv"
    cached_path = Path(CACHED)
    canon_path = Path(CANONICAL)
    try:
        candidates = pandas.read_csv(cached_path, sep='\t')
    except Exception as e:
        print("Trying to load cached version failed, falling back to canonical path")
        print("Relevant error was: {}".format(e))
        candidates = pandas.read_csv(canon_path, sep='\t')
    
    return candidates


def load_basedata():
    CACHED = r"C:\Users\piercetf\Downloads\CachedCETSAData\Complete CETSA analysis w F-37-4_Report_Delaney_Default (Normal).tsv"
    CANONICAL = r"T:\TGB\LSS\TGU\Users\Tomas\2024_CETSA_MS\Monocyte_CETSA_Statistical_Analysis\CETSA_ind_temp_analysis_starting_files\Complete CETSA analysis w F-37-4_Report_Delaney_Default (Normal).tsv"
    cached_path = Path(CACHED)
    canon_path = Path(CANONICAL)
    try:
        basedata = pandas.read_csv(cached_path, sep='\t')
    except Exception as e:
        print("Trying to load cached version failed, falling back to canonical path")
        print("Relevant error was: {}".format(e))
        basedata = pandas.read_csv(canon_path, sep='\t')
    
    return basedata

def remove_unipeptides(data_table, candidate_table) -> (pandas.DataFrame, pandas.DataFrame):
    """ Remove proteins identified by only 1 unique peptide from both the
    data table and the candidates table.
    Returns the results in the same order as the parameters.
    """
    onepeptide_candidates = candidate_table[candidate_table['# Unique Total Peptides'] == 1]
    unipeptide_ids = onepeptide_candidates['UniProtIds'].unique()
    multipeptide_candidates = candidate_table[candidate_table['# Unique Total Peptides'] > 1]
    
    def not_in_unipeptides(accession):
        return accession not in unipeptide_ids
    
    multipeptide_data_membership = data_table['PG.ProteinAccessions'].map(
        not_in_unipeptides
        )
    
    multipeptide_data_table = data_table[multipeptide_data_membership]
    
    return multipeptide_data_table, multipeptide_candidates

def remove_deprecated_columns(table):
    colnames = list(table.columns)
    for cname in colnames:
        if cname.startswith('[DEPRECATED]'):
            del table[cname]

def get_left(s :str) -> (str, str): # helper function
    return s.split(' ')[0]

def get_right(s :str) -> (str, str): # helper function
    return s.split(' ')[1]


def rename_special_columns(candidate_table):
    """take a prior table as input and produce a new table,
    where variables with special characters have been renamed to
    prevent the potential for problems.
    """
    return candidate_table.rename(columns={
        "# of Ratios" : "Number_of_Ratios",
        "% Change" : "Percent_Change",
        "# Unique Total Peptides" : "Number_Unique_Total_Peptides",
        "# Unique Total EG.Id" : "Number_Unique_Total_EGid"
        })
    
basedata = load_basedata()
candidates = load_candidates()

multipep_data, multipep_candidates = remove_unipeptides(basedata, candidates)

remove_deprecated_columns(multipep_candidates)

del multipep_candidates['Valid']


# add aliases for variables to avoid problems with special characters
multipep_candidates = rename_special_columns(multipep_candidates)

# split out between substance and temperature
multipep_candidates.loc[:, "Treatment_Numerator"] = multipep_candidates["Condition Numerator"].map(get_left)
multipep_candidates.loc[:, "Temperature_Numerator"] = multipep_candidates["Condition Numerator"].map(get_right)
multipep_candidates.loc[:, "Treatment_Denominator"] = multipep_candidates["Condition Denominator"].map(get_left)
multipep_candidates.loc[:, "Temperature_Denominator"] = multipep_candidates["Condition Denominator"].map(get_right)

# only consider comparisons at the same temperature
sametemp_multipep_candidates = multipep_candidates[multipep_candidates.Temperature_Numerator == multipep_candidates.Temperature_Denominator]
# convert temperature to an integer
sametemp_multipep_candidates.loc[:,'Temperature'] = sametemp_multipep_candidates.Temperature_Numerator.astype(numpy.int32)

