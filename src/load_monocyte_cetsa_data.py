# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:02:01 2024

@author: piercetf
"""

import pandas
from pathlib import Path
import seaborn
#import matplotlib
from matplotlib import pyplot

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
    
    # empty string is more accurate and avoids NaN-based pathology
    # later on
    basedata['PG.Genes'] = basedata['PG.Genes'].fillna('')
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
    
    multipeptide_data_table = data_table.loc[multipeptide_data_membership,:]
    
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


def load_data():
    """Load in data and perform filtering and data processing steps
    """
    basedata = load_basedata()
    candidates = load_candidates()
    
    multipep_data, multipep_candidates = remove_unipeptides(basedata, candidates)
    
    remove_deprecated_columns(multipep_candidates)
    
    del multipep_candidates['Valid'] # don't know, based on template
    
    # add aliases for variables to avoid problems with special characters
    multipep_candidates = rename_special_columns(multipep_candidates)
    
    # split out between substance and temperature
    multipep_candidates.loc[:, "Treatment_Numerator"] = multipep_candidates["Condition Numerator"].map(get_left)
    multipep_candidates.loc[:, "Temperature_Numerator"] = multipep_candidates["Condition Numerator"].map(get_right)
    multipep_candidates.loc[:, "Treatment_Denominator"] = multipep_candidates["Condition Denominator"].map(get_left)
    multipep_candidates.loc[:, "Temperature_Denominator"] = multipep_candidates["Condition Denominator"].map(get_right)
    
    # only consider comparisons at the same temperature
    sametemp_multipep_candidates = multipep_candidates.loc[multipep_candidates.Temperature_Numerator == multipep_candidates.Temperature_Denominator,:].copy()
    
    
    temp_ints = pandas.to_numeric(sametemp_multipep_candidates["Temperature_Numerator"])
    sametemp_multipep_candidates['Temperature'] = temp_ints
    
    # split out between substance and temperature
    multipep_data = multipep_data.assign(
        Treatment=multipep_data["R.Condition"].map(get_left),
        Temperature=pandas.to_numeric(multipep_data["R.Condition"].map(get_right))
        )
    
    return multipep_data, sametemp_multipep_candidates

def calc_total_protein_quantity(peptide_data):
    """Total the amount of material observed per-protein identification"""
    grouped = peptide_data.groupby(by=['PG.ProteinAccessions', 
                                     'PG.Genes', 
                                     'R.Replicate',
                                     'Treatment',
                                     'Temperature'])
    # I believe this is the total protein quantity
    total_fg = grouped.sum()['FG.Quantity'] 
    
    # closest known analog to dplyr `ungroup` function
    # see link
    # https://stackoverflow.com/questions/67144303/how-do-i-replicate-r-ungroup-in-python
    peptide_data['Total_FG_Quantity'] = total_fg.reindex(
        peptide_data[['PG.ProteinAccessions',
                      'PG.Genes',
                      'R.Replicate',
                      'Treatment',
                      'Temperature']]
        ).values

def display_counts(data):
    """display the number of tested temperature and number of detected proteins
    as a function of the replicate and treatment"""
    itemcounts = data.groupby(
        by=["R.Replicate", "Treatment"]
        ).nunique().loc[:,["PG.ProteinAccessions", "Temperature"]].reset_index()
    print(itemcounts)
    seaborn.barplot(data=itemcounts, 
                    x="Treatment", 
                    y="Temperature", 
                    hue="R.Replicate"
                    )
    pyplot.ylabel("Number of Temperatures tested")
    pyplot.show()
    
    seaborn.barplot(data=itemcounts,
                    x="Treatment",
                    y="PG.ProteinAccessions",
                    hue="R.Replicate"
                    )
    pyplot.ylabel("Number of detected proteins")

    pyplot.show()


def norm_protein_mintemp(data):
    """"Normalize the protein quantity against the protein quantity observed
    at the lowest tested temperature
    """
    mintemp = min(data['Temperature'])
    lowtempdata = data.loc[data['Temperature'] == mintemp,:].copy()
    lowtempdata.loc[:,'Referent_Protein'] = lowtempdata.loc[:, 'Total_FG_Quantity']
    lowtempgroups = lowtempdata.groupby(
        by=["PG.ProteinAccessions",
            "PG.Genes",
            "R.Replicate",
            "Treatment"]).min()
    merged_frame = data.join(lowtempgroups, how="left",
                             on=["PG.ProteinAccessions",
                                 "PG.Genes",
                                 "R.Replicate",
                                 "Treatment"],
                             rsuffix="_lowtemp")
    return data['Total_FG_Quantity'] / merged_frame['Referent_Protein']


def prepare_data():
    "General data loading routine, including filtering and normalization"
    filtered_data, filtered_candidates = load_data()
    calc_total_protein_quantity(filtered_data)
    display_counts(filtered_data)
    filtered_data.loc[:,"Normalized_FG_Quantity"] = norm_protein_mintemp(filtered_data)
    return filtered_data, filtered_candidates

