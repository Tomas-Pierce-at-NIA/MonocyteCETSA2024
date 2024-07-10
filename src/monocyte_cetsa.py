# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:15:23 2024

@author: piercetf
"""

# Based on the R script that was provided as a starting file for this project

import copy

import pandas
from pathlib import Path
import seaborn
from matplotlib import pyplot
import numpy
from sklearn import preprocessing, linear_model, svm, gaussian_process
from scipy import optimize, integrate

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


def interpolate(fitted_model, cat_encode, cuboid=True, feat_args=(3,)) -> pandas.DataFrame:
    "Use a fitted model to interpolate predicted thermal stability"
    temps = list(range(50,71))
    logtemps = numpy.log2(temps)
    groupcycle = [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
    interp_conditions = [groupcycle[i % 4] for i in range((71-50)*4)]
    interp_conditions = numpy.array(interp_conditions)
    interp_temp = numpy.array([temps, logtemps]).T
    interp_temp = numpy.repeat(interp_temp, 4, axis=0)
    base_interpol_feats = numpy.concatenate((interp_temp, 
                                             interp_conditions), axis=1)
    if cuboid:
        cuboid_featureprep = preprocessing.PolynomialFeatures(*feat_args)
        interpol_feats = cuboid_featureprep.fit_transform(base_interpol_feats)
        predictions = fitted_model.predict(interpol_feats)
    else:
        predictions = fitted_model.predict(base_interpol_feats)
    
    categories = cat_encode.categories_[0]
    
    dframe = pandas.DataFrame(base_interpol_feats,
                              columns=['Temperature',
                                       'log2(Temp)',
                                       *categories])
    
    dframe = dframe.assign(Prediction = predictions)
    
    dframe = dframe.assign(Treatment = cat_encode.inverse_transform(
        base_interpol_feats[:,2:])[:,0])
    
    return dframe
    

def fit_each_protein(raw_model, protein_groups, cuboid=True, feat_args=(3,)):
    cuboid_featureprep = preprocessing.PolynomialFeatures(*feat_args)
    
    identifiers = []
    modelpredicts = []
    fitted_models = {}
    
    for identifier, grouptable in protein_groups:
        raw_model = copy.deepcopy(raw_model)
        numeric_only = grouptable.drop(['PG.ProteinAccessions', 
                                        'PG.Genes', 
                                        'R.Replicate', 
                                        'Treatment'],
                                       axis=1,
                                       inplace=False)
        y = numpy.array(numeric_only['Normalized_FG_Quantity'])
        x_base = numpy.array(numeric_only[['Temperature',
                                           'arctan(Temp)',
                                           *categories]])
        if cuboid:
            x_data = cuboid_featureprep.fit_transform(x_base)
            

            try:
                raw_model.fit(x_data, y)
            except ValueError as er:
                print("Could not fit {}".format(identifier))
                print("reason: {}".format(er))
                print("skipping")
                continue
        else:
            try:
                raw_model.fit(x_base, y)
            except ValueError as er:
                print("Could not fit {}".format(identifier))
                print("reason: {}".format(er))
                print("skipping")
                continue
        
        dframe = interpolate(raw_model, treatmentEncoder, cuboid, feat_args)
        identifiers.append(identifier)
        modelpredicts.append(dframe)
        fitted_models[identifier[0]] = copy.deepcopy(raw_model)
        
    return identifiers, modelpredicts, fitted_models


def calc_differences(focused_subset):
    temps = focused_subset.Temperature.unique()
    protein_groups = focused_subset.groupby(by=['PG.ProteinAccessions'])
    frames = []
    
    QUANT = 'Normalized_FG_Quantity'
    
    for pair in protein_groups:
        ident, raw = pair
        dmso = raw.loc[raw['Treatment'] == 'DMSO', :]
        fisetin = raw.loc[raw['Treatment'] == 'Fisetin', :]
        quercetin = raw.loc[raw['Treatment'] == 'Quercetin', :]
        myricetin = raw.loc[raw['Treatment'] == 'Myricetin', :]
        
        dmso_avg = dmso.groupby(by=['Temperature']).mean(True).reset_index()
        fisetin_avg = fisetin.groupby(by=['Temperature']).mean(True).reset_index()
        quercetin_avg = quercetin.groupby(by=['Temperature']).mean(True).reset_index()
        myricetin_avg = myricetin.groupby(by=['Temperature']).mean(True).reset_index()
        
        fisetin_v_dmso = []
        quercetin_v_dmso = []
        myricetin_v_dmso = []
        
        fisetin_v_myricetin = []
        quercetin_v_myricetin = []
        
        
        for temp in temps:
            
            temp_dmso = dmso_avg[dmso_avg['Temperature'] == temp]
            temp_fisetin = fisetin_avg[fisetin_avg['Temperature'] == temp]
            temp_quercetin = quercetin_avg[quercetin_avg['Temperature'] == temp]
            temp_myricetin = myricetin_avg[myricetin_avg['Temperature'] == temp]
            
            dmso_idx = temp_dmso.first_valid_index()
            fisetin_idx = temp_fisetin.first_valid_index()
            quercetin_idx = temp_quercetin.first_valid_index()
            myricetin_idx = temp_myricetin.first_valid_index()
            
            if dmso_idx is None:
                fisetin_v_dmso.append(numpy.nan)
                quercetin_v_dmso.append(numpy.nan)
                myricetin_v_dmso.append(numpy.nan)
            else:
                dmso_mmt = temp_dmso.loc[dmso_idx, QUANT]
                if fisetin_idx is None:
                    fisetin_v_dmso.append(numpy.nan)
                else:
                    diff = temp_fisetin.loc[
                        fisetin_idx, QUANT] - dmso_mmt
                    fisetin_v_dmso.append(diff)
                if quercetin_idx is None:
                    quercetin_v_dmso.append(numpy.nan)
                else:
                    diff = temp_quercetin.loc[
                        quercetin_idx, QUANT] - dmso_mmt
                    quercetin_v_dmso.append(diff)
                if myricetin_idx is None:
                    myricetin_v_dmso.append(numpy.nan)
                else:
                    diff = temp_myricetin.loc[
                        myricetin_idx, QUANT] - dmso_mmt
                    myricetin_v_dmso.append(diff)
            
            if myricetin_idx is None:
                fisetin_v_myricetin.append(numpy.nan)
                quercetin_v_myricetin.append(numpy.nan)
            
            else:
                myricetin_mmt = temp_myricetin.loc[myricetin_idx, QUANT]
                if fisetin_idx is None:
                    fisetin_v_myricetin.append(numpy.nan)
                else:
                    diff = temp_fisetin.loc[
                        fisetin_idx, QUANT] - myricetin_mmt
                    fisetin_v_myricetin.append(diff)
                if quercetin_idx is None:
                    quercetin_v_myricetin.append(numpy.nan)
                else:
                    diff = temp_quercetin.loc[
                        quercetin_idx, QUANT] - myricetin_mmt
                    quercetin_v_myricetin.append(diff)
        
        table = {
            'Temperature' : temps,
            'PG.ProteinAccessions' : [ident] * len(temps),
            'Fisetin v DMSO' : fisetin_v_dmso,
            'Quercetin v DMSO' : quercetin_v_dmso,
            'Myricetin v DMSO' : myricetin_v_dmso,
            'Fisetin v Myricetin' : fisetin_v_myricetin,
            'Quercetin v Myricetin' : quercetin_v_myricetin
            }
        
        frame = pandas.DataFrame(table)
        
        frames.append(frame)
    
    differencesframe = pandas.concat(frames, ignore_index=True)
    
    return differencesframe

# arctangent model for 1 group
def arctan_model_1(x, k, w, x0, y0):
    return k * numpy.arctan(w * (x - x0)) + y0


def get_average_differences(differences, comparison='Fisetin v DMSO'):
    prot_deltas = differences.groupby(by=['PG.ProteinAccessions'])
    averages = []
    names = []
    for name, delta in prot_deltas:
        clean = delta.dropna()
        if len(clean) == 0:
            continue
        #seaborn.scatterplot(clean, x='Temperature', y='Fisetin v DMSO')
        
        simpson = integrate.simpson(clean[comparison], x=clean['Temperature'])
        
        span = max(clean['Temperature']) - min(clean['Temperature'])
        
        if span == 0:
            continue
        
        averages.append( simpson / span)
        names.append(name[0])
    return pandas.DataFrame({'PG.ProteinAccessions' : names,
                             comparison : averages}
                            )


if __name__ == '__main__':
    data, candidates = prepare_data()
    
    focused_subset = data.loc[:, ['PG.ProteinAccessions',
                                  'PG.Genes',
                                  'R.Replicate',
                                  'Temperature',
                                  'Treatment',
                                  'Normalized_FG_Quantity']]
    
    focused_subset.loc[:, "log2(Normalized FG Quantity)"] = numpy.log2(
        focused_subset.loc[:,"Normalized_FG_Quantity"]
        )
    
    # focused_subset.loc[:, "log2(Temperature)"] = numpy.log2(
    #     focused_subset.loc[:, "Temperature"]
    #     )
    
    focused_subset.loc[:, "arctan(Temp)"] = numpy.arctan(
        focused_subset.loc[:,"Temperature"]
        )
    
    treatmentEncoder = preprocessing.OneHotEncoder(sparse_output=False, 
                                                   dtype=numpy.int32)
    treatmentEncoder.fit(data.loc[:,['Treatment']])
    treatmentOneHot = treatmentEncoder.transform(data.loc[:, ['Treatment']])
    categories = treatmentEncoder.categories_[0]
    
    replicateEncoder = preprocessing.OneHotEncoder(sparse_output=False,
                                                   dtype=numpy.int32)
    replicatetable = replicateEncoder.fit_transform(data.loc[:,['R.Replicate']])
    replicate_identities = replicateEncoder.categories_[0]
    
    for i, category in enumerate(categories):
        focused_subset = focused_subset.assign(**{category : treatmentOneHot[:,i]})
        
    for i, repid in enumerate(replicate_identities):
        focused_subset = focused_subset.assign(**{f"Replicate{repid}" : 
                                                  replicatetable[:,i]})
        
    #breakpoint()
    #focused_subset = focused_subset.drop_duplicates()
    
    protein_groups = focused_subset.groupby(by=['PG.ProteinAccessions', 'PG.Genes'])
    
    # prot_idents, prot_preds, models = fit_each_protein(linear_model.RidgeCV(), 
    #                                                    protein_groups, 
    #                                                    True)
    
    # prot_idents, prot_preds, models = fit_each_protein(svm.SVR(kernel="rbf", 
    #                                                             C=0.9,
    #                                                             degree=10,
    #                                                             coef0=1.0),
    #                                                     protein_groups,
    #                                                     False)
    
    
    auc_table = []
    auc_columns = ["prot_ids", "gene_ids", "treatment", "replicate", "area under curve"]
    
    replicates = focused_subset['R.Replicate'].drop_duplicates(ignore_index = True)
    
    differences = calc_differences(focused_subset)
    
    diffs1 = get_average_differences(differences, 'Fisetin v DMSO')
    
    diffs2 = get_average_differences(differences, 'Quercetin v DMSO')
    
    diffs3 = get_average_differences(differences, 'Myricetin v DMSO')
    
    diffs4 = get_average_differences(differences, 'Fisetin v Myricetin')
    
    diffs5 = get_average_differences(differences, 'Quercetin v Myricetin')
    
    fisetin_candidates = diffs1[diffs1['Fisetin v DMSO'] > (diffs1['Fisetin v DMSO'].mean() + 2 * diffs1['Fisetin v DMSO'].std())]
    
    quercetin_candidates = diffs2[diffs2['Quercetin v DMSO'] > (diffs2['Quercetin v DMSO'].mean() + 2 * diffs2['Quercetin v DMSO'].std())]
    
    myricetin_candidates = diffs3[diffs3['Myricetin v DMSO'] > (diffs3['Myricetin v DMSO'].mean() + 2 * diffs3['Myricetin v DMSO'].std())]
    
    fisetin_targets = set(fisetin_candidates['PG.ProteinAccessions'])
    quercetin_targets = set(quercetin_candidates['PG.ProteinAccessions'])
    myricetin_targets = set(myricetin_candidates['PG.ProteinAccessions'])
    
    fisetin_candidates2 = diffs4[diffs4['Fisetin v Myricetin'] > (diffs4['Fisetin v Myricetin'].mean() + 2 * diffs4['Fisetin v Myricetin'].std())]
    quercetin_candidates2 = diffs5[diffs5['Quercetin v Myricetin'] > (diffs5['Quercetin v Myricetin'].mean() + 2 *diffs5['Quercetin v Myricetin'].std())]
    
    fisetin_targets2 = set(fisetin_candidates2['PG.ProteinAccessions'])
    quercetin_targets2 = set(quercetin_candidates2['PG.ProteinAccessions'])
    
    
    shared_nonsenolytic = fisetin_targets.intersection(quercetin_targets).intersection(myricetin_targets)
    
    shared_targets = fisetin_targets.intersection(quercetin_targets).difference(myricetin_targets)
    
    unique_fisetin_targets = fisetin_targets.difference(quercetin_targets).difference(myricetin_targets)
    unique_quercetin_targets = quercetin_targets.difference(myricetin_targets).difference(fisetin_targets)
    unique_myricetin_targets = myricetin_targets.difference(quercetin_targets).difference(fisetin_targets)
    
    fisetin_myricetin = fisetin_targets.intersection(myricetin_targets).difference(quercetin_targets)
    quercetin_myricetin = quercetin_targets.intersection(myricetin_targets).difference(fisetin_targets)
    
    print("Shared targets, probably not involved in senolytic pathways")
    for item in shared_nonsenolytic:
        print(item, end = ' ')
    print('\n"')
    
    print("shared targets of quercetin and fisetin but not myricetin")
    for item in shared_targets:
        print(item, end=' ')
    print('\n"')
    
    print("unique fisetin targets")
    for item in unique_fisetin_targets:
        print(item, end=' ')
    print('\n"')
    
    print("unique quercetin targets")
    for item in unique_quercetin_targets:
        print(item, end=' ')
    print('\n"')
    
    print("unique myricetin targets")
    for item in unique_myricetin_targets:
        print(item, end=' ')
    print('\n"')

    