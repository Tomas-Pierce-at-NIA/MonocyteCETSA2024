# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:15:23 2024

@author: piercetf
"""

# Based on the R script that was provided as a starting file for this project

# and also based very heavily on Childs' NPARC

import copy

import pandas
from pathlib import Path
import seaborn
from matplotlib import pyplot
import numpy
from sklearn import preprocessing, linear_model
from scipy import integrate, stats

import statsmodels.api as sm

import load_monocyte_cetsa_data as load


ALPHA = 0.05
EPSILON = 1e-10
R_THRESHOLD = 0.6
F2_THRESHOLD = 0.35 # Cohen f**2 large effect size criterion

VIEW = False

def logit(arr):
    return numpy.log(arr / (1 - arr))

def unlogit(arr):
    return 1 / (numpy.exp(-arr) + 1)


def crimp(datum):
    datum = float(datum)
    if datum >= 1:
        return 1.0 - EPSILON
    elif datum <= 0:
        return EPSILON
    else:
        return datum



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

def euclidean(vec1, vec2):
    v1 = numpy.array(vec1)
    v2 = numpy.array(vec2)
    diffs = v1 - v2
    squares = diffs ** 2
    total = squares.sum()
    return numpy.sqrt(total)


def logfold_euclidean(vec1, vec2):
    # log(A/B) = log(A) - log(B)
    v1 = numpy.array(vec1)
    v2 = numpy.array(vec2)
    log1 = numpy.log2(v1)
    log2 = numpy.log2(v2)
    return euclidean(log1, log2)
    

def group_differences(model):
    dmsoparams = model.coef_[0, [1, 5]]
    fisetinparams = model.coef_[0, [2, 6]]
    myricetinparams = model.coef_[0, [3, 7]]
    quercetinparams = model.coef_[0, [4, 8]]
    
    fisetin_v_dmso = logfold_euclidean(fisetinparams, dmsoparams)
    quercetin_v_dmso = logfold_euclidean(quercetinparams, dmsoparams)
    myricetin_v_dmso = logfold_euclidean(myricetinparams, dmsoparams)
    
    fisetin_v_myricetin = logfold_euclidean(fisetinparams, myricetinparams)
    quercetin_v_myricetin = logfold_euclidean(quercetinparams, myricetinparams)
    
    return [
        fisetin_v_dmso,
        quercetin_v_dmso,
        myricetin_v_dmso,
        fisetin_v_myricetin,
        quercetin_v_myricetin
        ]




if __name__ == '__main__':
    data, candidates = load.prepare_data()
    
    focused_subset = data.loc[:, ['PG.ProteinAccessions',
                                  'PG.Genes',
                                  'R.Replicate',
                                  'Temperature',
                                  'Treatment',
                                  'Normalized_FG_Quantity']]
    
    focused_subset.loc[:, "log2(Normalized FG Quantity)"] = numpy.log2(
        focused_subset.loc[:,"Normalized_FG_Quantity"]
        )
    
    focused_subset.loc[:, "arctan(Temp)"] = numpy.arctan(
        focused_subset.loc[:,"Temperature"]
        )
    
    treatmentEncoder = preprocessing.OneHotEncoder(sparse_output=False, 
                                                   dtype=numpy.int32)
    treatmentEncoder.fit(data.loc[:,['Treatment']])
    treatmentOneHot = treatmentEncoder.transform(data.loc[:, ['Treatment']])
    categories = treatmentEncoder.categories_[0]

    
    for i, category in enumerate(categories):
        focused_subset = focused_subset.assign(**{category : treatmentOneHot[:,i]})

        
    #breakpoint()
    #focused_subset = focused_subset.drop_duplicates()
    focused_subset.loc[:,"Crimped_Protein"] = focused_subset.loc[:,"Normalized_FG_Quantity"].map(crimp)
                                                                 
    protein_groups = focused_subset.groupby(by=['PG.ProteinAccessions', 'PG.Genes'])
    
    grouplings = numpy.concatenate([numpy.identity(4)] * (71-37), axis=0)
    templings = numpy.repeat(numpy.arange(37, 71), 4)[...,numpy.newaxis]
    spaninputs = numpy.concatenate((templings, grouplings), axis=1)
    
    templings_with_const = sm.add_constant(templings)
    
    
    cats = treatmentEncoder.categories_[0]
    
    
    
    interactionEncoder = preprocessing.PolynomialFeatures(interaction_only=True,
                                                          include_bias=False)
    
    span_interact_inputs = interactionEncoder.fit_transform(spaninputs)[:,:9]
    
    span_treatments = treatmentEncoder.inverse_transform(spaninputs[:,1:])[:,0]
    
    aware_models = []
    naive_models = []
    prot_ids = []
    
    for ident, table in protein_groups:
        cleantable = table.dropna()
        #cleantable = cleantable.loc[cleantable['Temperature'] > 37,:]
        if len(cleantable) == 0:
            continue
        if max(cleantable['Normalized_FG_Quantity']) > 1:
            continue
        if max(cleantable['Temperature']) == 37:
            continue
        
        
        # note that this loses index information
        inputdata = interactionEncoder.fit_transform(
            cleantable.loc[:, ['Temperature', *cats]]
            )
        inputs_table = pandas.DataFrame(data=inputdata,
                                 columns=interactionEncoder.get_feature_names_out())
        
        # get rid of columns of all zeros that are zeros definitionally
        inputs_table = inputs_table.loc[:, inputs_table.columns[0:9]]
        
        # so we have to get rid of index information here or pandas
        # will think these things are not aligned
        norm_prot = cleantable.reset_index()['Normalized_FG_Quantity']
        
        # this will place the normalized protein in the last column
        protein_table = inputs_table.assign(Norm_Prot = norm_prot)
        
        # check to see if not detected in a condition
        # if there is any condition in which the protein
        # is not detected, the protein is skipped
        totals = protein_table.sum()
        if any(totals == 0):
            continue
        
        with_constant = sm.add_constant(protein_table)
        
        naive = sm.Logit(endog=protein_table['Norm_Prot'],
                         exog=with_constant[['Temperature', 'const']],
                         missing="raise")
        
        try:
            naive_est = naive.fit(maxiter=101)
        except numpy.linalg.LinAlgError as le:
            print(le)
            print(ident)
            continue
        
        # have to exclude last column from predictor to avoid 
        # y ~ y problem
        aware = sm.Logit(endog=protein_table['Norm_Prot'],
                         exog=protein_table.loc[:, protein_table.columns[:-1]],
                         missing="raise")
        
        try:
            aware_est = aware.fit(maxiter=101)
        except numpy.linalg.LinAlgError as le:
            print(le)
            print(ident)
            continue
        
        aware_models.append(aware_est)
        
        naive_models.append(naive_est)
        
        prot_ids.append(ident)
        
        if VIEW:
            naive_preds = naive_est.predict(exog=templings)
            aware_preds = aware_est.predict(exog=span_interact_inputs)
            
            pred_table = pandas.DataFrame(
                data=span_interact_inputs,
                columns=interactionEncoder.get_feature_names_out()[:8])
            
            pred_table.loc[:,'Treatment'] = span_treatments
            
            pred_table.loc[:, 'Naive'] = naive_preds
            pred_table.loc[:, 'Aware'] = aware_preds
            
            protein_table.loc[:, "Treatment"] = treatmentEncoder.inverse_transform(
                protein_table.loc[:, cats]
                )[:,0]
            
            ax = seaborn.scatterplot(protein_table,
                                      x='Temperature',
                                      y='Norm_Prot',
                                      hue='Treatment')
            seaborn.lineplot(pred_table,
                              x='Temperature',
                              y='Naive',
                              color='grey',
                              ax=ax)
            seaborn.lineplot(pred_table,
                              x='Temperature',
                              y='Aware',
                              hue='Treatment',
                              ax=ax)
            pyplot.show()
        
    
    null_rss = []
    alt_rss = []
    for aware, naive in zip(aware_models, naive_models):
        if not aware.mle_retvals['converged']:
            continue
        if not naive.mle_retvals['converged']:
            continue
        null_rss.append(sum(naive.resid_pearson ** 2))
        alt_rss.append(sum(aware.resid_pearson ** 2))
    
    null_rss = numpy.array(null_rss)
    alt_rss = numpy.array(alt_rss)
    # restrict attention to where nonnNaN values can be had
    null_hyp_not_nan = ~numpy.isnan(null_rss)
    alt_hyp_not_nan = ~numpy.isnan(alt_rss)
    null_rss = null_rss[null_hyp_not_nan & alt_hyp_not_nan]
    alt_rss = alt_rss[null_hyp_not_nan & alt_hyp_not_nan]
    diff = null_rss - alt_rss
    alt_rss = alt_rss[~numpy.isnan(diff)]
    null_rss = null_rss[~numpy.isnan(diff)]
    # sigma0 squared
    sig_square = 0.5 * diff.var() / diff.mean()
    #sig_square = 0.5 * stats.median_abs_deviation(diff) / numpy.median(diff)
    
    diff_over_sig = diff / sig_square
    alt_over_sig = alt_rss / sig_square
    
    d1_est = stats.chi2.fit(diff_over_sig)[0]
    d2_est = stats.chi2.fit(alt_over_sig)[0]
    
    p_values = []
    
    total_pseudo_r2 = []
    rel_effect = []
    
    prot_names = []
    
    gene_names = []
    
    for ident, aware, naive in zip(prot_ids, aware_models, naive_models):
        # should only consider models which have converged,
        # otherwise cannot detect any significance of interaction
        if not aware.mle_retvals['converged']:
            continue
        if not naive.mle_retvals['converged']:
            continue
        local_alt_rss = sum(aware.resid_pearson ** 2)
        local_null_rss = sum(naive.resid_pearson ** 2)
        f_stat = (d2_est / d1_est) * ((local_null_rss - local_alt_rss) / local_alt_rss)
        p_val = stats.f.sf(f_stat, d1_est, d2_est)
        p_values.append(p_val)
        
        total_pseudo_r2.append(aware.prsquared)
        
        cohen_f2 = (aware.prsquared - naive.prsquared) / (1 - aware.prsquared)
        
        rel_effect.append(cohen_f2)
        
        prot_names.append(ident[0])
        
        gene_names.append(ident[1])
    
    
    prot_id_arr = numpy.array(prot_ids)
    
    stat_table = pandas.DataFrame({
        
        'PG.ProteinAccessions' : prot_names,
        'PG.Genes' : gene_names,
        'pval' : p_values,
        'pseudo R2' : total_pseudo_r2,
        'Cohen f2' : rel_effect,
        
        })
    
    prot_stats = stat_table.dropna()
    
    of_potential_interest = prot_stats.loc[prot_stats['pval'] < 1,:]
    
    corrected_stats = of_potential_interest.assign(
        adj_pval = stats.false_discovery_control(
            of_potential_interest.loc[:, 'pval'],
            method='bh'
            )
        )
    
    sorted_by_effect = corrected_stats.sort_values('Cohen f2').dropna()
    
    sorted_by_effect = sorted_by_effect.loc[sorted_by_effect.adj_pval < ALPHA, :]
    
    sorted_by_effect.to_csv("C:/Users/piercetf/OneDrive - National Institutes of Health/Documents/CETSA_reports/mostRecentReport.csv")
    
    