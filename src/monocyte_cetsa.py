# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:15:23 2024

@author: piercetf
"""

# Based on the R script that was provided as a starting file for this project

# and also based very heavily on Childs' NPARC


import itertools
import pandas
import seaborn
from matplotlib import pyplot
import numpy
from sklearn import preprocessing
from scipy import stats
import statsmodels.api as sm
from statsmodels.api import GLM
from statsmodels.api import families
from statsmodels.api import add_constant
import load_monocyte_cetsa_data as load


ALPHA = 0.05
EPSILON = 1e-9
R_THRESHOLD = 0.6
F2_THRESHOLD = 0.35 # Cohen f**2 large effect size criterion
TEMP = "Temperature"
NORMPROT = "Normalized_FG_Quantity"

VIEW = False
SCALE = True


def crimp(datum):
    datum = float(datum)
    if datum >= 1:
        return 1.0 - EPSILON
    elif datum <= 0:
        return EPSILON
    else:
        return datum    
    

def interact_param_count(base_params: int) -> int:
    
    return base_params + base_params - 1


def estimate_f_dist(aware_models, naive_models):
    null_rss = []
    alt_rss = []
    if len(aware_models) == 0 or len(naive_models) == 0:
        raise Exception("what the hell man?")
    for aware, naive in zip(aware_models, naive_models):
        if not aware.converged:
            continue
        if not naive.converged:
            continue
        null_rss.append(sum(naive.resid_pearson ** 2))
        alt_rss.append(sum(aware.resid_pearson ** 2))
    
    if len(null_rss) == 0:
        raise Exception("nani the fuck")
    
    null_rss = numpy.array(null_rss)
    alt_rss = numpy.array(alt_rss)
    
    null_hyp_not_nan = ~numpy.isnan(null_rss)
    alt_hyp_not_nan = ~numpy.isnan(alt_rss)
    
    null_rss = null_rss[null_hyp_not_nan & alt_hyp_not_nan]
    alt_rss = alt_rss[null_hyp_not_nan & alt_hyp_not_nan]
    
    diff = null_rss - alt_rss
    
    null_rss = null_rss[~numpy.isnan(diff)]
    alt_rss = alt_rss[~numpy.isnan(diff)]
    
    sig_square = 0.5 * diff.var() / diff.mean()
    
    diff_over_sig = diff / sig_square
    alt_over_sig = alt_rss / sig_square
    
    d1_est = stats.chi2.fit(diff_over_sig)[0]
    
    d2_est = stats.chi2.fit(alt_over_sig)[0]
    
    return sig_square, d1_est, d2_est


class TreatmentEncoder:
    """"
    paper thin wrapper around sklearn.preprocessing.OneHotEncoder
    to handle the specific ways in which it needs to be invoked
    and used to reshape our data
    """
    
    def __init__(self, table):
        
        self.encoder = preprocessing.OneHotEncoder(sparse_output=False,
                                                   dtype=numpy.int32)
        
        self.encoder.fit(table.loc[:, ['Treatment']])
        
        self.categories = self.encoder.categories_[0]
        
        self.n_iparams = interact_param_count(len(self.categories) + 1)
    
    def encode_treatments(self, table):
        if len(self.categories) == 0:
            raise ValueError("cannot encode empty set of categories")
        cat_arr = self.encoder.transform(table.loc[:, ['Treatment']])
        for i, category in enumerate(self.categories):
            table = table.assign(
                **{category : cat_arr[:, i]}
                )
        return table
    
    def decode_treatments(self, data):
        if isinstance(data, numpy.ndarray):
            return self.encoder.inverse_transform(data)[:,0]
        elif isinstance(data, pandas.DataFrame):
            subtable = data.loc[:, self.categories]
            return self.encoder.inverse_transform(subtable)[:,0]
        else:
            raise TypeError("cannot decode from type {}".format(type(data)))
    
    def with_treatment_col(self, dataframe):
        treatments = self.decode_categories(dataframe)
        return dataframe.assign(Treatment = treatments)


class InteractionEncoder:
    """
    A wrapper around sklearn.preprocessing.PolynomialFeatures specific
    to its use for interaction terms in a manner specific to the kind
    of analysis needed for CETSA, specifically that only interactions
    between treatment and temperature are to be considered, as 
    treatments are mutually exclusive.
    """
    
    def __init__(self, num_iparams, categories):
        
        self.n_iparams = num_iparams
        self.encoder = preprocessing.PolynomialFeatures(interaction_only=True,
                                                        include_bias=False)
        self.categories = categories
    
    def encode_interacts(self, data):
        if isinstance(data, numpy.ndarray):
            interacts = self.encoder.fit_transform(data)
            return interacts[:, :self.n_iparams]
        elif isinstance(data, pandas.DataFrame):
            subtable = data.loc[:, [TEMP, *self.categories]]
            interacts = self.encoder.fit_transform(subtable)
            colnames = self.encoder.get_feature_names_out([TEMP, *self.categories])
            table = pandas.DataFrame(data=interacts,
                                     columns=colnames)
            return table.loc[:, colnames[0:self.n_iparams]]
        else:
            raise TypeError("Cannot calc interactions on type {}".format(type(data)))
    
    def get_feature_list(self):
        interact_feats = self.encoder.get_feature_names_out([TEMP, *self.categories])
        return interact_feats[:self.n_iparams]


def display_alltreatment(protein_table, 
            aware_model, 
            naive_model, 
            num_cats, 
            interact_encoder, 
            treatment_encoder,
            ident):
    
    grouplings = numpy.concatenate([numpy.identity(num_cats)] * (71-37), axis=0)
    templings = numpy.repeat(numpy.arange(37, 71), num_cats)[...,numpy.newaxis]
    spaninputs = numpy.concatenate((templings, grouplings), axis=1)
    templings_with_const = sm.add_constant(templings)
    span_interact_inputs = interact_encoder.encode_interacts(spaninputs)
    span_treatments = treatment_encoder.decode_treatments(spaninputs[:,1:])
    
    naive_preds = naive_model.predict(exog=templings_with_const)
    aware_preds = aware_model.predict(exog=span_interact_inputs)
    
    pred_table = pandas.DataFrame(
        data=span_interact_inputs,
        columns=interact_encoder.get_feature_list())
    
    pred_table.loc[:,'Treatment'] = span_treatments
    
    pred_table.loc[:, 'Naive'] = naive_preds
    pred_table.loc[:, 'Aware'] = aware_preds
    
    protein_table.loc[:, "Treatment"] = treatment_encoder.decode_treatments(
        protein_table
        )
    
    ax = seaborn.scatterplot(protein_table,
                              x='Temperature',
                              y='Norm_Prot',
                              hue='Treatment')
    seaborn.lineplot(pred_table,
                      x='Temperature',
                      y='Naive',
                      color='black',
                      ax=ax)
    seaborn.lineplot(pred_table,
                      x='Temperature',
                      y='Aware',
                      hue='Treatment',
                      ax=ax)
    
    pyplot.title(ident[0])
    pyplot.show(block=False)
    pyplot.pause(0.25)
    pyplot.close()


def fit_alltreatment_curves(protein_groups, treatment_encoder, interact_encoder, view=VIEW):
    aware_models = []
    naive_models = []
    prot_ids = []
    for ident, table in protein_groups:
        cleantable = table.dropna()
        if len(cleantable) == 0:
            continue
        if max(cleantable['Normalized_FG_Quantity']) > 1:
            continue
        if max(cleantable['Temperature']) == 37:
            continue
        # note that this loses index information
        inputs_table = interact_encoder.encode_interacts(cleantable)
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
        
        naive = GLM(endog=protein_table['Norm_Prot'],
                    exog=with_constant[['const', 'Temperature']],
                    family=families.Binomial())
        
        naive_est = naive.fit()
        
        aware = GLM(endog=protein_table['Norm_Prot'],
                    exog=inputs_table,
                    family=families.Binomial())
        
        aware_est = aware.fit()
        
        aware_models.append(aware_est)
        naive_models.append(naive_est)
        prot_ids.append(ident)
        if view:
            display_alltreatment(protein_table, 
                    aware_est, 
                    naive_est, 
                    len(treatment_encoder.categories),
                    interact_encoder,
                    treatment_encoder,
                    ident)
    return (aware_models, naive_models, prot_ids)


def f_test(aware_model, naive_model, d2, d1):
    alt_rss = sum(aware_model.resid_pearson ** 2)
    null_rss = sum(naive_model.resid_pearson ** 2)
    f_stat = (d2 / d1) * ((null_rss - alt_rss) / alt_rss)
    pval = stats.f.sf(f_stat, d1, d2)
    return pval
        

def alltreatment_analysis(data, candidates, show=False):

    focused_subset = data.loc[:, ['PG.ProteinAccessions',
                                  'PG.Genes',
                                  'R.Replicate',
                                  'Temperature',
                                  'Treatment',
                                  'Normalized_FG_Quantity']]
    treatment_encoder = TreatmentEncoder(focused_subset)
    interact_encoder = InteractionEncoder(treatment_encoder.n_iparams,
                                          treatment_encoder.categories)
    focused_subset = treatment_encoder.encode_treatments(focused_subset)
    focused_subset.loc[:,"Crimped_Protein"] = focused_subset.loc[:,"Normalized_FG_Quantity"].map(crimp)
    protein_groups = focused_subset.groupby(by=['PG.ProteinAccessions', 'PG.Genes'])
    aware_models, naive_models, prot_ids = fit_alltreatment_curves(
        protein_groups,
        treatment_encoder,
        interact_encoder,
        show
        )
    sig_square, d1_est, d2_est = estimate_f_dist(aware_models, naive_models)
    p_values = []
    total_pseudo_r2 = []
    rel_effect = []
    prot_names = []
    gene_names = []
    for ident, aware, naive in zip(prot_ids, aware_models, naive_models):
        # should only consider models which have converged,
        # otherwise cannot detect any significance of interaction
        if not aware.converged:
            continue
        if not naive.converged:
            continue
        p_val = f_test(aware, naive, d2_est, d1_est)
        p_values.append(p_val)
        total_pseudo_r2.append(aware.pseudo_rsquared())
        cohen_f2 = (aware.pseudo_rsquared() - naive.pseudo_rsquared()) / (1 - aware.pseudo_rsquared())
        rel_effect.append(cohen_f2)
        prot_names.append(ident[0])
        gene_names.append(ident[1])
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
        bh_pval = stats.false_discovery_control(
            of_potential_interest.loc[:, 'pval'],
            method='bh'
            )
        )
    sorted_by_effect = corrected_stats.sort_values('Cohen f2').dropna()
    print(len(sorted_by_effect))
    sorted_by_effect = sorted_by_effect.loc[sorted_by_effect.bh_pval < ALPHA, :]
    #sorted_by_effect.to_csv("C:/Users/piercetf/OneDrive - National Institutes of Health/Documents/CETSA_reports/mostRecentReport.csv")
    return sorted_by_effect


def pairwise_modelcomp(data, show=False):
    cols = ['PG.ProteinAccessions','PG.Genes','R.Replicate','Temperature','Treatment', 'Normalized_FG_Quantity']
    focused_subset = data.loc[:, cols]
    if SCALE:
        minmax = preprocessing.MinMaxScaler()
        refocused_subset = focused_subset.assign(
            Temperature = minmax.fit_transform(
                focused_subset.loc[:, ['Temperature']]
                )
            )
    else:
        refocused_subset = focused_subset
    # at some point this loses index order information
    treat_enc = TreatmentEncoder(refocused_subset)
    treat_encoded = treat_enc.encode_treatments(refocused_subset)
    interact_enc = InteractionEncoder(treat_enc.n_iparams, treat_enc.categories)
    interacting = interact_enc.encode_interacts(treat_encoded)
    # so we have to get rid of the other index to preserve corresponding orders
    rerefocused_subset = refocused_subset.reset_index()
    
    interacting.loc[:, NORMPROT] = rerefocused_subset.loc[:, NORMPROT]
    interacting.loc[:, ['PG.ProteinAccessions', 'PG.Genes']] = rerefocused_subset.loc[:,['PG.ProteinAccessions','PG.Genes']]
    
    prepped_data = add_constant(interacting)
    
    aware_models = []
    naive_models = []
    protein_ids = []
    gene_ids = []
    treatment1 = []
    treatment2 = []
    
    protein_groups = prepped_data.groupby(by=['PG.ProteinAccessions', 'PG.Genes'])
    
    for ident, prot_table in protein_groups:
        
        for cond1, cond2 in itertools.combinations(treat_enc.categories, 2):
            
            mask1 = prot_table[cond1] == 1
            mask2 = prot_table[cond2] == 1
            pairmask = mask1 | mask2
            
            subtable = prot_table.loc[pairmask, :]
            
            subtable = subtable.dropna()
            
            if len(subtable) == 0 or max(subtable[NORMPROT]) > 1:
                continue
            
            ie = InteractionEncoder(interact_param_count(3), [cond1, cond2])
            ie.encode_interacts(subtable)
            
            naive = GLM(endog=subtable[NORMPROT],
                        exog=subtable[['const', 'Temperature']],
                        family=families.Binomial())
            
            aware = GLM(endog=subtable[NORMPROT],
                        exog=subtable[ie.get_feature_list()],
                        family=families.Binomial())
            
            naive_est = naive.fit()
            aware_est = aware.fit()
            
            if show:
                treatments = treat_enc.decode_treatments(subtable)
                subtable.loc[:, "Treatment"] = treatments
                if SCALE:
                    subtable.loc[:,"RawTemp"] = minmax.inverse_transform(
                        subtable[['Temperature']]
                        )
                else:
                    subtable.loc[:,"RawTemp"] = subtable.loc[:, "Temperature"]
                
                ncats = len(treat_enc.categories)
                
                grouplings = numpy.concatenate([numpy.identity(ncats)] * (71-37), axis=0)
                templings = numpy.repeat(numpy.arange(37, 71), ncats)[...,numpy.newaxis]
                spaninputs = numpy.concatenate((templings, grouplings), axis=1)
                rescaler = preprocessing.MinMaxScaler()
                templings = rescaler.fit_transform(templings)
                rescaledspan = rescaler.fit_transform(spaninputs)
                
                ptable = pandas.DataFrame(
                    data = interact_enc.encoder.transform(
                        rescaledspan
                        )[:,:interact_enc.n_iparams],
                    columns = interact_enc.get_feature_list()
                    )
                ptable2 = sm.add_constant(ptable)
                
                ptable2 = ptable2.loc[(ptable2[cond1]==1) | (ptable2[cond2]==1)]
                
                ptable2.loc[:,"naive"] = naive_est.predict(ptable2[['const',
                                                                    'Temperature']])
                ptable2.loc[:,"aware"] = aware_est.predict(ptable2[
                    ie.get_feature_list()
                    ])
                
                ptable2.loc[:,"Treatment"] = treat_enc.decode_treatments(ptable2)
                
                
                unscaled_inputs = rescaler.inverse_transform(
                    ptable2[['Temperature', *treat_enc.categories]]
                    )
                ptable2.loc[:, "RawTemp"] = unscaled_inputs[:,0]
                
                colors = seaborn.color_palette('hls', n_colors=len(treat_enc.categories))
                palette = dict(zip(treat_enc.categories, colors))
                
                ax = seaborn.scatterplot(subtable,
                                         x="RawTemp",
                                         y=NORMPROT,
                                         hue="Treatment",
                                         palette=palette)
                
                seaborn.lineplot(ptable2,
                                 x="RawTemp",
                                 y="naive",
                                 color="black",
                                 ax=ax)
                
                seaborn.lineplot(ptable2,
                                 x="RawTemp",
                                 y="aware",
                                 hue="Treatment",
                                 palette=palette,
                                 ax=ax)
                
                pyplot.show()
                
                
            aware_models.append(aware_est)
            naive_models.append(naive_est)
            protein_ids.append(ident[0])
            gene_ids.append(ident[1])
            treatment1.append(cond1)
            treatment2.append(cond2)
    
    sigsquare, d1_est, d2_est = estimate_f_dist(aware_models, naive_models)
    
    pvals = []
    cohen_f2 = []
    pseudo_r2s = []
    
    for aware, naive in zip(aware_models, naive_models):
        pval = f_test(aware, naive, d2=d2_est, d1=d1_est)
        pvals.append(pval)
        rel_effect = (aware.pseudo_rsquared() - naive.pseudo_rsquared()) / (1 - aware.pseudo_rsquared())
        cohen_f2.append(rel_effect)
        pseudo_r2s.append(aware.pseudo_rsquared())
    
    bh_pvals = stats.false_discovery_control(pvals)
    
    stat_table = pandas.DataFrame({
        'PG.ProteinAccessions' : protein_ids,
        'PG.Genes' : gene_ids,
        'Cond1' : treatment1,
        'Cond2' : treatment2,
        'pseudo R2' : pseudo_r2s,
        'Cohen f2' : cohen_f2,
        'pval' : pvals,
        'bh_pval' : bh_pvals
        })
    
    return stat_table
    

if __name__ == '__main__':
    
    data, candidates = load.prepare_data()
    allcondtable = alltreatment_analysis(data, candidates, show=False)
    pairtable = pairwise_modelcomp(data, show=True)
    irrelevant1 = (pairtable['Cond1'] == 'Fisetin') & (pairtable['Cond2'] == 'Quercetin')
    irrelevant2 = (pairtable['Cond1'] == 'Quercetin') & (pairtable['Cond2'] == "Fisetin")
    not_relevant = irrelevant1 | irrelevant2
    relevant = pairtable.loc[~not_relevant,:]
    
    relevant.loc[:,'bh_pval'] = stats.false_discovery_control(relevant['pval'])
    
    cohenf2_thresh = relevant.loc[relevant['Cohen f2'] >= 0.02, :]
    
    bypair = cohenf2_thresh.groupby(by=['Cond1', 'Cond2'])
    pass
    