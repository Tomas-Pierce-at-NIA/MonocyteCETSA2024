# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:08:11 2024

@author: piercetf
"""

import pandas
import load_monocyte_cetsa_data as load
from scipy import integrate, stats

P_REL_QUANT = 'Normalized_FG_Quantity'
ALPHA = 0.05

def split_treatments(dframe):
    treatments = list(dframe.loc[:, 'Treatment'].drop_duplicates())
    treatframes = []
    for treat in treatments:
        subframe = dframe.loc[dframe['Treatment'] == treat, :]
        treatframes.append(subframe)
    return [treatframes, treatments]


def get_abc(diff_frame):
    "get area between curves"
    integral = integrate.simpson(y=diff_frame[['diffs']].array,
                                 x=diff_frame[['Temperature']].array)
    return integral

def get_treatment(table :pandas.DataFrame, condition :str):
    return table.loc[table['Treatment'] == condition, :]


def make_comparisons(dmso, fisetin, myricetin, quercetin):
    dmso_vals = dmso.loc[:, P_REL_QUANT]
    fis_vals = fisetin.loc[:, P_REL_QUANT]
    myr_vals = myricetin.loc[:, P_REL_QUANT]
    quer_vals = quercetin.loc[:, P_REL_QUANT]
    
    
    dvf_u = stats.mannwhitneyu(dmso_vals, fis_vals)
    dvq_u = stats.mannwhitneyu(dmso_vals, quer_vals)
    dvm_u = stats.mannwhitneyu(dmso_vals, myr_vals)
    mvf_u = stats.mannwhitneyu(myr_vals, fis_vals)
    mvq_u = stats.mannwhitneyu(myr_vals, quer_vals)
    
    try:
        kruskal = stats.kruskal(dmso_vals, fis_vals, myr_vals, quer_vals)
    except:
        breakpoint()
    
    test_pvals = [kruskal.pvalue,
                  dvf_u.pvalue,
                  dvq_u.pvalue,
                  dvm_u.pvalue,
                  mvf_u.pvalue,
                  mvq_u.pvalue
                  ]
    
    return test_pvals

def cmp_prot_temp(dataframe :pandas.DataFrame):
    "make separate comparisons for each protein at each temperature"
    groups = dataframe.groupby(by=['PG.ProteinAccessions', 'Temperature'])
    comparisontable = []
    for idtuple, group in groups:
        prot, temp = idtuple
        if temp == 37:
            # all values are normalized against the 37 C condition
            # which means that 
            # A) the measurements are all the same, which is numerically bad
            # and also 
            # B) the values at that temperature are not biologically interesting
            # because they've been normalized to be the same
            continue
        dmso = get_treatment(group, 'DMSO')
        fisetin = get_treatment(group, 'Fisetin')
        myricetin = get_treatment(group, 'Myricetin')
        quercetin = get_treatment(group, 'Quercetin')
        test_pvals = make_comparisons(dmso=dmso,
                                       fisetin=fisetin,
                                       myricetin=myricetin,
                                       quercetin=quercetin)
        row = [prot, temp, *test_pvals]
        comparisontable.append(row)
    
    comp_frame = pandas.DataFrame(data=comparisontable,
                                  columns=['PG.ProteinAccessions',
                                           'Temperature',
                                           'Kruskal p-val',
                                           'U-rank test (DMSO v Fisetin)',
                                           'U-rank test (DMSO v Quercetin)',
                                           'U-rank test (DMSO v Myricetin)',
                                           'U-rank test (Myricetin v Fisetin)',
                                           'U-rank test (Myricetin v Quercetin)'
                                           ]
                                  )
    
    return comp_frame


def make_deltas(dmso, fisetin, myricetin, quercetin):
    "calculate differences for a specific condition (caller sets up)"
    
    # I could spend a bunch of time getting things to line up
    # and taking the average of the differences
    # or I can just take the means of the target populations
    # and report their differences,
    # and have an easier time
    
    dmso_avg = dmso[P_REL_QUANT].mean()
    fisetin_avg = fisetin[P_REL_QUANT].mean()
    myricetin_avg = myricetin[P_REL_QUANT].mean()
    quercetin_avg = quercetin[P_REL_QUANT].mean()
    
    diff_f_d = fisetin_avg - dmso_avg
    diff_q_d = quercetin_avg - dmso_avg
    diff_m_d = myricetin_avg - dmso_avg
    
    diff_f_m = fisetin_avg - myricetin_avg
    diff_q_m = quercetin_avg - myricetin_avg
    
    return [diff_f_d, diff_q_d, diff_m_d, diff_f_m, diff_q_m]


def delta_prot_temp(dataframe :pandas.DataFrame):
    "measure differences between conditions for each protein at each temperature"
    groups = dataframe.groupby(by=['PG.ProteinAccessions', 'Temperature'])
    differencetable = []
    for idtuple, group in groups:
        prot, temp = idtuple
        if temp == 37:
            # difference will be zero in this case for all conditions
            continue
        dmso = get_treatment(group, 'DMSO')
        fisetin = get_treatment(group, 'Fisetin')
        myricetin = get_treatment(group, 'Myricetin')
        quercetin = get_treatment(group, 'Quercetin')
        diffs = make_deltas(dmso=dmso, 
                            fisetin=fisetin, 
                            myricetin=myricetin,
                            quercetin=quercetin
                            )
        row = [prot, temp, *diffs]
        differencetable.append(row)
        
    delta_frame = pandas.DataFrame(data=differencetable,
                                   columns=['PG.ProteinAccessions',
                                            'Temperature',
                                            'Fisetin v DMSO',
                                            'Quercetin v DMSO',
                                            'Myricetin v DMSO',
                                            'Fisetin v Myricetin',
                                            'Quercetin v Myricetin'])
    
    return delta_frame
    

def get_counts(datatable :pandas.DataFrame, alpha=0.05):
    prot_groups = datatable.groupby(by=['PG.ProteinAccessions'])
    colnames = datatable.columns
    measure_cols = colnames.difference(['PG.ProteinAccessions', 'Temperature'])
    table = []
    for idtuple, prot in prot_groups:
        isbelow = prot.loc[:, measure_cols] < 0.05
        belowcount = isbelow.sum()
        row = [*idtuple, *belowcount]
        table.append(row)
    count_frame = pandas.DataFrame(data=table,
                                   columns=['PG.ProteinAccessions',
                                            *measure_cols])
    return count_frame
    

def main(alpha=0.05):
    data, cand = load.prepare_data()
    subdata = data.loc[:, ['PG.ProteinAccessions',
                           'R.Replicate', 
                           'Temperature', 
                           'Treatment', 
                           'Normalized_FG_Quantity']]
    subdata = subdata.drop_duplicates()
    
    differences = delta_prot_temp(subdata)
    differences = differences.dropna()
    
    comparisons = cmp_prot_temp(subdata)
    
    comparisons = comparisons.dropna()
    
    fulltable = comparisons.merge(differences,
                      on=['PG.ProteinAccessions', 'Temperature'],
                      how='inner')
    
    count_frame = get_counts(comparisons, alpha=alpha)
    
    return count_frame, fulltable 


if __name__ == '__main__':
    counts, full = main(alpha=ALPHA)
    full = full.assign(kruskal_low = full['Kruskal p-val'] < ALPHA)
    full = full.assign(fis_v_dmso_low = full['U-rank test (DMSO v Fisetin)'] < ALPHA)
    full = full.assign(quer_v_dmso_low = full['U-rank test (DMSO v Quercetin)'] < ALPHA)
    full = full.assign(myr_v_dmso_low = full['U-rank test (DMSO v Myricetin)'] < ALPHA)
    full = full.assign(fis_v_myr_low = full['U-rank test (Myricetin v Fisetin)'] < ALPHA)
    full = full.assign(quer_v_myr_low = full['U-rank test (Myricetin v Quercetin)'] < ALPHA)
    
    