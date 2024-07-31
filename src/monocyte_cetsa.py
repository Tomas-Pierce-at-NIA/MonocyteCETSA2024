# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:15:23 2024

@author: piercetf
"""

# Based on the R script that was provided as a starting file for this project

# and also based very heavily on Childs' NPARC

import cProfile

import itertools
import logging
import os
import multiprocessing

import pandas

import matplotlib

# I love having to do this because people can't
# expose their parameters properly
matplotlib.rcParams['legend.loc'] = 'upper right'

import seaborn
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import numpy
from sklearn import preprocessing
from scipy import stats
import statsmodels.api as sm
from statsmodels.api import GLM, families
import load_monocyte_cetsa_data as load


ALPHA = 0.05
EPSILON = 1e-9
R_THRESHOLD = 0.6
F2_THRESHOLD = 0.35 # Cohen f**2 large effect size criterion
TEMP = "Temperature"
NORMPROT = "Normalized_FG_Quantity"

CEASE_TOKEN = "Finished"

def hypothetical_inputs(dataset):
    treatments = list(dataset['Treatment'].unique())
    t_max = dataset['Temperature'].max()
    t_min = dataset['Temperature'].min()
    span = numpy.arange(t_min, t_max + 1)
    temp_span = numpy.repeat(span, len(treatments))
    treatarr = numpy.array(treatments * ((t_max + 1) - t_min))
    dframe = pandas.DataFrame({
        'Temperature' : temp_span,
        'Treatment' : treatarr
        })
    return dframe

class TreatmentEncoder:
    """"
    paper thin wrapper around sklearn.preprocessing.OneHotEncoder
    to handle the specific ways in which it needs to be invoked
    and used to reshape our data
    """
    
    def __init__(self, table):
        
        self.encoder = preprocessing.OneHotEncoder(sparse_output=False,
                                                   dtype=numpy.int32)
                                                   #drop='first')
        
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


class ProteinModel:
    
    @property
    def resid_deviance(self):
        if self.est is None:
            raise Exception("no residuals because not yet trained")
        else:
            return self.est.resid_deviance
    
    def pseudo_rsquared(self):
        if self.est is None:
            raise Exception("No R-squared because not yet trained")
        else:
            return self.est.pseudo_rsquared()


class NaiveModel(ProteinModel):
    
    def __init__(self, subtable :pandas.DataFrame, scaler :preprocessing.MinMaxScaler):
        
        self.scaler = scaler
        subtable = subtable.loc[:, [NORMPROT, 'Temperature', 'Treatment']]
        subtable = subtable.dropna()
        self.data = subtable.loc[:,:]
        self.model = None
        self.est = None
        self.data.loc[:,'ScaledTemp'] = self.scaler.transform(self.data[['Temperature']])
        self.data = sm.add_constant(self.data, prepend=True, has_constant='add')
    
    def fit(self):
        self.model = GLM(self.data[NORMPROT], 
                         self.data[['const', 'ScaledTemp']],
                         family=families.Binomial())
        self.est = self.model.fit()
        return self.est
    
    def predict(self, inputdata :pandas.DataFrame):
        indata = inputdata.assign(
            ScaledTemp = self.scaler.transform(inputdata[['Temperature']])
            )
        indata = sm.add_constant(indata, prepend=True, has_constant='add')
        predictions = self.est.predict(indata[['const', 'ScaledTemp']])
        return predictions
    
    @property
    def converged(self):
        if self.est is None:
            return False
        else:
            return self.est.converged
    
    @property
    def resid_pearson(self):
        if self.est is None:
            raise Exception("no residuals because not yet trained")
        else:
            return self.est.resid_pearson


class AwareModel(ProteinModel):
    
    def __init__(self, subtable :pandas.DataFrame, treatment_coder :TreatmentEncoder, scaler :preprocessing.MinMaxScaler):
        self.scaler = scaler
        self.treat_coder = treatment_coder
        subtable = subtable.loc[:, [NORMPROT, 'Temperature', 'Treatment']]
        subtable = subtable.dropna()
        self.data = treatment_coder.encode_treatments(subtable)
        self.data.loc[:, "ScaledTemp"] = scaler.transform(subtable[['Temperature']])
        self.iparams = treatment_coder.n_iparams
        categories = treatment_coder.categories
        self.categories = categories
        self.interact_enc = preprocessing.PolynomialFeatures(interaction_only=True,
                                                             include_bias=False)
        self.interact_enc.fit(self.data[['ScaledTemp', *categories]])
        interacting = self.interact_enc.transform(self.data[['ScaledTemp', *categories]])
        interact_feats = self.interact_enc.get_feature_names_out()
        self.data.loc[:, interact_feats[:self.iparams]] = interacting[:, :self.iparams]
        
        self.model = None
        self.est = None
    
    def fit(self):
        varnames = self.interact_enc.get_feature_names_out()[:self.iparams]
        self.model = GLM(endog=self.data[NORMPROT],
                         exog=self.data[varnames],
                         family=families.Binomial())
        self.est = self.model.fit()
        return self.est
    
    def predict(self, inputdata :pandas.DataFrame):
        indata = self.treat_coder.encode_treatments(inputdata)
        scaledin = indata.assign(
            ScaledTemp = self.scaler.transform(inputdata[['Temperature']])
            )
        interacting = self.interact_enc.transform(scaledin[['ScaledTemp', *self.categories]])
        interact_feats = self.interact_enc.get_feature_names_out()[:self.iparams]
        modelinputs = pandas.DataFrame(data=interacting[:,:self.iparams],
                                       columns=interact_feats)
        preds = self.est.predict(modelinputs)
        return preds
    
    @property
    def converged(self):
        if self.est is None:
            return False
        else:
            return self.est.converged
    
    @property
    def resid_pearson(self):
        if self.est is None:
            raise Exception("no residuals because not yet trained")
        else:
            return self.est.resid_pearson


class NaivePairModel(NaiveModel):
    
    def __init__(self, subtable, scaler, cond1, cond2):
        idx1 = (subtable['Treatment'] == cond1)
        idx2 = (subtable['Treatment'] == cond2)
        restrictedtable = subtable.loc[idx1 | idx2, :]
        super().__init__(restrictedtable, scaler)
        self.cond1 = cond1
        self.cond2 = cond2

class AwarePairModel(ProteinModel):
    
    def __init__(self, subtable, treatment_coder, scaler, cond1, cond2):
        idx1 = subtable['Treatment'] == cond1
        idx2 = subtable['Treatment'] == cond2
        restrictedtable = subtable.loc[idx1 | idx2, :]
        subtable = restrictedtable.loc[:, ["Temperature", "Treatment", NORMPROT]]
        subtable = subtable.dropna()
        self.treat_coder = treatment_coder
        self.scaler = scaler
        self.cond1 = cond1
        self.cond2 = cond2
        self.interact_coder = preprocessing.PolynomialFeatures(interaction_only=True,
                                                               include_bias=False)
        s_table = treatment_coder.encode_treatments(subtable)
        s_table.loc[:, "ScaledTemp"] = scaler.transform(s_table[['Temperature']])
        indep_varnames = ['ScaledTemp', cond1, cond2]
        self.interact_coder.fit(s_table[indep_varnames])
        interact_arr = self.interact_coder.transform(s_table[indep_varnames])
        interact_feats = self.interact_coder.get_feature_names_out()
        s_table.loc[:, interact_feats[:-1]] = interact_arr[:,:-1]
        self.data = s_table
        self.categories = [cond1, cond2]
        self.iparams = len(interact_feats) - 1
        self.treat_coder = treatment_coder
        
        self.model = None
        self.est = None
        
        self.cond1 = cond1
        self.cond2 = cond2
    
    def fit(self):
        varnames = self.interact_coder.get_feature_names_out()[:self.iparams]
        self.model = GLM(endog=self.data[NORMPROT],
                         exog=self.data[varnames],
                         family=families.Binomial()
                         )
        self.est = self.model.fit()
        return self.est
    
    def predict(self, indata):
        scaledindata = indata.assign(
            ScaledTemp = self.scaler.transform(indata[['Temperature']])
            )
        scaledindata = self.treat_coder.encode_treatments(scaledindata)
        indepvarnames = ['ScaledTemp', self.cond1, self.cond2]
        interact_arr = self.interact_coder.transform(scaledindata[indepvarnames])
        interact_feats = self.interact_coder.get_feature_names_out()
        scaledindata.loc[:, interact_feats[:-1]] = interact_arr[:, :-1]
        modelinputs = pandas.DataFrame(
            data = interact_arr[:, :-1],
            columns=interact_feats[:-1]
            )
        preds = self.est.predict(modelinputs)
        return preds
    
    @property
    def converged(self):
        if self.est is None:
            return False
        else:
            return self.est.converged
    
    @property
    def resid_pearson(self):
        if self.est is None:
            raise Exception("no residuals because not yet trained")
        else:
            return self.est.resid_pearson


class ProteinDisplayPdf:
    
    def __init__(self,
                 sharedmodel_filename :str,
                 pairmodels_filename :str,
                 un_sharedmodeled_filename :str,
                 un_pairmodeled_filename :str):
        
        self.sharedmodel_fname = sharedmodel_filename
        self.pairmodel_fname = pairmodels_filename
        self.unshare_fname = un_sharedmodeled_filename
        self.unpair_fname = un_pairmodeled_filename
        
        self.share_pages = PdfPages(sharedmodel_filename, keep_empty=False)
        self.pair_pages = PdfPages(pairmodels_filename, keep_empty=False)
        self.noshared_pages = PdfPages(un_sharedmodeled_filename, keep_empty=False)
        self.unpaired_pages = PdfPages(un_pairmodeled_filename, keep_empty=False)
        
        self._figure = pyplot.figure()
        self._ax = self._figure.add_subplot(111)
        
        colors = seaborn.color_palette('hls', 4)
        treats = ['DMSO', 'Fisetin', 'Quercetin', 'Myricetin']
        self.palette = dict(zip(treats, colors))
        
        self.manager = multiprocessing.Manager()
        
        self.unpairq = self.manager.Queue()
        
        self.unpair_handleproc = multiprocessing.Process(
            target=_unpair_render2,
            args=(self.unpaired_pages, self.unpairq, self.palette)
            )
        
        self.unpair_handleproc.start()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def close(self):
        self.share_pages.close()
        self.pair_pages.close()
        self.noshared_pages.close()
        self.signal_unpair_complete()
        self.unpair_handleproc.join()
        #self.unpairq.close()
        self.unpaired_pages.close()
    
    def display_pairmodel(self, aware, naive, ptable, hypotable, cond1, cond2, prot_id):
        idx1 = hypotable['Treatment'] == cond1
        idx2 = hypotable['Treatment'] == cond2
        idx = idx1 | idx2
        subtable = hypotable.loc[idx,:]
        subtable = subtable.reset_index()
        aware_preds = aware.predict(subtable)
        naive_preds = naive.predict(subtable)
        
        subtable.loc[:,"Aware"] = aware_preds
        subtable.loc[:,"Naive"] = naive_preds
        
        #figure = pyplot.figure()
        #ax = figure.add_subplot(111)
        seaborn.scatterplot(ptable,
                            x='Temperature',
                            y=NORMPROT,
                            hue='Treatment',
                            ax=self._ax,
                            palette=self.palette
            )
        seaborn.lineplot(subtable,
                         x='Temperature',
                         y='Naive',
                         color='black',
                         ax=self._ax,
                         estimator=None,
                         n_boot=0,
                         errorbar=None
                         )
        seaborn.lineplot(subtable,
                         x='Temperature',
                         y='Aware',
                         hue='Treatment',
                         ax=self._ax,
                         estimator=None,
                         n_boot=0,
                         errorbar=None,
                         palette=self.palette
                         )
        self._ax.set_title(f"{prot_id}_{cond1}_{cond2}_pairmodel")
        self.pair_pages.savefig(self._ax.get_figure())
        self._ax.cla()
    
    def display_unpaired(self, ptable, cond1, cond2, prot_id):
        
        self.unpairq.put((ptable, cond1, cond2, prot_id))
    
    def signal_unpair_complete(self):
        
        self.unpairq.put("Finished")

def _unpair_render2(pdf_handle, data_queue, palette):
    proc_logger = logging.Logger("_unpair_render")
    proc_logger.setLevel(logging.DEBUG)
    import os
    userprof = os.environ['USERPROFILE']
    filehand = logging.FileHandler(f'{userprof}\\Documents\\unpair.log')
    proc_logger.addHandler(filehand)
    to_show = data_queue.get()
    while to_show != "Finished":
        try:  
            ptable, cond1, cond2, prot_id = to_show
            proc_logger.info(f"attempting to graph {prot_id} for {cond1} and {cond2}")
            if len(ptable) == 0:
                proc_logger.info(f"{prot_id} has no data")
                to_show = data_queue.get()
                continue
            # elif len(ptable.dropna()) == 0:
            #     proc_logger.info(f"{prot_id} has no non-null data")
            #     proc_logger.info(f"{ptable.columns}")
            #     proc_logger.info(f"{len(ptable.index)}")
            
            ax = seaborn.scatterplot(ptable,
                                x='Temperature',
                                y=NORMPROT,
                                hue='Treatment'
                                )
            ax.set_title(f"{prot_id}_{cond1}_{cond2}_nopair")
            pdf_handle.savefig(ax.get_figure())
            pyplot.close(ax.get_figure())
            to_show = data_queue.get()
            
            proc_logger.info(f"Attempted plotting {prot_id}")
            proc_logger.debug(f"ptable of shape rows={len(ptable.index)}")
            proc_logger.debug(f"and cols={len(ptable.columns)}")
        
        except Exception as e:
            proc_logger.exception("Failure on unpaired rendering",
                                  exc_info=e)
            raise e
    pdf_handle.close()
            

def interact_param_count(base_params: int) -> int:
    return base_params + base_params - 1


def estimate_f_dist(aware_models, naive_models):
    """
    Estimates the empirical f-distribution from the alterate and null
    models fit to each protein in the data

    Parameters
    ----------
    aware_models : list
        collection of models aware of treatment group distinctions
    naive_models : list
        collection of models not aware of treatment group distinctions

    Raises
    ------
    Exception
        when either there are no aware models, no naive models, or no converged
        null models

    Returns
    -------
    sig_square : float
        sigma squared parameter of estimated f-distribution
    d1_est : float
        d1 parameter estimate for empirical f-distribution
    d2_est : float
        d2 parameter estimate for empirical f-distribution

    """
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

def f_test(aware_model, naive_model, d2, d1):
    "Run an F-test comparing a null (naive) model and a treatment-aware model"
    alt_rss = sum(aware_model.resid_pearson ** 2)
    null_rss = sum(naive_model.resid_pearson ** 2)
    f_stat = (d2 / d1) * ((null_rss - alt_rss) / alt_rss)
    pval = stats.f.sf(f_stat, d1, d2)
    return pval

def cohen_f2(aware_model, naive_model):
    top = aware_model.pseudo_rsquared() - naive_model.pseudo_rsquared()
    bottom = 1 - aware_model.pseudo_rsquared()
    return top / bottom

def main(data, candidates):
    
    controltemp = data['Temperature'].min()
    
    logging.basicConfig(filename="{}\\Documents\\tomas_cetsa.log".format(
        os.environ['USERPROFILE']), 
                        encoding='utf-8', 
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    userprofile = os.environ['USERPROFILE']
    
    with ProteinDisplayPdf(f"{userprofile}\\Documents\\sharedmodel.pdf",
                           f"{userprofile}\\Documents\\pairedmodel.pdf",
                           f"{userprofile}\\Documents\\unshared.pdf",
                           f"{userprofile}\\Documents\\unshared.pdf"
                           ) as pdf:
        hypothetical = hypothetical_inputs(data)
        proteingroups = data.groupby(by=['PG.ProteinAccessions', 'PG.Genes'])
        treatments = data['Treatment'].unique()
        minmax = preprocessing.MinMaxScaler()
        minmax.fit(data[['Temperature']])
        treat_coder = TreatmentEncoder(data)
        protein_ids = []
        gene_ids = []
        pair_ids = []
        naive_models = []
        aware_models = []
        fitting_failed = []
        unobserved_at_lowest_temp = []
        not_logit_decay = []
        i = 0
        for protein_id, protein_table in proteingroups:

            print(protein_id)
            pairs = itertools.combinations(treatments, 2)
            for cond1, cond2 in pairs:
                idx1 = protein_table['Treatment'] == cond1
                idx2 = protein_table['Treatment'] == cond2
                hypo_idx1 = hypothetical['Treatment'] == cond1
                hypo_idx2 = hypothetical['Treatment'] == cond2
                restrict_hypo = hypothetical.loc[hypo_idx1 | hypo_idx2, :]
                restrict_table = protein_table.loc[idx1|idx2, :]
                
                cond1_tmin = protein_table.loc[idx1, 'Temperature'].min()
                cond2_tmin = protein_table.loc[idx2, 'Temperature'].min()
                
                if cond1_tmin > controltemp or cond2_tmin > controltemp:
                    logger.info(
                        f"""
                        fitting for {protein_id} is not possible because
                        relationship to abundance at control temperature
                        is unknown because {protein_id} is not observed
                        at control temperature.
                        min temp for {cond1} on {protein_id} was {cond1_tmin}
                        min temp for {cond2} on {protein_id} was {cond2_tmin}
                        """
                        )
                    #fitting_failed.append(protein_id[0])
                    unobserved_at_lowest_temp.append((*protein_id, cond1, cond2))
                    focusedres = restrict_table.loc[:, ['Temperature',
                                                        NORMPROT,
                                                        'Treatment']].copy()
                    pdf.display_unpaired(focusedres,
                                         cond1,
                                         cond2,
                                         protein_id[0])
                    continue
                
                if restrict_table[NORMPROT].max() > 1:
                    cond1_idx = restrict_table['Treatment'] == cond1
                    cond2_idx = restrict_table['Treatment'] == cond2
                    cond1_max_prot = restrict_table.loc[cond1_idx, NORMPROT].max()
                    cond2_max_prot = restrict_table.loc[cond2_idx, NORMPROT].max()
                    logger.info(f"""{protein_id} max protein exceeds physiological
                                when comparing conditions {cond1} having {cond1_max_prot}
                                and {cond2} having {cond2_max_prot}
                                """)
                    not_logit_decay.append((*protein_id, cond1, cond2))
                    focusedres = restrict_table.loc[:, ['Temperature',
                                                        NORMPROT,
                                                        'Treatment']].copy()
                    pdf.display_unpaired(focusedres,
                                         cond1,
                                         cond2,
                                         protein_id[0])
                    continue
                    
                try:
                    naive_pairmod = NaivePairModel(restrict_table, 
                                                   minmax, 
                                                   cond1, 
                                                   cond2)
                    naive_pairmod.fit()
                    aware_pairmod = AwarePairModel(restrict_table, 
                                                   treat_coder, 
                                                   minmax, 
                                                   cond1, 
                                                   cond2)
                    aware_pairmod.fit()
                    protein_ids.append(protein_id[0])
                    gene_ids.append(protein_id[1])
                    pair_ids.append((cond1, cond2))
                    naive_models.append(naive_pairmod)
                    aware_models.append(aware_pairmod)
                    pdf.display_pairmodel(aware_pairmod,
                                          naive_pairmod,
                                          restrict_table,
                                          restrict_hypo,
                                          cond1,
                                          cond2,
                                          protein_id[0])
                except ValueError as ve:
                    fitting_failed.append(protein_id[0])
                    logger.debug("fitting on {} failed".format(protein_id[0]),
                                  exc_info=ve)
                    logger.info(
                        f"""min temp for {cond1} on {protein_id} was {cond1_tmin}
                        min temp for {cond2} on {protein_id} was {cond2_tmin}
                        """
                        )
                    pdf.display_unpaired(restrict_table,
                                         cond1,
                                         cond2,
                                         protein_id[0])
                
                i = i + 1
                    
        
    sigsq, d1_est, d2_est = estimate_f_dist(aware_models,
                                                naive_models)
        
    pvals = []
    cohen_f2s = []
    for naive, aware in zip(naive_models, aware_models):
        pval = f_test(aware, naive, d2=d2_est, d1=d1_est)
        pvals.append(pval)
        f2_score = cohen_f2(aware, naive)
        cohen_f2s.append(f2_score)
    
    corrected_pvals = stats.false_discovery_control(pvals, method='bh')
    
    outputtable = pandas.DataFrame({
        'PG.ProteinAccessions' : protein_ids,
        'PG.Genes' : gene_ids,
        'Cohen f2' : cohen_f2s,
        'pval' : pvals,
        'bh_pval' : corrected_pvals
        })
    
    return outputtable, fitting_failed, unobserved_at_lowest_temp, not_logit_decay

if __name__ == '__main__':
    data, candidates = load.prepare_data()
    results = main(data, candidates)