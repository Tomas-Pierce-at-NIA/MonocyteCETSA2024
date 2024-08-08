# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:15:23 2024

@author: piercetf
"""

# Based on the R script that was provided as a starting file for this project

# and also based very heavily on Childs' NPARC



import itertools
import logging
import os
import multiprocessing

import pandas
import matplotlib
import sympy

# I love having to do this because people can't
# expose their parameters properly
matplotlib.rcParams['legend.loc'] = 'upper right'

import seaborn
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import numpy
from sklearn import preprocessing
from scipy import stats
from scipy import integrate
from scipy import optimize
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
    
    @property
    def resid_pearson(self):
        if self.est is None:
            raise Exception("no residuals because not yet trained")
        else:
            return self.est.resid_pearson
    
    @property
    def resid_response(self):
        if self.est is None:
            raise Exception("no residuals because not yet trained")
        else:
            return self.est.resid_response
    
    @property
    def resid_working(self):
        if self.est is None:
            raise Exception("no residuals because not yet trained")
        else:
            return self.est.resid_working
    
    @property
    def resid_anscombe(self):
        if self.est is None:
            raise Exception("no residuals because not yet trained")
        else:
            return self.est.resid_anscombe
    
    @property
    def resid_anscombe_scaled(self):
        if self.est is None:
            raise Exception("no residuals because not yet trained")
        else:
            return self.est.resid_anscombe_scaled
    
    @property
    def resid_anscombe_unscaled(self):
        if self.est is None:
            raise Exception("no residuals because not yet trained")
        else:
            return self.est.resid_anscombe_unscaled
    
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
        self.reg_est = None
        self.data.loc[:,'ScaledTemp'] = self.scaler.transform(self.data[['Temperature']])
        self.data = sm.add_constant(self.data, prepend=True, has_constant='add')
    
    def fit(self):
        self.model = GLM(self.data[NORMPROT], 
                         self.data[['const', 'ScaledTemp']],
                         family=families.Binomial())
        #self.reg_est = self.model.fit_regularized(alpha=0.1, L1_wt=0.0)
        #self.est = self.model.fit(params=self.reg_est.params)
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
        self.reg_est = None
        self.est = None
    
    def fit(self):
        varnames = self.interact_enc.get_feature_names_out()[:self.iparams]
        self.model = GLM(endog=self.data[NORMPROT],
                         exog=self.data[varnames],
                         family=families.Binomial())
        self.reg_est = self.model.fit_regularized(alpha=0.1, L1_wt=0.0)
        self.est = self.model.fit(params=self.reg_est.params)
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
        self.reg_est = None
        self.est = None
        
        self.cond1 = cond1
        self.cond2 = cond2
        
    
    def fit(self):
        varnames = self.interact_coder.get_feature_names_out()[:self.iparams]
        self.model = GLM(endog=self.data[NORMPROT],
                         exog=self.data[varnames],
                         family=families.Binomial()
                         )
        #self.reg_est = self.model.fit_regularized(alpha=0.1, L1_wt=0.0)
        #self.est = self.model.fit(params=self.reg_est.params)
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
    
    def t_infl(self):
        temp = sympy.Symbol('temperature', real=True)
        cond1 = sympy.Symbol('cond1', real=True)
        cond2 = sympy.Symbol('cond2', real=True)
        wt = sympy.Symbol('weight_temp', real=True)
        wc1 = sympy.Symbol('weight_cond1', real=True)
        wc2 = sympy.Symbol('weight_cond2', real=True)
        wtc1 = sympy.Symbol('weight_temp_cond1', real=True)
        wtc2 = sympy.Symbol('weight_temp_cond2', real=True)
        f = wt*temp + wc1*cond1 + wc2*cond2 + wtc1*temp*cond1 + wtc2*temp*cond2
        _x = sympy.Symbol('x')
        logistic = 1 / (1 + sympy.exp(-_x))
        modelform = logistic.subs(_x, f)
        substitutions = [(wt, self.est.params.iloc[0]),
                            (wc1, self.est.params.iloc[1]),
                            (wc2, self.est.params.iloc[2]),
                            (wtc1, self.est.params.iloc[3]),
                            (wtc2, self.est.params.iloc[4])
                            ]
        
        paramed_model = modelform.subs(substitutions)
        
        cond1_model = paramed_model.subs([(cond1, 1),
                                          (cond2, 0)])
        cond2_model = paramed_model.subs([(cond1, 0),
                                          (cond2, 1)])
        
        cond1_dt2 = cond1_model.diff(temp).diff(temp)
        cond2_dt2 = cond2_model.diff(temp).diff(temp)
        
        cond1_dt2_f = sympy.lambdify(temp, cond1_dt2, 'numpy')
        cond2_dt2_f = sympy.lambdify(temp, cond2_dt2, 'numpy')
        
        cond1_Tincl_res = optimize.minimize_scalar(cond1_dt2_f, bounds=[0,1])
        cond2_Tincl_res = optimize.minimize_scalar(cond2_dt2_f, bounds=[0,1])
        
        scaled_cond1_Tincl = cond1_Tincl_res.x
        scaled_cond2_Tincl = cond2_Tincl_res.x
        
        real_scaleds = self.scaler.inverse_transform([[scaled_cond1_Tincl],
                                                      [scaled_cond2_Tincl]])
        
        cond1_Tincl = real_scaleds[0,0]
        cond2_Tincl = real_scaleds[1,0]
        
        delta_Tincl = cond1_Tincl - cond2_Tincl
        
        return cond1_Tincl, cond2_Tincl, delta_Tincl
    
    def permutation_test(self):
        pass
        

class SimpleDisplayPdf:
    
    def __init__(self,
                 filename: str):
        self.filename = filename
        self.paired_pages = PdfPages(filename, keep_empty=False)
        self._figure = pyplot.figure()
        self._ax = self._figure.add_subplot(111)
        colors = seaborn.color_palette('hls', 4)
        treats = ['DMSO', 'Fisetin', 'Quercetin', 'Myricetin']
        self.palette = dict(zip(treats, colors))
        self.paired_pages_drawn = 0
    
    def close(self):
        self.paired_pages.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    
    def display_pairmodel(self, aware, naive, ptable, hypotable, cond1, cond2, prot_id, shownull=True):
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
        if shownull:
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
        self.paired_pages.savefig(self._ax.get_figure())
        self._ax.cla()
        
        self.paired_pages_drawn += 1
        
        return self.paired_pages_drawn

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
        
        self.paired_pages_drawn = 0
        
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
    
    def display_pairmodel(self, aware, naive, ptable, hypotable, cond1, cond2, prot_id, shownull=True):
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
        if shownull:
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
        
        self.paired_pages_drawn += 1
        
        return self.paired_pages_drawn
    
    def display_unpaired(self, ptable, cond1, cond2, prot_id):
        
        self.unpairq.put((ptable, cond1, cond2, prot_id))
    
    def signal_unpair_complete(self):
        
        self.unpairq.put("Finished")


def _pair_render(pdf_handle, data_queue, palette):
    proc_logger = logging.Logger("_paired_render")
    proc_logger.setLevel(logging.DEBUG)
    

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


def _permutation_test(model, permutations=1000):
    # run permutation test for a single protein
    test_data = model.data.reset_index()
    n = len(test_data)
    base_preds = model.predict(test_data[['Temperature',
                                          'Treatment']])
    base_diffs = test_data[NORMPROT] - base_preds
    base_mse = numpy.sum(base_diffs**2) / n
    
    better_scores = 0
    
    for _ in range(permutations):
        test_data.loc[:, 'Treatment'] = numpy.random.permutation(
            test_data['Treatment']
            )
        preds = model.predict(test_data[['Temperature',
                                         'Treatment']])
        diffs = test_data[NORMPROT] - preds
        mse = numpy.sum(diffs**2) / n
        if mse <= base_mse:
            better_scores += 1
    
    pval = (better_scores + 1) / (permutations + 1)

    return pval

def _bootstrap_test(model, repeats=1000):
    test_data = model.data.reset_index()
    n = len(test_data)
    base_preds = model.predict(test_data[['Temperature',
                                          'Treatment']])
    base_diffs = test_data[NORMPROT] - base_preds
    base_mse = numpy.sum(base_diffs**2) / n
    better_scores = 0
    
    treatments = test_data['Treatment'].drop_duplicates()
    
    for _ in range(repeats):
        test_data.loc[:, "Treatment"] = treatments.sample(
            n = n,
            replace = True
            ).values
        
        preds = model.predict(test_data[['Temperature', 'Treatment']])
        diffs = test_data[NORMPROT] - preds
        mse = numpy.sum(diffs**2) / n
        if mse <= base_mse:
            better_scores += 1
    pval = (better_scores + 1) / (repeats + 1)
    return pval

# based closely on
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html
def permutation_tests(models):

    # run permutation tests, recycling the process pool to relieve
    # the costs of setting it up and tearing it down

    with multiprocessing.Pool(processes=15) as pool:
        pvalues = pool.map(_permutation_test, models)
        
    return pvalues


def bootstrap_tests(models):
    
    with multiprocessing.Pool(processes=15) as pool:
        pvalues = pool.map(_bootstrap_test, models)
    
    return pvalues


def first_n_permutation_tests(models, n=100):
    
    pvals = []
    
    for i in range(n):
        pval = _permutation_test(models[i])
        pvals.append(pval)
        print(pval)
    
    return pvals
        


def estimate_f_dist(aware_models, naive_models):
    """
    Attempts to estimate an empirical F-distribution of the null
    from the collection of treatment-aware and treatment-naive
    models.
    Note that the underlying assumption of NPARC is that an aware model
    compares the effect of a treatment to a vehicle control,
    which means that only models that compare to a vehicle control
    should be used as input to estimating the null distribution,
    and also means that the F-distribution should only be used
    for evaluating the models which compare a treatment to a vehicle control.
    """
    # words cannot describe how much I hate
    # the way that pandas indexes work
    aware_models = numpy.array(aware_models)
    naive_models = numpy.array(naive_models)
    aware_rss = []
    naive_rss = []
    
    for i, aware in enumerate(aware_models):
        naive = naive_models[i]
        if (not aware.converged) or (not naive.converged):
            continue
        a_rss = numpy.sum(aware.resid_response**2)
        aware_rss.append(a_rss)
        n_rss = numpy.sum(naive.resid_response**2)
        naive_rss.append(n_rss)
    
    aware_rss = numpy.array(aware_rss)
    naive_rss = numpy.array(naive_rss)
    
    diffs_rss = naive_rss - aware_rss
    sigsquare = (diffs_rss.var() / diffs_rss.mean()) * 0.5
    
    d1_dist = stats.chi2.fit(diffs_rss)
    d1_est = d1_dist[0] / sigsquare
    
    d2_dist = stats.chi2.fit(aware_rss)
    d2_est = d2_dist[0] / sigsquare
    #breakpoint()
    
    return sigsquare, d1_est, d2_est

    
def f_test(aware_model, naive_model, d2, d1):
    naive_rss = numpy.sum(naive_model.resid_response**2)
    aware_rss = numpy.sum(aware_model.resid_response**2)
    f_stat = (d2 / d1) * (naive_rss - aware_rss) / aware_rss
    # the correctness of the f-test hinges on this line
    pval = stats.f.sf(f_stat, dfn=d1, dfd=d2)
    return pval


def cohen_f2(aware_model, naive_model):
    top = aware_model.pseudo_rsquared() - naive_model.pseudo_rsquared()
    bottom = 1 - aware_model.pseudo_rsquared()
    return top / bottom

def integrate_pair_model(aware_model, cond, ltemp=37, htemp=70):
    def pred_fixed_cond(temp):
        microtable = pandas.DataFrame(data=[[cond, temp]],
                                      columns=['Treatment', 'Temperature']
                                      )
        res = aware_model.predict(microtable)
        return res[0]
    
    return integrate.quad(pred_fixed_cond, ltemp, htemp)

def fit_all_pair_models(data, candidates, min_dpoints=1):
    
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
        treat_lefts = []
        treat_rights = []
        naive_prs = [] # null model R2 approximation
        aware_prs = [] # aware model R2 approximation
        #cond1_integrals = []
        #cond2_integrals = []
        
        #c1S_err = []
        #c2S_err = []
        
        above_physio = []
        i = 0
        #paired_pagenumbers = {}
        #pagenumbers_paired = []
        number_dpoints = []
        
        for protein_id, protein_table in proteingroups:

            print(protein_id)
            pairs = itertools.combinations(treatments, 2)
            #pairs = itertools.product({'DMSO'}, set(treatments) - {'DMSO'} )
            for cond1, cond2 in pairs:
                if (cond1 == 'Fisetin' and cond2 == 'Quercetin') or (cond1 == 'Quercetin' and cond2 == 'Myricetin'):
                    continue
                idx1 = protein_table['Treatment'] == cond1
                idx2 = protein_table['Treatment'] == cond2


                restrict_table = protein_table.loc[idx1|idx2, :]
                
                if len(restrict_table) < min_dpoints:
                    continue # allow user to skip proteins with less than a threshold number of datapoints
                    
                if restrict_table[NORMPROT].max() > 1:
                    not_logit_decay.append((*protein_id, cond1, cond2))
                    continue # skip proteins not suited for analysis by NPARC
                
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
                                         protein_id[1])
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

                    treat_lefts.append(cond1)
                    treat_rights.append(cond2)
                    naive_prs.append(naive_pairmod.pseudo_rsquared())
                    aware_prs.append(aware_pairmod.pseudo_rsquared())
                    

                    
                    above_physio.append(restrict_table[NORMPROT].max() > 1)
                    number_dpoints.append(len(restrict_table))
                    
                except ValueError as ve:
                    fitting_failed.append(protein_id[0])
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
                                         protein_id[1])
                    continue
                
                i = i + 1

    cohen_f2s = []
    for naive, aware in zip(naive_models, aware_models):
        f2_score = cohen_f2(aware, naive)
        cohen_f2s.append(f2_score)

    
    outputtable = pandas.DataFrame({
        'PG.ProteinAccessions' : protein_ids,
        'PG.Genes' : gene_ids,
        'Cohen f2' : cohen_f2s,
        'Treatment1' : treat_lefts,
        'Treatment2' : treat_rights,
        'null psuedo-R2' : naive_prs,
        'alt psuedo-R2' : aware_prs,
        'num_datapoints' : number_dpoints,
        'above physio' : above_physio,
        'naive_models' : naive_models,
        'aware_models' : aware_models
        })
    
    return (outputtable, 
            fitting_failed, 
            unobserved_at_lowest_temp, 
            not_logit_decay)


def main():
    
    data, candidates = load.prepare_data()
    focused_subset = data.loc[:, ['PG.ProteinAccessions',
                                  'PG.Genes',
                                  'R.Replicate',
                                  'Temperature', 
                                  'Treatment', 
                                  NORMPROT]]
    deduplicated_data = focused_subset.drop_duplicates()
    print('data is ready')
    results = fit_all_pair_models(deduplicated_data, candidates, 30)
    outtab, fitfail, unobserv_lowtemp, notlogit = results
    print("models are fitted")
    
    # need this later but cannot calculate after name unbinding
    mintemp = min(deduplicated_data['Temperature'])
    maxtemp = max(deduplicated_data['Temperature'])
    hypodata = hypothetical_inputs(deduplicated_data)
    
    # allow data to be deallocated once no longer directly in use
    # which will hopefully relieve memory pressure
    del data
    del deduplicated_data
    # will need candidates information later
    
    outtab.loc[:, 'ftest_pval'] = -1.0
    
    against_vehicle = outtab[outtab['Treatment1'] == 'DMSO']
    
    sigsq, d1_est, d2_est = estimate_f_dist(against_vehicle['aware_models'],
                                    against_vehicle['naive_models'])
    
    for idx in against_vehicle.index:
        aware_model = against_vehicle.loc[idx, 'aware_models']
        naive_model = against_vehicle.loc[idx, 'naive_models']
        ftest_pval = f_test(aware_model, naive_model, d2_est, d1_est)
        outtab.loc[idx, 'ftest_pval'] = ftest_pval
    
    
    cross_treat = outtab[outtab['Treatment1'] != 'DMSO']
    
    sigsq2, d1_cest, d2_cest = estimate_f_dist(cross_treat['aware_models'],
                                               cross_treat['naive_models'])
    
    for idx in cross_treat.index:
        aware_model = cross_treat.loc[idx, 'aware_models']
        naive_model = cross_treat.loc[idx, 'naive_models']
        ftest_pval = f_test(aware_model, naive_model, d2_cest, d1_cest)
        outtab.loc[idx, 'ftest_pval'] = ftest_pval
    
    outtab.loc[:, 'bh_pval'] = stats.false_discovery_control(outtab['ftest_pval'],
                                                             method='bh')
    
    notlogitframe = pandas.DataFrame(data=notlogit,
                                      columns=['ProteinId',
                                              'GeneId',
                                              'Cond1',
                                              'Cond2'])
    unobs_lowtemp_frame = pandas.DataFrame(data=unobserv_lowtemp,
                                            columns=['ProteinId',
                                                    'GeneId',
                                                    'Cond1',
                                                    'Cond2'])
    userprof = os.environ['USERPROFILE']
    
    sigtable = outtab.loc[outtab['bh_pval'] < ALPHA, :].copy()

    sigtable.loc[:, "pagenum"] = -1.0
    sigtable.loc[:,"cond1_area"] = -1.0
    sigtable.loc[:,"cond2_area"] = -1.0
    with SimpleDisplayPdf(f"{userprof}\\Documents\\pairedsig.pdf") as pdfhand:
        for i in sigtable.index:
            naive = sigtable.loc[i, 'naive_models']
            aware = sigtable.loc[i, 'aware_models']
            pagenum = pdfhand.display_pairmodel(aware, 
                                      naive, 
                                      aware.data, 
                                      hypodata,
                                      aware.cond1,
                                      aware.cond2,
                                      sigtable.loc[i,'PG.Genes'])
            sigtable.loc[i, "pagenum"] = pagenum
    
    sigtable.loc[:, 'cond1_tm'] = -1.0
    sigtable.loc[:, 'cond2_tm'] = -1.0
    sigtable.loc[:, "cond1_Tinfl"] = -1.0
    sigtable.loc[:, "cond2_Tinfl"] = -1.0
    sigtable.loc[:, "delta_Tinfl"] = -1.0
    
    for i in sigtable.index:
        #naive = sigtable.loc[i, 'naive_models']
        aware = sigtable.loc[i, 'aware_models']
        preds = aware.predict(hypodata)
        cond1_preds = preds[hypodata['Treatment'] == aware.cond1]
        cond2_preds = preds[hypodata['Treatment'] == aware.cond2]
        cond1_temps = hypodata.loc[hypodata['Treatment'] == aware.cond1, 'Temperature']
        cond2_temps = hypodata.loc[hypodata['Treatment'] == aware.cond2, 'Temperature']
        cond1_area = integrate.simpson(cond1_preds, x=cond1_temps)
        cond2_area = integrate.simpson(cond2_preds, x=cond2_temps)
        sigtable.loc[i,'cond1_area'] = cond1_area
        sigtable.loc[i,'cond2_area'] = cond2_area
        cond1_tinfl, cond2_tinfl, delta_tinfl = aware.t_infl()
        sigtable.loc[i, 'cond1_Tinfl'] = cond1_tinfl
        sigtable.loc[i, 'cond2_Tinfl'] = cond2_tinfl
        sigtable.loc[i, 'delta_Tinfl'] = delta_tinfl
        
        def cond1pred(temp):
            df = pandas.DataFrame(data=[[aware.cond1, temp]],
                                  columns=['Treatment', 'Temperature'])
            pred = aware.predict(df)
            return pred.loc[0]
        
        def cond2pred(temp):
            df = pandas.DataFrame(data=[[aware.cond2, temp]],
                                  columns=['Treatment', 'Temperature'])
            pred = aware.predict(df)
            return pred.loc[0]
        
        def cond1_tm_diff(temp):
            p = cond1pred(temp)
            return abs(p - 0.5)
        
        def cond2_tm_diff(temp):
            p = cond2pred(temp)
            return abs(p - 0.5)
        
        tempbounds = (mintemp, maxtemp)
        
        cond1_tm_res = optimize.minimize_scalar(cond1_tm_diff, bounds=tempbounds)
        cond2_tm_res = optimize.minimize_scalar(cond2_tm_diff, bounds=tempbounds)
        
        sigtable.loc[i, 'cond1_tm'] = cond1_tm_res.x
        sigtable.loc[i, 'cond2_tm'] = cond2_tm_res.x
    
    sigtable.loc[:, 'permut_pvals'] = permutation_tests(sigtable['aware_models'])
    
    del sigtable['naive_models']
    del sigtable['aware_models']
    del outtab['naive_models']
    del outtab['aware_models']
    
    sigtable['deltaTm'] = sigtable['cond1_tm'] - sigtable['cond2_tm']
    sigtable['deltaS'] = sigtable['cond1_area'] - sigtable['cond2_area']
    sigtable['foldS'] = sigtable['cond1_area'] / sigtable['cond2_area']
    sigtable['logfoldS'] = numpy.log(sigtable['cond1_area']) - numpy.log(sigtable['cond2_area'])
    
    protein_ident = candidates.loc[:,['UniProtIds',
                                      'ProteinNames',
                                      'ProteinDescriptions',
                                      'ProteinGroups']].drop_duplicates()
    
    sigtable = sigtable.merge(protein_ident,
                              how='left',
                              left_on=['PG.ProteinAccessions'],
                              right_on=['UniProtIds'],
                              validate='m:1')
    
    sigtable.to_csv(f"{userprof}\\Documents\\cetsa_nparc_results_signif.csv")
    outtab.to_csv(f'{userprof}\\Documents\\cetsa_nparc_results.csv')
    notlogitframe.to_csv(f"{userprof}\\Documents\\cetsa_nonsigmoid_proteins.csv")
    unobs_lowtemp_frame.to_csv(f"{userprof}\\Documents\\cetsa_no_mintemp.csv")
    with open(f'{userprof}\\Documents\\fitfailed.txt', 'w') as fitfailhandle:
        fitfailhandle.write('Following protein ids were attempted to fit and failed\n')
        fitfailhandle.writelines(fitfail)
    print("done")
    
    return outtab, sigtable

if __name__ == '__main__':
    outtab, sigtable = main()
