import gc

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from time import time
from tqdm import tqdm

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfpd = tfp.distributions

def reduce_mdn_pars(mdn_pars, n_mc):

    mus, sigs, pis = tf.split(mdn_pars, 3, axis=1)
    mus = tf.concat(tf.split(mus, n_mc, axis=0), axis=1)
    sigs = tf.concat(tf.split(sigs, n_mc, axis=0), axis=1)
    pis = tf.concat(tf.split(pis, n_mc, axis=0), axis=1)

    mdn_pars = tf.concat([mus, sigs, pis], axis=1)

    return mdn_pars

def generateGaussianMixtures(mdn_pars, n_mix):

    mixtureDists = [0]*len(mdn_pars)
    for i, pars in enumerate(mdn_pars):
        mixtureDists[i] = tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(logits=pars[2*n_mix:]),
            components_distribution=tfpd.Normal(
                loc=pars[:n_mix], scale=pars[n_mix:2*n_mix]
            )
        )
    return mixtureDists

def generateLogNormalMixture(logMdnPars, n_mix):

    mixtureDists = [0]*len(logMdnPars)
    for i, pars in enumerate(logMdnPars):
        mixtureDists[i] = tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(logits=pars[2*n_mix:]),
            components_distribution=tfpd.LogNormal(
                loc=pars[:n_mix], scale=pars[n_mix:2*n_mix]
            )
        )
    return mixtureDists

def binarysearch_inverse(f, delta = 1e-10):
    def f_1(y):
        lo, hi = find_bounds(f, y)
        return binary_search(f, y, lo, hi, delta)
    return f_1

def find_bounds(f, y):
    x = 1
    while f(x) < y:
        x *= 2.
    lo = -20 if (x == 1) else x/2
    return lo, x

def binary_search(f, y, lo, hi, delta):
    while lo <= hi:
        x = (lo + hi) / 2.
        if f(x) < y:
            lo = x + delta
        elif f(x) > y:
            hi = x - delta
        else:
            return x
    return hi if (f(hi)-y < y-f(lo)) else lo

def calculateStatistics(listOfDists, delta=1e-10):

    n_sources = len(listOfDists)
    med_low_upp = np.empty((n_sources, 7))
    
    for i, dist in tqdm(enumerate(listOfDists)):
        cdf = lambda x: dist.cdf(x).numpy()
        quantile_func_fit = binarysearch_inverse(cdf, delta)
        med_low_upp[i,0] = quantile_func_fit(0.5)
        med_low_upp[i,1] = quantile_func_fit(0.16)
        med_low_upp[i,2] = quantile_func_fit(0.84)
        med_low_upp[i,3] = quantile_func_fit(0.0015)
        med_low_upp[i,4] = quantile_func_fit(0.9985)
        med_low_upp[i,5] = dist.mean().numpy()
        med_low_upp[i,6] = dist.stddev().numpy()

    return med_low_upp

class Metrics(tfk.callbacks.Callback):

    def __init__(
        self, val_data, targets, indices, n_mix,
        norm_mean, norm_std, path, n_samples=100,
        precision=1e-10, is_ss = False
    ):
        super().__init__()
        self.validation_data = val_data
        self.targets = targets
        self.z_targets = np.exp(targets)
        self.indices = indices
        self.n_mix = n_mix
        self.n_samples = n_samples
        self.n_z = 500
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.path = path
        self.is_ss = is_ss
        self.precision = precision

    def on_train_begin(self, logs={}):
        self.pdf = []
        self.pdf_mc = []
        self.scaled_pdf = []
        self.log_rms = []
        self.log_rms_mc = []
        self.med_low_upp = []

    def on_train_end(self, logs={}):
        t0 = time()
        self.n_samples = 100
        if not self.is_ss:
            mdn_pars_mc = self.model(self.validation_data, self.n_samples).numpy()
        else:
            _, _, _, _, _, _, mdn_pars_mc, _ = self.model(self.validation_data, self.n_samples)
            mdn_pars_mc = mdn_pars_mc.numpy()
        reduced_mdn_pars = reduce_mdn_pars(mdn_pars_mc, self.n_samples).numpy()
        reduced_mdn_pars[:, :self.n_mix * self.n_samples] = reduced_mdn_pars[:, :self.n_mix * self.n_samples] * self.norm_std + self.norm_mean
        reduced_mdn_pars[:, self.n_mix * self.n_samples:2*self.n_mix * self.n_samples] = reduced_mdn_pars[:, self.n_mix * self.n_samples:2*self.n_mix * self.n_samples] * self.norm_std
        lgms = generateLogNormalMixture(reduced_mdn_pars, self.n_mix * self.n_samples)
        self.med_low_upp = calculateStatistics(lgms, delta=self.precision)

        idx_outliers = (self.z_targets < self.med_low_upp[:, 3]) | (self.z_targets > self.med_low_upp[:, 4])
        idx_1sig = (self.z_targets >= self.med_low_upp[:,1]) & (self.z_targets <= self.med_low_upp[:, 2])
        percent_outliers = np.count_nonzero(idx_outliers) / len(self.z_targets) * 100
        percent_1sig = np.count_nonzero(idx_1sig) / len(self.z_targets) * 100
        scatter = np.sqrt(
            np.mean(
                (
                    (self.med_low_upp[:,2] - self.med_low_upp[:,1]) / (2 * self.med_low_upp[:,0])
                )**2
            )
        )
        bias = np.abs((self.med_low_upp[:,0] - self.z_targets) / (1 + self.z_targets))
        rms_bias = np.sqrt(
            np.mean(
                (
                    (self.med_low_upp[:,0] - self.z_targets) / (1 + self.z_targets)
                )**2
            )
        )
        idx_alt_outliers = bias > 3 * rms_bias
        percent_alt_outliers = np.count_nonzero(idx_alt_outliers) / len(self.z_targets) * 100
        
        metrics = {
            'Outlier Percentage': percent_outliers,
            'Alternative Outlier Percentage': percent_alt_outliers,
            'Percentage within 1 sigma': percent_1sig,
            'RMS Scatter': scatter,
            'RMS Bias': rms_bias
        }
        self.metrics = pd.DataFrame(metrics, index=[0])
        print(f"Time for end of train metrics: {time()-t0} s")

    def on_epoch_end(self, epoch, logs={}):
        t0 = time()
        if not self.is_ss:
            mdn_pars = self.model(self.validation_data).numpy()
            mdn_pars_mc = self.model(self.validation_data, self.n_samples).numpy()
        else:
            _, _, _, _, _, _, mdn_pars, _ = self.model(self.validation_data)
            mdn_pars = mdn_pars.numpy()
            _, _, _, _, _, _, mdn_pars_mc, _ = self.model(self.validation_data, self.n_samples)
            mdn_pars_mc = mdn_pars_mc.numpy()
        
        reduced_mdn_pars = reduce_mdn_pars(mdn_pars_mc, self.n_samples).numpy()

        mdn_pars[:, :self.n_mix] = mdn_pars[:, :self.n_mix] * self.norm_std + self.norm_mean
        mdn_pars[:, self.n_mix:2*self.n_mix] = mdn_pars[:, self.n_mix:2*self.n_mix] * self.norm_std
        reduced_mdn_pars[:, :self.n_mix * self.n_samples] = reduced_mdn_pars[:, :self.n_mix * self.n_samples] * self.norm_std + self.norm_mean
        reduced_mdn_pars[:, self.n_mix * self.n_samples:2*self.n_mix * self.n_samples] = reduced_mdn_pars[:, self.n_mix * self.n_samples:2*self.n_mix * self.n_samples] * self.norm_std

        mdn_means = np.sum(mdn_pars[:, :self.n_mix] * tf.nn.softmax(mdn_pars[:, 2*self.n_mix:]).numpy(), axis=1)
        reduced_mdn_means = np.sum(
            reduced_mdn_pars[:, :self.n_mix*self.n_samples] * tf.nn.softmax(reduced_mdn_pars[:, 2*self.n_mix*self.n_samples:]).numpy(),
            axis=1
        )
        self.log_rms.append(
            np.mean((mdn_means - self.targets)**2)
        )
        self.log_rms_mc.append(
            np.mean((reduced_mdn_means - self.targets)**2)
        )

        sample_mdn = mdn_pars[self.indices]
        sample_reduced_mdn = reduced_mdn_pars[self.indices]

        sample_gms = generateGaussianMixtures(sample_mdn, self.n_mix)
        sample_reduced_gms = generateGaussianMixtures(sample_reduced_mdn, self.n_mix * self.n_samples)
        lgms = generateLogNormalMixture(sample_reduced_mdn, self.n_mix * self.n_samples)

        logz_vals = np.linspace(-4,4,self.n_z)
        z_vals = np.linspace(1e-10,14, self.n_z)
        sample_pdfs = np.empty((
            len(self.indices), self.n_z
        ))
        sample_reduced_pdfs = np.empty((
            len(self.indices), self.n_z
        ))
        sample_scaled_pdfs = np.empty((
            len(self.indices), self.n_z
        ))

        for i,idx in enumerate(self.indices):
            sample_pdfs[i] = sample_gms[i].prob(logz_vals)
            sample_reduced_pdfs[i] = sample_reduced_gms[i].prob(logz_vals)
            sample_scaled_pdfs[i] = lgms[i].prob(z_vals)
        
        self.pdf.append(sample_pdfs)
        self.pdf_mc.append(sample_reduced_pdfs)
        self.scaled_pdf.append(sample_scaled_pdfs)

        if epoch % 100 == 0:
            print('saving metrics values')
            pdf = np.array(self.pdf)
            pdf_mc = np.array(self.pdf_mc)
            scaled_pdf = np.array(self.scaled_pdf)
            logrms = np.array(self.log_rms)
            logrms_mc = np.array(self.log_rms_mc)

            np.save(self.path + 'pdf.npy', pdf)
            np.save(self.path + 'pdf_mc.npy', pdf_mc)
            np.save(self.path + 'scaled_pdf.npy', scaled_pdf)
            np.save(self.path + 'logrms.npy', logrms)
            np.save(self.path + 'logrms_mc.npy', logrms_mc)

        print(f'\nTime for metrics: {time()-t0} s')

        del (
            mdn_pars, mdn_pars_mc, reduced_mdn_pars, mdn_means, reduced_mdn_means, 
            sample_mdn, sample_reduced_mdn, sample_gms, sample_reduced_gms, sample_pdfs,
            sample_reduced_pdfs, sample_scaled_pdfs, lgms
        )
        gc.collect()
        
        return