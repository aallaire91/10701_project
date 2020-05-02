import numpy as np
import pandas as pd
import constants
from sklearn.preprocessing import StandardScaler
from sklearn import mixture, decomposition
from copy import deepcopy
import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt


#  return list of numpy array
def preprocess_demos_list(demo_list,preprocess_func_keys,preprocess_func_params):
    if len(preprocess_func_keys) == 0:
        return demo_list

    demo_list_proc = deepcopy(demo_list)

    for k in preprocess_func_keys:
        preprocess_func = constants.pp_func_label_map[k]
        # pca needs to operate over entire set
        if ('pca' in k) or ('lda' in k):
            data_sizes = []
            data_concat = np.empty((0,demo_list_proc[0].shape[1]))
            for i in range(0,len(demo_list_proc)):
                data_sizes.append(demo_list_proc[i].shape[0])
                data_concat = np.vstack((data_concat,demo_list_proc[i]))
            if preprocess_func_params[k] is None:
                data_concat_proc = preprocess_func(data_concat)
            else:
                data_concat_proc = preprocess_func(data_concat,preprocess_func_params[k])

            start_i=0
            for i in range(0,len(demo_list_proc)):
                end_i = start_i + int(data_sizes[i])
                demo_list_proc[i] = data_concat_proc[start_i:end_i]
                start_i = end_i
        # all others operate on each demo individually
        else:
            for i in range(0, len(demo_list_proc)):
                if preprocess_func_params[k] is None:
                    demo_list_proc[i] = preprocess_func(demo_list_proc[i])
                else:
                    demo_list_proc[i] = preprocess_func(demo_list_proc[i], preprocess_func_params[k])

    return demo_list_proc


def preprocess_demos_df(demo_df,preprocess_funcs,preprocess_func_params):
    if len(preprocess_funcs) == 0:
        return demo_df
    demo_inds = np.unique(demo_df.index.droplevel(level=2))

    demo_df_proc = demo_df.copy()
    demo_list = []
    for i in range(0, demo_inds.shape[0]):
        demo_list.append(demo_df_proc.loc[demo_inds[i],:].to_numpy())

    demo_list_proc = preprocess_demos_list(demo_list,preprocess_funcs,preprocess_func_params)

    demo_df_proc = pd.DataFrame()

    for i in range(0, len(demo_list_proc)):
        n = demo_list_proc[i].shape[0]
        if demo_list_proc[i].shape[1] == demo_df.shape[1]:
            cols = demo_df.columns
        else:
            cols = list(range(0,demo_list_proc[i].shape[1] ))
        ind_arrays = [[demo_inds[i][0]]*n,[demo_inds[i][1]]*n,range(0,n)]
        ind_tuples = list(zip(*ind_arrays))
        inds = pd.MultiIndex.from_tuples(ind_tuples,names=['task','id','i'])
        this_df = pd.DataFrame(demo_list_proc[i],index=inds,columns=cols)
        demo_df_proc = demo_df_proc.append(this_df)

    return demo_df_proc




def low_pass(demo,cutoff_freq = 10):
    demo_filt = butter_lowpass_filter(demo,cutoff_freq,30,5)
    return demo_filt

def decimate(demo,factor=3):
    demo_dec = demo[0::factor,:]
    return demo_dec

def standardize(demo):
    scaler = StandardScaler()
    scaler.fit(demo)
    demo_sc = scaler.transform(demo)
    return demo_sc

def window_slds(demo,width=2):
    windowed = np.zeros((demo.shape[0], width * demo.shape[1]))
    for i in range(0,width):
        windowed[0:demo.shape[0]- i,i*demo.shape[1]:(i+1)*demo.shape[1]] = demo[i:demo.shape[0], :]

    return windowed[0:demo.shape[0] - (width-1),:]

def time_augment(demo):
    time_augmented = np.zeros((demo.shape[0], demo.shape[1] + 1))
    time_augmented[:, :-1] = demo
    time_augmented[:, demo.shape[1]] = np.arange(0, demo.shape[0], 1)
    return time_augmented

def kpca(dataset,n=12):
    kpca = decomposition.KernelPCA(n, kernel='rbf', eigen_solver='auto', n_jobs=-1,remove_zero_eig=True)
    dataset = kpca.fit_transform(dataset)

    return dataset

def pca(dataset,p=0.95):
    pca = decomposition.PCA(p)
    dataset = pca.fit_transform(dataset)

    return dataset


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data,axis=0)
    return y
