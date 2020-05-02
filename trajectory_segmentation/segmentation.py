import numpy as np
import pandas as pd
from models import TimeVaryingGaussianMixtureModel,HiddenSemiMarkovModel,k_gmeans,k_bic,AutoregressiveMarkovModel


def get_segmentation_functions(substr=""):
    seg_funcs = []
    seg_func_keys = []
    for k in globals().keys():
        if ("gmm_seg" in k) or ("hsmm_seg" in k):
            if substr == "":
                seg_func_keys.append(k)
                seg_funcs.append(globals()[k])
            elif substr in k:
                seg_func_keys.append(k)
                seg_funcs.append(globals()[k])

    seg_funcs_dict = dict(zip(seg_func_keys,seg_funcs))
    return seg_funcs_dict


def segment_demos(demo_df,ann_df,segment_func,prune):
    k = ann_df.label.unique().size

    demo_inds = pd.unique(demo_df.index.droplevel(level=2))
    demo_list = []

    for i in range(0,demo_inds.shape[0]):
        this_demo_ann = ann_df.loc[demo_inds[i]].copy()
        start_i = int(this_demo_ann.iloc[0]["i0"])
        stop_i = int(this_demo_ann.iloc[-1]["i1"])

        demo_list.append(demo_df.loc[demo_inds[i],:].to_numpy()[start_i:stop_i+1,:])

    seg_res = segment_func(demo_list,k)

    total_seg_df = pd.DataFrame()
    for i in range(0,demo_inds.shape[0]):

        seg_np = convert_transitions_to_segments(seg_res[i][0], seg_res[i][1],prune)

        seg_df = pd.DataFrame(seg_np,columns=["i0","i1","label"] )
        seg_df['task'] =[demo_inds[i][0]] * seg_np.shape[0]
        seg_df['id'] = [demo_inds[i][1]] * seg_np.shape[0]
        seg_df['segment_no'] = range(0, seg_df.shape[0])
        seg_df = seg_df.set_index(['task', 'id', 'segment_no'])

        total_seg_df = total_seg_df.append(seg_df.copy())

    return total_seg_df


def gmm_seg_k_dp(demo_list,k):
    gmm = TimeVaryingGaussianMixtureModel(max_clusters=k*2,dp=True)
    for i in range(0,len(demo_list)):
        gmm.add_demo(demo_list[i])
    return gmm.fit()

def gmm_seg_k_bic(demo_list,k):
    gmm = TimeVaryingGaussianMixtureModel(k_func=k_bic,max_clusters=k*2)
    for i in range(0,len(demo_list)):
        gmm.add_demo(demo_list[i])
    return gmm.fit()

def gmm_seg_k_gmeans(demo_list,k):
    gmm = TimeVaryingGaussianMixtureModel(k_func=k_gmeans,max_clusters=k*2)
    for i in range(0,len(demo_list)):
        gmm.add_demo(demo_list[i])
    return gmm.fit()

def gmm_seg_k_fixed(demo_list,k):
    gmm = TimeVaryingGaussianMixtureModel(k=k)
    for i in range(0,len(demo_list)):
        gmm.add_demo(demo_list[i])
    return gmm.fit()

# def ar_hdp_hsmm_seg(demo_list,k=None):
#     hmm = AutoregressiveMarkovModel()
#     for i in range(0,len(demo_list)):
#         hmm.add_demo(demo_list[i])
#     return hmm.fit()

def hdp_hsmm_seg(demo_list, k=None):
    hmm = HiddenSemiMarkovModel()
    for i in range(0,len(demo_list)):
        hmm.add_demo(demo_list[i])
    return hmm.fit()

def convert_transitions_to_segments(transitions,seg_labels,prune=False):

    seg_np = np.empty((0,3))
    start_i = 0
    prev_label = -1
    for i in range(1, transitions.size):
        if prune:

            if ((transitions[i] - transitions[start_i]) > 5) & (seg_labels[i-1] != prev_label):
                seg_np = np.vstack((seg_np,[[transitions[start_i],transitions[i]-1,seg_labels[i-1]]]))
                start_i = i
                prev_label = seg_labels[i-1]
            # elif (i < transitions.size - 1) & ((transitions[i] - transitions[start_i]) <= 3) & (seg_labels[i-1] != prev_label):
            #     if(start_i > 0):
            #         seg_np[-1,1] = transitions[start_i] + (transitions[i] - transitions[start_i])//2
            #         transitions[i] = transitions[start_i] + (transitions[i] - transitions[start_i])//2 + 1
            #         start_i = i
            #         seg_labels[i-1] = prev_label
            #     else :
            #         transitions[i] = start_i
            #         prev_label = seg_labels[i-1]
            elif (seg_labels[i-1] == prev_label):
                seg_np[-1, 1] = transitions[i]-1
                start_i = i
        else:
            if abs(transitions[i-1] - (transitions[i]-1)) <=1:
                transitions[i] = transitions[i] +1
            seg_np = np.vstack((seg_np,[[transitions[i-1],transitions[i]-1,seg_labels[i-1]]]))


    if seg_np.size==0:
        print("oops")
    pred_label_dict = dict(zip(np.unique(seg_np[:,2]).tolist(),range(0,np.unique(seg_np[:,2]).size)))
    for i,old_label in enumerate(seg_np[:,2]):
        seg_np[i,2] = pred_label_dict[old_label]
    seg_np[-1,1] += 1
    return seg_np
