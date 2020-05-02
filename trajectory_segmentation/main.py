from load_data import load_preprocessed_pkl,get_pp_suffix,load_jigsaws_data_from_pkl
import plot_results
from preprocessing import preprocess_demos_df
import os
from os import listdir
from os.path import isfile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import constants
import segmentation
import metrics
import resource
import gc
# if not os.path.exists('./plots'):
#     os.mkdir('./plots')
#
# plot_results.plot_all_and_save(kin_df,ann_df,None,savedir="./plots/jigsaws/no_filter/")
# plot_results.plot_all_and_save(kin_df,ann_df,preprocessing.low_pass_1,savedir="./plots/jigsaws/low_pass_1/")
# plot_results.plot_all_and_save(kin_df,ann_df,preprocessing.low_pass_5,savedir="./plots/jigsaws/low_pass_5/")
# plot_results.plot_all_and_save(kin_df,ann_df,preprocessing.low_pass_10,savedir="./plots/jigsaws/low_pass_10/")
pd.options.mode.chained_assignment = None

def get_ids_by_skill(skill="E"):
    ids = []
    for id in constants.skill_map.keys():
        if constants.skill_map[id] is skill:
            ids.append(id)
    return ids

def n_choose_k(n,k):
    return np.random.choice(range(0, n), (k,), replace=False)

def main():

    kin_df, ann_df = load_preprocessed_pkl(feature_type='quat',pkl_path=constants.pkl_path,pp_funcs = ['lp','dec','std'],pp_func_params={'lp':1.5,'dec':3,'std':None})
    kin_df = kin_df.sort_index()
    ann_df = ann_df.sort_index()
    ann_df = fill_unlabeled_segments(ann_df)
    expert_ids = get_ids_by_skill("E")
    kin_df_expert = kin_df.loc(axis=0)[:,expert_ids,:][constants.slave_cols_quat]
    ann_df_expert = ann_df.loc(axis=0)[:,expert_ids,:]

    suture_df = kin_df_expert.xs("Suturing",drop_level=False)
    knot_tying_df = kin_df_expert.xs("Knot_Tying",drop_level=False)
    needle_passing_df = kin_df_expert.xs("Needle_Passing",drop_level=False)
    suture_ann_df = ann_df_expert.xs("Suturing",drop_level=False)
    knot_ann_tying_df = ann_df_expert.xs("Knot_Tying",drop_level=False)
    needle_ann_passing_df = ann_df_expert.xs("Needle_Passing",drop_level=False)

    # choose task type and features
    demo_df = suture_df.copy()
    demo_ann_df = suture_ann_df.copy()
    demo_inds = pd.unique(demo_df.index.droplevel(level=2))

    preprocess_funcs = ['win','kpca','lp']
    preprocess_func_params = dict(zip(preprocess_funcs,[10,8,1]))
    demo_df_proc = preprocess_demos_df(demo_df,preprocess_funcs,preprocess_func_params)
    for ind in demo_inds:
        plot_results.plot_gt_px_map_seg_v_time(demo_df_proc.loc[ind], demo_ann_df.loc[ind], ind, savedir="./results/plots/pos_vel_win10_kpca8_lp_test/")

    # from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=2, random_state=0)
    #
    # from matplotlib import pyplot as plt
    # plt.figure(figsize=(6, 5))
    # constants.segment_colors_dict
    # tsne_ann_df = demo_ann_df.loc[]
    # for i in range(demo_inds:
    #
    #     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    # plt.legend()
    # plt.show()

    seg_funcs_dict = segmentation.get_segmentation_functions('hsmm')

    # hmm_func_plot_labels = ["AR_HDP_HSMM", "HDP_HSMM"]
    hmm_func_plot_labels = ["HDP_HSMM"]

    exp_metrics_df = pd.DataFrame()
    exp_seg_dfs = []
    for seg_key in seg_funcs_dict.keys():
        exp_name = seg_key

        seg_df  = segmentation.segment_demos(demo_df_proc, demo_ann_df, seg_funcs_dict[seg_key], False)
        for i in range(0, demo_inds.shape[0]):
            this_ann_df = demo_ann_df.loc[demo_inds[i]]
            start_i = this_ann_df.iloc[0]["i0"]
            seg_df.loc[demo_inds[i], ["i0", "i1"]] = seg_df.loc[demo_inds[i], ["i0", "i1"]].values + start_i

        exp_seg_dfs.append(seg_df)

    for ind in demo_inds:
        this_seg_list = []

        for i in range(0, len(exp_seg_dfs)):
            this_seg_list.append(exp_seg_dfs[i].loc[ind])

        plot_results.plot_demo_seg_results(demo_df_proc.loc[ind], demo_ann_df.loc[ind], ind, this_seg_list,
                                           hmm_func_plot_labels, "./results/plots/pos_vel_win10_kpca8_lp_test/")

        # ax = plt.subplot(1,1,1)
        # pca_fig = plt.figure()
        # col_labels = np.linspace(0,demo_df_proc.shape[1]-1,demo_df_proc.shape[1]).tolist()
        # colors = plot_results.get_colors(demo_df_proc.shape[1],'gist_rainbow')
        # linestyles = ['-']*demo_df_proc.shape[1]
        # plot_results.plot_trajectories_v_time(ax,demo_df_proc,col_labels,col_labels,colors,linestyles,'PCA Components',False)
        # demo_df_proc = demo_df_proc.rename(columns = dict(zip(np.linspace(0,demo_df_proc.shape[1]-1,demo_df_proc.shape[1]).tolist(),constants.slave_cols_quat)))

        #
        # plot_results.plot_kpca_3d(pca_fig,demo_df_proc.iloc[:,0:3].values,demo_ann_df,segment_colors_dict)

        #plot_results.plot_all_and_save(demo_df,demo_ann_df,segment_colors_dict,"./results/plots/hdp_hsmm_eval/")
        # metrics_df = metrics.compute_intrinsic_metrics(demo_df_proc, seg_df)
        # metrics_df = metrics_df.rename(columns={0: exp_name})
        # exp_metrics_df = pd.concat((exp_metrics_df, metrics_df), axis=1)



def generate_features(feat_cols,pp_func_l1_dicts,pp_func_l2_dicts):
    pp_funcs_L1_list = [['lp','dec','std']]*3
    pp_func_L1_params_list = [{'lp':1.5,'dec':3,'std':None},{'lp':5,'dec':3,'std':None},{'lp':10,'dec':3,'std':None}]
    task_list = ["Suturing","Knot_Tying","Needle_Passing","All"]
    skill = "E"
    for L1i in range(0,len(pp_funcs_L1_list)):
        pp_L1_suffix = get_pp_suffix(pp_funcs_L1_list[L1i],pp_func_L1_params_list[L1i])
        kin_df, ann_df = load_preprocessed_pkl(feature_type='quat',pkl_path=constants.pkl_path,pp_funcs = pp_funcs_L1_list[L1i],pp_func_params=pp_func_L1_params_list[L1i])
        save_annotations(ann_df, pp_L1_suffix)


def run_seg_func_cv_within_task_eval(segfunc_key, feat_key, pp_funcs_L2_list,pp_func_L2_params_list,task_key=""):
    pp_funcs_L1= ['lp','dec','std']
    pp_func_L1_params = {'lp':1.5,'dec':3,'std':None}
    skill = "E"
    pp_L1_suffix = get_pp_suffix(pp_funcs_L1,pp_func_L1_params)
    kin_df, ann_df = load_preprocessed_pkl(feature_type='quat_eedist_3dsig',pkl_path=constants.pkl_path,pp_funcs = pp_funcs_L1,pp_func_params=pp_func_L1_params)
    kin_df = kin_df.sort_index()
    ann_df = ann_df.sort_index()
    # ann_df = fill_unlabeled_segments(ann_df)
    save_annotations(ann_df, "")

    expert_ids = get_ids_by_skill(skill)
    kin_df_expert = kin_df.loc(axis=0)[:,expert_ids,:]
    ann_df_expert = ann_df.loc(axis=0)[:,expert_ids,:]

    if task_key is "":

        for task in constants.tasks[1:]:

            task_kin_df = kin_df_expert.xs(task,drop_level=False).copy()
            task_ann_df = ann_df_expert.xs(task,drop_level=False).copy()

            savefile_suffix = "_" + task
            save_annotations(task_ann_df, savefile_suffix)

            task_features_df = task_kin_df[constants.feat_columns_map[feat_key]]
            save_features(task_features_df, savefile_suffix + "_" + feat_key)
            run_feature_L2_cv(task_features_df, task_ann_df, segfunc_key,savefile_suffix+"_"+feat_key,pp_funcs_L2_list,pp_func_L2_params_list)
    else:
        task_kin_df = kin_df_expert.xs(task_key, drop_level=False).copy()
        task_ann_df = ann_df_expert.xs(task_key, drop_level=False).copy()

        savefile_suffix = "_" + task_key
        save_annotations(task_ann_df, savefile_suffix)

        task_features_df = task_kin_df[constants.feat_columns_map[feat_key]]
        save_features(task_features_df, savefile_suffix + "_" + feat_key)
        run_feature_L2_cv(task_features_df, task_ann_df, segfunc_key, savefile_suffix + "_" + feat_key, pp_funcs_L2_list,
                          pp_func_L2_params_list)

def run_feature_cv_across_task_eval(segfunc_key = "gt"):
    pd.options.mode.chained_assignment = None
    pp_funcs_L1= ['lp','dec','std']
    pp_func_L1_params = {'lp':1.5,'dec':3,'std':None}

    pp_funcs_L2_list = [None]+[['kpca','lp']]*5 + [['win','kpca','lp']]*4
    pp_func_L2_params_list = [None,{'kpca':6,'lp':1.5},{'kpca':8,'lp':1.5},{'kpca':10,'lp':1.5},{'kpca':12,'lp':1.5},{'kpca':14,'lp':1.5},{'win':2,'kpca':8,'lp':1.5},{'win':3,'kpca':8,'lp':1.5},{'win':5,'kpca':8,'lp':1.5},{'win':10,'kpca':8,'lp':1.5}]

    skill = "E"
    pp_L1_suffix = get_pp_suffix(pp_funcs_L1,pp_func_L1_params)
    kin_df, ann_df = load_preprocessed_pkl(feature_type='quat_eedist_3dsig',pkl_path=constants.pkl_path,pp_funcs = pp_funcs_L1,pp_func_params=pp_func_L1_params)
    kin_df = kin_df.sort_index()
    ann_df = ann_df.sort_index()
    # ann_df = fill_unlabeled_segments(ann_df)
    save_annotations(ann_df, "")

    expert_ids = get_ids_by_skill(skill)
    kin_df_expert = kin_df.loc(axis=0)[:,expert_ids,:].copy()
    ann_df_expert = ann_df.loc(axis=0)[:,expert_ids,:].copy()

    savefile_suffix = "_all"
    save_annotations(ann_df, savefile_suffix)
    for feat in constants.feat_columns_map.keys():
        task_features_df = kin_df_expert[constants.feat_columns_map[feat]]
        save_features(task_features_df, savefile_suffix + "_" + feat)
        run_feature_L2_cv(task_features_df, ann_df_expert, segfunc_key, savefile_suffix + "_" + feat,
                          pp_funcs_L2_list, pp_func_L2_params_list)


def run_feature_cv_within_task_eval(segfunc_key = "gt",task_key=""):
    pd.options.mode.chained_assignment = None
    pp_funcs_L1= ['lp','dec','std']
    pp_func_L1_params = {'lp':1.5,'dec':3,'std':None}

    pp_funcs_L2_list = [None]+[['kpca']]*5 + [['win','kpca']]*25
    pp_func_L2_params_list = [None,{'kpca':6},{'kpca':8},{'kpca':10},{'kpca':12},{'kpca':14},
                              {'win': 2, 'kpca': 6}, {'win': 3, 'kpca': 6}, {'win': 5, 'kpca': 6},{'win': 10, 'kpca': 6}, {'win': 15, 'kpca': 6},
                              {'win':2,'kpca':8},{'win':3,'kpca':8},{'win':5,'kpca':8},{'win':10,'kpca':8},{'win':15,'kpca':8},
                              {'win': 2, 'kpca': 10}, {'win': 3, 'kpca': 10}, {'win': 5, 'kpca': 10},{'win': 10, 'kpca': 10}, {'win': 15, 'kpca': 10},
                              {'win': 2, 'kpca': 12}, {'win': 3, 'kpca': 12}, {'win': 5, 'kpca': 12},{'win': 10, 'kpca': 12}, {'win': 15, 'kpca': 12},
                              {'win': 2, 'kpca': 14}, {'win': 3, 'kpca': 14}, {'win': 5, 'kpca': 14},{'win': 10, 'kpca': 14}, {'win': 15, 'kpca': 14},
                              ]

    pp_funcs_L2_list = [['win']]*5
    pp_func_L2_params_list = [{'win': 2}, {'win': 3},{'win': 5}, {'win': 10}, {'win': 15}]

    skill = "E"
    pp_L1_suffix = get_pp_suffix(pp_funcs_L1,pp_func_L1_params)
    kin_df, ann_df = load_preprocessed_pkl(feature_type='quat_eedist_3dsig',pkl_path=constants.pkl_path,pp_funcs = pp_funcs_L1,pp_func_params=pp_func_L1_params)
    kin_df = kin_df.sort_index()
    ann_df = ann_df.sort_index()
    # ann_df = fill_unlabeled_segments(ann_df)
    save_annotations(ann_df, "")

    expert_ids = get_ids_by_skill(skill)
    kin_df_expert = kin_df.loc(axis=0)[:,expert_ids,:]
    ann_df_expert = ann_df.loc(axis=0)[:,expert_ids,:]

    if task_key is "":
        for task in constants.tasks:

            task_kin_df = kin_df_expert.xs(task,drop_level=False).copy()
            task_ann_df = ann_df_expert.xs(task,drop_level=False).copy()

            savefile_suffix = "_" + task
            save_annotations(task_ann_df, savefile_suffix)
            for feat in constants.feat_columns_map.keys():
                task_features_df = task_kin_df[constants.feat_columns_map[feat]]
                save_features(task_features_df, savefile_suffix + "_" + feat)
                run_feature_L2_cv(task_features_df, task_ann_df, segfunc_key,savefile_suffix+"_"+feat,pp_funcs_L2_list,pp_func_L2_params_list)
    else:
        task_kin_df = kin_df_expert.xs(task_key, drop_level=False).copy()
        task_ann_df = ann_df_expert.xs(task_key, drop_level=False).copy()

        savefile_suffix = "_" + task_key
        save_annotations(task_ann_df, savefile_suffix)
        for feat in constants.feat_columns_map.keys():
            task_features_df = task_kin_df[constants.feat_columns_map[feat]]
            save_features(task_features_df, savefile_suffix + "_" + feat)
            run_feature_L2_cv(task_features_df, task_ann_df, segfunc_key, savefile_suffix + "_" + feat,
                              pp_funcs_L2_list, pp_func_L2_params_list)



def run_feature_L2_cv(kin_df,ann_df,seg_func_key, savefile_suffix,pp_funcs_list,pp_func_params_list):

    for i in range(0,len(pp_func_params_list)):
        if pp_funcs_list[i] is None:
            run_loo_cv(kin_df, ann_df, seg_func_key, savefile_suffix)
        else:
            pp_key = get_pp_suffix(pp_funcs_list[i],pp_func_params_list[i])
            kin_df_proc = preprocess_demos_df(kin_df,pp_funcs_list[i],pp_func_params_list[i])
            this_suffix = savefile_suffix + pp_key
            save_features(kin_df_proc,this_suffix)
            run_loo_cv(kin_df_proc,ann_df,seg_func_key,this_suffix)


def run_loo_cv(kin_df,ann_df,seg_func_key,savefile_suffix):
    cv_seg_dfs = []
    demo_inds = np.unique(kin_df.index.droplevel(level=2))
    seg_funcs_dict = constants.seg_func_label_map
    # do leave one out cv
    cv_metrics_df = pd.DataFrame()

    for k in range(0, demo_inds.size):

        loo_demo_inds = demo_inds[np.arange(demo_inds.size) != k]
        loo_kin_df = kin_df.loc(axis=0)[kin_df.index.droplevel(level=2).isin(loo_demo_inds) ]
        loo_ann_df = ann_df.loc(axis=0)[ann_df.index.droplevel(level=2).isin(loo_demo_inds) ]
        for i in range(0, loo_demo_inds.shape[0]):
            this_ann_df = loo_ann_df.loc[loo_demo_inds[i]]
            this_kin_df = loo_kin_df.loc[loo_demo_inds[i]]
            if this_ann_df.iloc[-1,1] > (this_kin_df.shape[0]-1):
                this_ann_df.iloc[-1, 1] = this_kin_df.shape[0]-1

        if seg_func_key is "gt":
            seg_df = loo_ann_df.copy()
        else:
            seg_df = segmentation.segment_demos(loo_kin_df, loo_ann_df, seg_funcs_dict[seg_func_key], False)
            for i in range(0, loo_demo_inds.shape[0]):
                this_ann_df = loo_ann_df.loc[loo_demo_inds[i]]
                this_seg_def = seg_df.loc[loo_demo_inds[i]]
                start_i = this_ann_df.iloc[0][0]
                end_i = this_ann_df.iloc[-1][1]
                seg_df.loc[loo_demo_inds[i], ["i0", "i1"]] = seg_df.loc[loo_demo_inds[i], ["i0", "i1"]].values + start_i
                this_seg_def.iloc[-1,1] = end_i


        cv_seg_dfs.append(seg_df)

        this_metrics_df = metrics.compute_metrics(loo_kin_df,seg_df,loo_ann_df)
        this_metrics_df["name"] =  savefile_suffix[1:] +"_" + seg_func_key
        this_metrics_df = this_metrics_df.set_index(["name"])

        cv_metrics_df = cv_metrics_df.append(this_metrics_df)
        print(resource.getrusage(resource.RUSAGE_SELF))
        print("After GC: ")
        gc.collect()
        print(resource.getrusage(resource.RUSAGE_SELF))
    best_cv = int(np.argmax(cv_metrics_df.loc(axis=1)["munkres"].values))
    best_cv_suffix = savefile_suffix + "_" + seg_func_key + "_cv" + str(best_cv)
    save_segmentations(cv_seg_dfs[best_cv],best_cv_suffix)

    metrics_avg = np.sum(cv_metrics_df.values,axis=0)/cv_metrics_df.shape[0]
    metrics_avg = metrics_avg[:,np.newaxis].T
    metrics_df = pd.DataFrame(metrics_avg,columns=["ari","munkres","si","dbi"])
    metrics_df["name"] = cv_metrics_df.index[0]
    metrics_df["best_cv"] = best_cv # this means nothing for gt
    metrics_df = metrics_df.set_index(["name"])
    save_metrics(metrics_df, savefile_suffix +"_" + seg_func_key)

def run_gmm_cv():
    # Feature Selection Cross Validation
    segfunc_keys = [f for f in constants.seg_func_label_map if "gmm_" in f]
    feat_key = "pos_eedist"
    pp_funcs_L2_list = [['win', 'kpca']]
    pp_func_L2_params_list = [{'win':3,'kpca':12}]
    for segfunc_key in segfunc_keys:
        run_seg_func_cv_within_task_eval(segfunc_key, feat_key, pp_funcs_L2_list, pp_func_L2_params_list, task_key="")

def run_hsmm_cv():
    # Feature Selection Cross Validation
    segfunc_key = "hdp_hsmm"
    feat_key = "pos_eedist"
    pp_funcs_L2_list = [['win', 'kpca']]
    pp_func_L2_params_list = [{'win':3,'kpca':12}]
    run_seg_func_cv_within_task_eval(segfunc_key, feat_key, pp_funcs_L2_list, pp_func_L2_params_list, task_key="")

def save_metrics(metrics_df,savefile_suffix):
    if os.path.exists("./results/data/") == False:
        os.mkdir("./results/data/")
    if os.path.exists("./results/data/metrics/") == False:
        os.mkdir("./results/data/metrics/")
    path = os.path.join("./results/data/metrics", "metrics" + savefile_suffix + ".pkl")
    print("    Saved data to " + path + ".")
    metrics_df.to_pickle(path)


def save_annotations(ann_df, savefile_suffix):
    if os.path.exists("./results/data/") == False:
        os.mkdir("./results/data/")
    if os.path.exists("./results/data/annotations/") == False:
        os.mkdir("./results/data/annotations/")
    path = os.path.join("./results/data/annotations", "annotations" + savefile_suffix + ".pkl")
    print("    Saved data to " + path + ".")
    ann_df.to_pickle(path)


def save_features(feature_df, savefile_suffix):
    if os.path.exists("./results/data/") == False:
        os.mkdir("./results/data/")
    if os.path.exists("./results/data/features/") == False:
        os.mkdir("./results/data/features/")
    path = os.path.join("./results/data/features", "features" + savefile_suffix + ".pkl")
    print("    Saved data to " + path + ".")
    feature_df.to_pickle(path)


def save_segmentations(seg_df, savefile_suffix):
    if os.path.exists("./results/data/") == False:
        os.mkdir("./results/data/")
    if os.path.exists("./results/data/segmentations/") == False:
        os.mkdir("./results/data/segmentations/")
    path = os.path.join("./results/data/segmentations", "segmentations" + savefile_suffix + ".pkl")
    print("    Saved data to " + path + ".")
    seg_df.to_pickle(path)


def fill_unlabeled_segments(ann_df):
    ann_df_edited = pd.DataFrame()
    demo_inds = np.unique(ann_df.index.droplevel(level=2))
    for i in range(demo_inds.shape[0]):
        this_ann_df = ann_df.xs(demo_inds[i],drop_level=False)
        this_ann_df_edited = pd.DataFrame()
        count = 0
        for j in range(this_ann_df.shape[0]-1):
            end_j = this_ann_df.iloc[j, 1]
            next_start_j = this_ann_df.iloc[j + 1, 0]
            this_entry = this_ann_df.iloc[j, :].copy()
            this_entry.name = (this_ann_df.index[0][0],this_ann_df.index[0][1],count)
            this_ann_df_edited = this_ann_df_edited.append(pd.DataFrame(this_entry).T)
            count += 1
            if (next_start_j - end_j) > 2:
                new_entry = this_ann_df.iloc[j, :].copy()
                new_entry.name = (this_ann_df.index[0][0], this_ann_df.index[0][1], count)
                new_entry.iloc[0, 0] = end_j + 1
                new_entry.iloc[0, 1] = next_start_j - 1
                new_entry.iloc[0, 2] = "G16"
                this_ann_df_edited = this_ann_df_edited.append(pd.DataFrame(new_entry).T)
                count += 1
            elif (next_start_j - end_j) > 1:
                this_ann_df_edited.iloc[-1,1] = next_start_j - 1

        this_entry = this_ann_df.iloc[-1, :].copy()
        this_entry.name = (this_ann_df.index[0][0], this_ann_df.index[0][1], count)
        this_ann_df_edited = this_ann_df_edited.append(pd.DataFrame(this_entry).T)

        ann_df_edited = ann_df_edited.append(this_ann_df_edited)
    return ann_df_edited

def load_metrics(substr_list=[""]):
    metrics_path = os.path.join(constants.results_path,"data/metrics/")
    file_list = []
    dir_list = listdir(metrics_path)
    for f in dir_list:
        if isfile(os.path.join(metrics_path, f)):
            match = True
            for ss in substr_list:
                if (ss is not "") and (ss not in f):
                    match = False
            if match:
                file_list.append(f)

    metrics_df = pd.DataFrame()
    for fname in file_list:
        metrics_df = metrics_df.append(pd.read_pickle(os.path.join(metrics_path,fname)))

    metrics_df = normalize_si(metrics_df)
    return metrics_df

def normalize_si(metrics_df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(np.array([[-1],[1]]))
    metrics_df_norm = metrics_df.copy()
    metrics_df_norm.loc(axis=1)["si"] = scaler.transform(metrics_df_norm.loc(axis=1)["si"].values[:,np.newaxis])
    return metrics_df_norm


def export_metrics_to_csv(savepath,substr_list = [""]):
    metrics_df = load_metrics(substr_list)
    metrics_df.to_csv(savepath)

def rank_features(metrics_df):
    si_ranking = np.flip(np.argsort(metrics_df["si"]))
    dbi_ranking = np.argsort(metrics_df["dbi"])
    ari_ranking = np.flip(np.argsort(metrics_df["ari"]))
    munrkes_ranking =np.flip(np.argsort(metrics_df["munkres"]))

    intrinsic_score = np.zeros(si_ranking.shape)
    extrinsic_score = np.zeros(si_ranking.shape)
    overall_score = np.zeros(si_ranking.shape)
    for i in range(0,si_ranking.size):
        intrinsic_score[si_ranking[i]] += i
        intrinsic_score[dbi_ranking[i]] += i
        extrinsic_score[ari_ranking[i]] += i
        extrinsic_score[munrkes_ranking[i]] += i

    overall_score = intrinsic_score + extrinsic_score
    intrinsic_ranking = np.argsort(intrinsic_score)
    extrinsic_ranking = np.argsort(extrinsic_score)
    overall_ranking = np.argsort(overall_score)
    return intrinsic_ranking,extrinsic_ranking,overall_ranking

def load_feat_basic_metrics(func_key,feat_key):
    kt_metrics_df = load_metrics([ "Knot_Tying", feat_key+"_" + func_key])
    kt_metrics_df = kt_metrics_df.sort_index()
    kt_idx = pd.Index([i[11:] for i in kt_metrics_df.index if "lp" not in i])

    s_metrics_df = load_metrics(["Suturing",feat_key+"_"+func_key])
    s_metrics_df = s_metrics_df.sort_index()
    s_idx = pd.Index([i[9:] for i in s_metrics_df.index if "lp" not in i])

    np_metrics_df = load_metrics(["Needle_Passing",feat_key+"_"+func_key])
    np_metrics_df = np_metrics_df.sort_index()
    np_idx = pd.Index([i[15:] for i in np_metrics_df.index if "lp" not in i])

    idx = kt_idx.intersection(s_idx)
    idx = np_idx.intersection(idx)

    kt_idx = pd.Index(["Knot_Tying_" + i for i in idx])
    s_idx = pd.Index(["Suturing_" + i for i in idx])
    np_idx = pd.Index(["Needle_Passing_" + i for i in idx])
    kt_metrics_df = kt_metrics_df.loc(axis=0)[kt_idx]
    s_metrics_df = s_metrics_df.loc(axis=0)[s_idx]
    np_metrics_df = np_metrics_df.loc(axis=0)[np_idx]

    kt_intr_ranking, kt_extr_ranking, kt_overall_ranking = rank_features(kt_metrics_df)
    s_intr_ranking, s_extr_ranking, s_overall_ranking = rank_features(s_metrics_df)
    np_intr_ranking, np_extr_ranking, np_overall_ranking = rank_features(np_metrics_df)

    metrics_df = kt_metrics_df.copy()
    metrics_df = metrics_df.reindex(idx)
    metrics_df.loc[:, :] = (kt_metrics_df.values + np_metrics_df.values + s_metrics_df.values) / 3.0
    # intr_ranking, extr_ranking, overall_ranking = rank_features(metrics_df)
    # #
    # sorted_features_intr = idx[intr_ranking].to_numpy()[:, np.newaxis]
    # sorted_features_extr = idx[extr_ranking].to_numpy()[:, np.newaxis]
    # sorted_features_overall = idx[overall_ranking].to_numpy()[:, np.newaxis]
    # print(sorted_features_overall)
    return metrics_df



def load_feat_pca_metrics(func_key,feat_key):

    kt_metrics_df = load_metrics([func_key, "Knot_Tying", feat_key+"_kpca"])
    kt_metrics_df = kt_metrics_df.sort_index()
    kt_idx = pd.Index([i[11:] for i in kt_metrics_df.index if "lp" not in i])

    s_metrics_df = load_metrics([func_key, "Suturing",feat_key+"_kpca"])
    s_metrics_df = s_metrics_df.sort_index()
    s_idx = pd.Index([i[9:] for i in s_metrics_df.index if "lp" not in i])

    np_metrics_df = load_metrics([func_key, "Needle_Passing",feat_key+"_kpca"])
    np_metrics_df = np_metrics_df.sort_index()
    np_idx = pd.Index([i[15:] for i in np_metrics_df.index if "lp" not in i])

    idx = kt_idx.intersection(s_idx)
    idx = np_idx.intersection(idx)

    kt_idx = pd.Index(["Knot_Tying_" + i for i in idx])
    s_idx = pd.Index(["Suturing_" + i for i in idx])
    np_idx = pd.Index(["Needle_Passing_" + i for i in idx])
    kt_metrics_df = kt_metrics_df.loc(axis=0)[kt_idx]
    s_metrics_df = s_metrics_df.loc(axis=0)[s_idx]
    np_metrics_df = np_metrics_df.loc(axis=0)[np_idx]

    kt_intr_ranking, kt_extr_ranking, kt_overall_ranking = rank_features(kt_metrics_df)
    s_intr_ranking, s_extr_ranking, s_overall_ranking = rank_features(s_metrics_df)
    np_intr_ranking, np_extr_ranking, np_overall_ranking = rank_features(np_metrics_df)

    metrics_df = kt_metrics_df.copy()
    metrics_df = metrics_df.reindex(idx)
    metrics_df.loc[:, :] = (kt_metrics_df.values + np_metrics_df.values + s_metrics_df.values) / 3.0
    # intr_ranking, extr_ranking, overall_ranking = rank_features(metrics_df)
    # #
    # sorted_features_intr = idx[intr_ranking].to_numpy()[:, np.newaxis]
    # sorted_features_extr = idx[extr_ranking].to_numpy()[:, np.newaxis]
    # sorted_features_overall = idx[overall_ranking].to_numpy()[:, np.newaxis]
    # print(sorted_features_overall)
    return metrics_df

def load_feat_window_metrics(func_key,feat_key):
    kt_metrics_df = load_metrics([func_key, "Knot_Tying", feat_key+"_win"])
    kt_metrics_df = kt_metrics_df.sort_index()
    kt_idx = pd.Index([i[11:] for i in kt_metrics_df.index if "lp" not in i])

    s_metrics_df = load_metrics([func_key, "Suturing",feat_key+"_win"])
    s_metrics_df = s_metrics_df.sort_index()
    s_idx = pd.Index([i[9:] for i in s_metrics_df.index if "lp" not in i])

    np_metrics_df = load_metrics([func_key, "Needle_Passing",feat_key+"_win"])
    np_metrics_df = np_metrics_df.sort_index()
    np_idx = pd.Index([i[15:] for i in np_metrics_df.index if "lp" not in i])

    idx = kt_idx.intersection(s_idx)
    idx = np_idx.intersection(idx)

    kt_idx = pd.Index(["Knot_Tying_" + i for i in idx])
    s_idx = pd.Index(["Suturing_" + i for i in idx])
    np_idx = pd.Index(["Needle_Passing_" + i for i in idx])
    kt_metrics_df = kt_metrics_df.loc(axis=0)[kt_idx]
    s_metrics_df = s_metrics_df.loc(axis=0)[s_idx]
    np_metrics_df = np_metrics_df.loc(axis=0)[np_idx]

    kt_intr_ranking, kt_extr_ranking, kt_overall_ranking = rank_features(kt_metrics_df)
    s_intr_ranking, s_extr_ranking, s_overall_ranking = rank_features(s_metrics_df)
    np_intr_ranking, np_extr_ranking, np_overall_ranking = rank_features(np_metrics_df)

    metrics_df = kt_metrics_df.copy()
    metrics_df = metrics_df.reindex(idx)
    metrics_df.loc[:, :] = (kt_metrics_df.values + np_metrics_df.values + s_metrics_df.values) / 3.0
    # intr_ranking, extr_ranking, overall_ranking = rank_features(metrics_df)
    # #
    # sorted_features_intr = idx[intr_ranking].to_numpy()[:, np.newaxis]
    # sorted_features_extr = idx[extr_ranking].to_numpy()[:, np.newaxis]
    # sorted_features_overall = idx[overall_ranking].to_numpy()[:, np.newaxis]
    # print(sorted_features_overall)
    return metrics_df

def sorted_features(func_key, feat_key=""):
    kt_metrics_df = load_metrics([func_key, "Knot_Tying", feat_key])
    kt_metrics_df = kt_metrics_df.sort_index()
    kt_idx = pd.Index([i[11:] for i in kt_metrics_df.index if "lp" not in i])

    s_metrics_df = load_metrics([func_key, "Suturing", feat_key ])
    s_metrics_df = s_metrics_df.sort_index()
    s_idx = pd.Index([i[9:] for i in s_metrics_df.index if "lp" not in i])

    np_metrics_df = load_metrics([func_key, "Needle_Passing", feat_key ])
    np_metrics_df = np_metrics_df.sort_index()
    np_idx = pd.Index([i[15:] for i in np_metrics_df.index if "lp" not in i])

    idx = kt_idx.intersection(s_idx)
    idx = np_idx.intersection(idx)

    kt_idx = pd.Index(["Knot_Tying_" + i for i in idx])
    s_idx = pd.Index(["Suturing_" + i for i in idx])
    np_idx = pd.Index(["Needle_Passing_" + i for i in idx])
    kt_intr_ranking, kt_extr_ranking, kt_overall_ranking = rank_features(kt_metrics_df)
    s_intr_ranking, s_extr_ranking, s_overall_ranking = rank_features(s_metrics_df)
    np_intr_ranking, np_extr_ranking, np_overall_ranking = rank_features(np_metrics_df)
    kt_metrics_df = kt_metrics_df.loc(axis=0)[kt_idx]
    s_metrics_df = s_metrics_df.loc(axis=0)[s_idx]
    np_metrics_df = np_metrics_df.loc(axis=0)[np_idx]



    metrics_df = kt_metrics_df.copy()
    metrics_df = metrics_df.reindex(idx)
    metrics_df.loc[:, :] = (kt_metrics_df.values + np_metrics_df.values + s_metrics_df.values) / 3.0
    intr_ranking, extr_ranking, overall_ranking = rank_features(metrics_df)

    sorted_features_intr = idx[intr_ranking].to_numpy()[:, np.newaxis]
    sorted_features_extr = idx[extr_ranking].to_numpy()[:, np.newaxis]
    sorted_features_overall = idx[overall_ranking].to_numpy()[:, np.newaxis]

    return sorted_features_intr


# gesture legend
def plot_fig1():
    if os.path.exists(constants.paper_savefolder) == False:
        os.mkdir(constants.paper_savefolder)

    plot_results.plot_gesture_color_legend(constants.segment_colors_dict, constants.gesture_label_descr_map,constants.paper_savefolder+"fig1.pdf")
    os.system("pdfcrop " + constants.paper_savefolder+"fig1.pdf" + " " + constants.paper_savefolder+"fig1.pdf")

def plot_fig2():
    kin_df, ann_df = load_jigsaws_data_from_pkl()
    index = [["Knot_Tying","D003"],["Knot_Tying", "F003"],["Knot_Tying", "E003"]]

    plot_results.plot_trajectory_3d(kin_df, ann_df, index, ["\\textbf{Novice}","\\textbf{Intermediate}","\\textbf{Expert}"],constants.segment_colors_dict, False, True, constants.paper_savefolder+"fig2.pdf")


def plot_fig3():
    kin_df, ann_df = load_preprocessed_pkl(feature_type='quat',pkl_path=constants.pkl_path,pp_funcs = ['lp','dec','std'],pp_func_params={'lp':1.5,'dec':3,'std':None})
    kin_df = kin_df[constants.slave_cols_quat]
    index = ["Knot_Tying","E003"]
    plot_results.plot_gt_px_map_seg_v_time(kin_df.loc(axis=0)[index[0],index[1]],ann_df.loc(axis=0)[index[0],index[1]],index,constants.paper_savefolder,"fig3a.pdf" )
    index = ["Suturing", "E003"]
    plot_results.plot_gt_px_map_seg_v_time(kin_df.loc(axis=0)[index[0],index[1]],ann_df.loc(axis=0)[index[0],index[1]],index,constants.paper_savefolder,"fig3b.pdf" )
    index = ["Needle_Passing", "E003"]
    plot_results.plot_gt_px_map_seg_v_time(kin_df.loc(axis=0)[index[0],index[1]],ann_df.loc(axis=0)[index[0],index[1]],index,constants.paper_savefolder,"fig3c.pdf" )


# Feature Selection/PCA Results
def plot_fig4():
    #
    gmm_dp_metrics_df = pd.DataFrame()
    for feat_key in constants.feat_columns_map.keys():
        gmm_dp_metrics_df = gmm_dp_metrics_df.append(load_feat_pca_metrics("gmm_dp",feat_key))
        gmm_dp_metrics_df = gmm_dp_metrics_df.append(load_feat_basic_metrics("gmm_dp",feat_key))


    plot_results.plot_feature_pca_eval(gmm_dp_metrics_df,constants.paper_savefolder+"fig4a.pdf")

    feat_key = "pos_eedist"
    gmm_dp_metrics_df = pd.DataFrame()
    gmm_dp_metrics_df = gmm_dp_metrics_df.append(load_feat_pca_metrics("gmm_dp", feat_key))
    gmm_dp_metrics_df = gmm_dp_metrics_df.append(load_feat_basic_metrics("gmm_dp", feat_key))
    gmm_dp_metrics_df = gmm_dp_metrics_df.append(load_feat_window_metrics("gmm_dp", feat_key))
    plot_results.plot_window_pca_eval(gmm_dp_metrics_df, constants.paper_savefolder + "fig4b.pdf")

def plot_fig5():
    features = ["pos_eedist_win3_kpca12"]#,"pos_eedist_win3_kpca12","pos_eedist_win10_kpca10","pos_eedist_win10_kpca12"]
    seg_funcs = ['gmm_dp','gmm_bic','gmm_fixed','hdp_hsmm']
    gt_seg_df = pd.DataFrame()
    pred_seg_dfs = {}
    feat_dfs= {}
    metrics_dfs = pd.DataFrame()
    seg_func = "hdp_hsmm"
    best_trials = {}
    for task in constants.tasks:
        gt_seg_df = gt_seg_df.append(pd.read_pickle(os.path.join(constants.results_path,"data","annotations","annotations_"+task+".pkl")))
        demo_inds = pd.Index(np.unique(gt_seg_df.index.droplevel(level=2)))
        for feat in features:
            feat_filename = "features_" + task + "_" + feat + ".pkl"
            feat_dfs[task + "_" + feat] = pd.read_pickle(os.path.join(constants.results_path, "data", "features", feat_filename))
            for seg_func in seg_funcs:
                metrics_dfs = metrics_dfs.append( load_metrics([task,feat,seg_func]))
                best_cv = int(metrics_dfs.iloc[-1,-1])
                seg_filename = "segmentations_"+task+"_"+feat+"_"+seg_func+"_"+"cv"+str(best_cv)+".pkl"
                pred_seg_dfs[task+"_"+feat+"_"+seg_func] = pd.read_pickle(os.path.join(constants.results_path,"data","segmentations",seg_filename))
                demo_inds = demo_inds.intersection(pd.Index(np.unique(pred_seg_dfs[task+"_"+feat+"_"+seg_func].index.droplevel(level=2))))


        for feat in features:
            for seg_func in seg_funcs:
                best_trials[task+"_"+feat+"_"+seg_func] = get_best_trial(pred_seg_dfs[task+"_"+feat+"_"+seg_func] ,gt_seg_df,demo_inds)

    # plot knot tying results
    savepaths = [os.path.join(constants.results_path,"paper","fig5a.pdf"),os.path.join(constants.results_path,"paper","fig5b.pdf"),os.path.join(constants.results_path,"paper","fig5c.pdf")]
    plot_results.plot_seg_results(feat_dfs,gt_seg_df,pred_seg_dfs,best_trials,savepaths)



def get_best_trial(pred_seg_df,gt_seg_df,demo_inds):


    best_ind = demo_inds[0]
    best_score = 0
    for ind in demo_inds:
        pred_labels = metrics.convert_segments_to_labels(pred_seg_df.loc(axis=0)[ind[0],ind[1]])
        gt_labels = metrics.convert_segments_to_labels(gt_seg_df.loc(axis=0)[ind[0],ind[1]])
        if(len(pred_labels) < len(gt_labels)):
            pred_labels += [pred_labels[-1]]*(len(gt_labels)-len(pred_labels) )
        score = metrics.munkres_score(gt_labels,pred_labels)
        if score > best_score:
            best_score = score
            best_ind = ind

    return best_ind


if __name__ == "__main__":
    # Feature Selection Cross Validation
    # run_feature_cv_within_task_eval("gmm_dp")
    # run_feature_cv_within_task_eval("gmm_gt")
    #
    # # Segmentation Cross Val Experiments (Note: this takes a very long time, specifically for the hmm)
    # run_gmm_cv()
    # run_hsmm_cv()

    # generate figs for report
    plot_fig1()
    plot_fig2()
    plot_fig3() # not used
    plot_fig4() # so this is really 3
    plot_fig5() # and this is really 4

    # save metrics for table to human readable format
    export_metrics_to_csv(constants.results_path + "paper/metrics_seg_cv.csv",["pos_eedist_win3_kpca12"])


